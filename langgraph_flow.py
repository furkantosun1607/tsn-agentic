"""
TSN Media — LangGraph Orchestration Layer (AI Orchestration Layer)
==================================================================
Bu dosya sistemin karar mekanizmasidir.

demo_simulation.py ile birebir zihinsel harita:
1) ingest_article
   RSS'ten veya DB'den gelen ham haber kaydını doğrular.
2) mcp_browser_node
   Demo'daki "[MCP Tool] Initiating Browser Context..." satırının production
   karşılığıdır. URL varsa canlı sayfadan metin zenginleştirmesi yapabilir.
3) scoring_node
   Demo'daki "[LangChain Node] Sending content to Local Ollama..." adımıdır.
4) route_by_score
   Demo'daki "SCORE < 50 -> REJECTED" ve "SCORE >= 50 -> APPROVED" ayrımıdır.
5) categorize_and_summarize_node
   Demo'daki CrewAI orkestrasyonudur: kategori seçimi ve TL;DR üretimi.
6) voice_synthesis_node / video_generation_node
   Demo'daki ses üretimi adımının production node'larıdır. Video şimdilik
   genişletme noktasıdır.
7) personalization_node
   Demo'da gösterilmeyen ama backend ürünleşme tarafında kullanılan ek
   segmentasyon adımıdır; onaylanan haberden kullanıcı etiketleri üretir.
8) save_results_node
   Terminaldeki "PIPELINE COMPLETED" satırının production karşılığıdır.

Graph akisi:
START -> ingest_article -> route_after_ingest
ready -> mcp_browser_node -> scoring_node -> route_by_score
approved -> categorize_and_summarize_node -> voice_synthesis_node ->
video_generation_node -> personalization_node -> save_results_node -> END
discard/error -> discard_node -> save_results_node -> END
"""

from __future__ import annotations

import logging
import os
import sys
import asyncio
from typing import Literal

from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

# Path setup — backend ve proje içe aktarılabilir olmalı
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_backend_dir = os.path.join(_project_root, "backend")
for _p in [_project_root, _backend_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ai_workers.crew import TsnMediaCrew
from ai_workers.media_nodes import video_generation_node, voice_synthesis_node
from ai_workers.personalization_chain import run_personalization_chain

logger = logging.getLogger("langgraph_flow")

# Threshold: ortam değişkeninden okunur, varsayılan 50
SCORE_THRESHOLD = int(os.environ.get("SCORE_THRESHOLD", "50"))
MCP_BROWSER_ENABLED = os.environ.get("MCP_BROWSER_ENABLED", "false").lower() == "true"
MCP_BROWSER_TEXT_LIMIT = int(os.environ.get("MCP_BROWSER_TEXT_LIMIT", "2000"))

logger.info("LangGraph flow yüklendi | SCORE_THRESHOLD=%d", SCORE_THRESHOLD)


# ===========================================================================
# STATE — ArticleState
# Bir haberin pipeline boyunca taşıdığı tüm veri yapısı.
# Her LangGraph node bu dict'i okur ve kısmen günceller ({**state, key: val}).
# ===========================================================================

class ArticleState(TypedDict):
    # -- Girdi (main_langgraph.py tarafından doldurulur) --------------------
    article_id: int
    title: str
    source_url: str | None
    content: str
    available_categories: str
    score_threshold: int          # SCORE_THRESHOLD env'den gelir

    # -- MCP Browser çıktısı ------------------------------------------------
    mcp_excerpt: str | None       # Canlı URL okumasından gelen kısa metin

    # -- CrewAI Çıktıları ---------------------------------------------------
    quality_score: int | None     # scoring_node tarafından doldurulur
    categories: list[str]         # categorize_and_summarize_node → DB'den
    summary: str | None           # categorize_and_summarize_node → DB'den

    # -- Media URLs (Phase 2 stub'ları) -------------------------------------
    audio_url: str | None         # voice_synthesis_node
    video_url: str | None         # video_generation_node

    # -- Personalization (LangChain LCEL) -----------------------------------
    personalization_tags: list[str]

    # -- Flow Kontrol -------------------------------------------------------
    status: str         # pending | approved | discarded | completed | error
    error_message: str | None
    retry_count: int


# ===========================================================================
# YARDIMCI FONKSİYONLAR
# ===========================================================================

def _read_score_from_db(article_id: int) -> int:
    """
    Scoring crew çalıştıktan sonra DB'den puanı okur.
    SaveQualityScoreTool puanı doğrudan DB'ye yazdığı için
    CrewAI çıktısını parse etmek yerine DB'yi sorgulamak daha güvenilirdir.
    """
    from app.core.database import SessionLocal
    from app.models.article import Article

    session = SessionLocal()
    try:
        article = session.query(Article).filter(Article.id == article_id).first()
        if article and article.ai_quality_score is not None:
            return int(article.ai_quality_score)
        logger.warning(
            "[Flow] _read_score_from_db → ID=%d için puan DB'de bulunamadı, 0 döndürülüyor.",
            article_id,
        )
        return 0
    except Exception as exc:
        logger.error("[Flow] DB'den puan okunurken hata: %s", exc)
        return 0
    finally:
        session.close()


def _read_categories_from_db(article_id: int) -> list[str]:
    """Kategorize adımından sonra DB'den kategori adlarını okur."""
    from app.core.database import SessionLocal
    from app.models.article import Article

    session = SessionLocal()
    try:
        article = session.query(Article).filter(Article.id == article_id).first()
        # Article modelinde iki farklı kategori ilişkisi bulunur:
        #   category      -> RSS kaynağından gelen tek "varsayılan" kategori
        #   ai_categories -> CrewAI'nin article_categories junction tablosuna
        #                    yazdığı çoklu AI kategorileri
        #
        # Demo akışında "Assigned Categories: ['Gündem']" olarak görünen çıktı
        # CrewAI tarafından üretilen sınıflandırmadır; bu yüzden burada
        # article.ai_categories okunmalıdır.
        if article and article.ai_categories:
            return [cat.name for cat in article.ai_categories]
        return []
    except Exception as exc:
        logger.error("[Flow] DB'den kategoriler okunurken hata: %s", exc)
        return []
    finally:
        session.close()


def _read_summary_from_db(article_id: int) -> str | None:
    """Özetleme adımından sonra DB'den özeti okur."""
    from app.core.database import SessionLocal
    from app.models.article import Article

    session = SessionLocal()
    try:
        article = session.query(Article).filter(Article.id == article_id).first()
        return article.summary if article else None
    except Exception as exc:
        logger.error("[Flow] DB'den özet okunurken hata: %s", exc)
        return None
    finally:
        session.close()


# ===========================================================================
# NODE 1 — ingest_article
# Ham haberi doğrular ve state'i hazırlar.
# Çok kısa içerikler CrewAI'ya gitmeden burada elenir (token tasarrufu).
# ===========================================================================

def ingest_article(state: ArticleState) -> ArticleState:
    """
    Pipeline'ın ilk durağı. Ham haber metnini kabul edip doğrular.
    İçerik yeterince uzunsa "pending" → "ready" yapar.
    Yeterli değilse hemen "discarded" işaret eder; scoring'e gitmez.
    """
    article_id = state["article_id"]
    content_len = len((state.get("content") or "").strip())
    title = (state.get("title") or "").strip()

    logger.info(
        "[ingest_article] ID=%d | Başlık: %.60s | İçerik uzunluğu: %d",
        article_id, title, content_len,
    )

    if not title:
        return {
            **state,
            "status": "discarded",
            "error_message": "Başlık eksik — haber işleme alınamaz.",
        }

    if content_len < 50:
        return {
            **state,
            "status": "discarded",
            "error_message": f"İçerik çok kısa ({content_len} karakter < 50 eşiği).",
        }

    logger.info("[ingest_article] ID=%d ✓ Doğrulama geçti.", article_id)
    return {**state, "status": "pending"}


# ===========================================================================
# CONDITIONAL EDGE — route_after_ingest
# demo_simulation.py RSS'ten veri alamazsa "continue" ile bir sonraki URL'e geçer.
# Production graph'ta aynı fikri burada uyguluyoruz: eksik/geçersiz haberler
# scoring'e hiç gitmez, doğrudan discard_node'a yönlenir.
# ===========================================================================

def route_after_ingest(
    state: ArticleState,
) -> Literal["mcp_browser_node", "discard_node"]:
    """
    ingest_article sonucuna göre ilk yol ayrımını yapar.

    Bu fonksiyon özellikle maliyet kontrolü için önemlidir: başlığı olmayan veya
    içeriği çok kısa olan haberleri LLM'e göndermek hem token israfı yaratır hem
    de demo akışındaki "item yoksa atla" davranışına ters düşer.
    """
    if state.get("status") == "discarded":
        return "discard_node"
    return "mcp_browser_node"


# ===========================================================================
# NODE 2 — mcp_browser_node
# Demo'daki canlı tarayıcı adımının production karşılığı.
# ===========================================================================

async def _read_url_text_with_playwright(url: str) -> str:
    """
    Playwright ile haber URL'ini açıp sayfanın görünen metninden kısa bir bölüm
    döndürür.

    Bu yardımcı fonksiyon bilerek küçük tutuldu:
    - MCP server ayrı bir süreç olarak da çalışabilir.
    - LangGraph node'u ise aynı davranışı doğrudan pipeline içinde göstermek
      için bu fonksiyonu kullanabilir.
    - Kodun bu görevde çalıştırılması istenmediği için entegrasyon noktası
      ayrıntılı yorumla bırakıldı; runtime ortamında Playwright kurulu değilse
      node güvenli şekilde mevcut DB içeriğiyle devam eder.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            await page.evaluate("window.scrollBy(0, 700)")
            await page.wait_for_timeout(750)
            text = await page.evaluate("document.body.innerText")
            return (text or "").strip()
        finally:
            await browser.close()


def mcp_browser_node(state: ArticleState) -> ArticleState:
    """
    Canlı haber URL'inden ek metin okumaya çalışan MCP/Browser düğümü.

    demo_simulation.py içinde bu adım görsel amaçla gerçek Chrome penceresini
    açıp birkaç saniye sonra kapatır. Production worker'da aynı davranışın
    daha kontrollü hali kullanılır:
        - URL yoksa mevcut DB içeriğiyle devam edilir.
        - MCP_BROWSER_ENABLED=false ise tarayıcı açılmaz; demo eşleşmesi log ve
          state seviyesinde korunur.
        - MCP_BROWSER_ENABLED=true ve Playwright hazırsa sayfa metni okunur.
        - Okunan metin content'in önüne değil, sonuna eklenir; DB'den gelen
          normalize edilmiş metin önceliğini kaybetmez.

    Böylece bu node hem sunumda anlatılan MCP Browser adımını temsil eder hem de
    tarayıcı bağımlılığı olmadığı ortamlarda pipeline'ı kırmaz.
    """
    article_id = state["article_id"]
    source_url = state.get("source_url")

    logger.info(
        "[mcp_browser_node] ID=%d | Browser context hazırlanıyor | URL=%s",
        article_id,
        source_url or "yok",
    )

    if not source_url:
        logger.info(
            "[mcp_browser_node] ID=%d | URL bulunmadı; DB içeriği kullanılacak.",
            article_id,
        )
        return {**state, "mcp_excerpt": None}

    if not MCP_BROWSER_ENABLED:
        logger.info(
            "[mcp_browser_node] ID=%d | MCP_BROWSER_ENABLED=false; "
            "demo akışı korunuyor, canlı tarayıcı okuması atlandı.",
            article_id,
        )
        return {**state, "mcp_excerpt": None}

    try:
        live_text = asyncio.run(_read_url_text_with_playwright(source_url))
    except Exception as exc:
        logger.warning(
            "[mcp_browser_node] ID=%d | Tarayıcı okuması başarısız: %s. "
            "Mevcut DB içeriğiyle devam ediliyor.",
            article_id,
            exc,
        )
        return {**state, "mcp_excerpt": None}

    excerpt = live_text[:MCP_BROWSER_TEXT_LIMIT]
    merged_content = "\n\n[Canlı sayfadan MCP Browser ile okunan ek metin]\n" + excerpt

    logger.info(
        "[mcp_browser_node] ID=%d | Canlı içerik okundu | Ek metin uzunluğu=%d",
        article_id,
        len(excerpt),
    )

    return {
        **state,
        "content": ((state.get("content") or "") + merged_content)[:4000],
        "mcp_excerpt": excerpt,
    }


# ===========================================================================
# NODE 3 — scoring_node
# CrewAI scoring_crew() çalıştırır.
# SaveQualityScoreTool puanı DB'ye yazar; node ardından DB'den okur.
# score >= threshold → "approved", aksi → "discarded"
# ===========================================================================

def scoring_node(state: ArticleState) -> ArticleState:
    """
    CrewAI Scoring Agent'ı çalıştırır. (Veya LangChain doğrudan LLM çağrısı)
    SUNUMDAKİ YERİ: Terminalde sarı renkte dönen 
    "[LangChain Node] Sending content to Local Ollama for AI Analysis..." adımıdır.
    Burada haberin 1-100 arası kalite skoru atanır.
    """
    article_id = state["article_id"]
    threshold = state.get("score_threshold", SCORE_THRESHOLD)

    logger.info(
        "[scoring_node] ID=%d | CrewAI scoring_crew başlatılıyor... "
        "(eşik: %d)",
        article_id,
        threshold,
    )

    inputs = {
        "article_id": article_id,
        "title": state["title"],
        "content": state["content"],
        "available_categories": state["available_categories"],
    }

    try:
        TsnMediaCrew().scoring_crew().kickoff(inputs=inputs)
    except Exception as exc:
        logger.error(
            "[scoring_node] ID=%d | CrewAI hatası: %s", article_id, exc
        )
        return {
            **state,
            "status": "error",
            "error_message": f"scoring_crew hatası: {exc}",
            "quality_score": 0,
        }

    # SaveQualityScoreTool DB'ye yazdı; şimdi okuyoruz
    score = _read_score_from_db(article_id)
    new_status = "approved" if score >= threshold else "discarded"

    logger.info(
        "[scoring_node] ID=%d | Puan=%d | Eşik=%d | Karar=%s",
        article_id, score, threshold, new_status.upper(),
    )

    return {**state, "quality_score": score, "status": new_status}


# ===========================================================================
# CONDITIONAL EDGE — route_by_score
# Şemadaki "Reject Low Quality" / "Approve High Quality" ayrımı.
# ===========================================================================

def route_by_score(
    state: ArticleState,
) -> Literal["categorize_and_summarize_node", "discard_node"]:
    """
    Puanlama sonucuna göre pipeline'ı yönlendirir (Conditional Edge).
    SUNUMDAKİ YERİ: Terminaldeki "SCORE < 50 (THRESHOLD). Content REJECTED."
    veya "SCORE >= 50. Content APPROVED." ayrımının teknik olarak yapıldığı yerdir.
    
    Bu fonksiyon bir yapay zeka ajanı DEĞİL, LangGraph'ın yol ayrımı (switch-case) mantığıdır.
    """
    if state.get("status") == "approved":
        return "categorize_and_summarize_node"
    return "discard_node"


# ===========================================================================
# NODE 4a — categorize_and_summarize_node
# Sadece onaylanan haberler için.
# CrewAI categorize_summary_crew() çalıştırır (quality_score ile).
# ===========================================================================

def categorize_and_summarize_node(state: ArticleState) -> ArticleState:
    """
    Kategorizasyon ve özetleme işlemlerini sırayla yürütür.
    SUNUMDAKİ YERİ: Terminalde mor renkte görünen "[CrewAI Orchestrator] Delegating tasks..." adımı.
    
    Bu aşamada 2 farklı ajan (Categorization ve Summarization) sıralı (sequential) olarak çalıştırılır.
    - @Categorization_Agent haber metnine bakar ve uygun kategoriyi seçer.
    - @Summarization_Agent haberi maddeler halinde özetler (TL;DR).
    """
    article_id = state["article_id"]
    score = state.get("quality_score") or 0

    logger.info(
        "[cat_sum_node] ID=%d | CrewAI categorize_summary_crew başlatılıyor... "
        "(puan: %d)",
        article_id,
        score,
    )

    inputs = {
        "article_id": article_id,
        "title": state["title"],
        "content": state["content"],
        "available_categories": state["available_categories"],
    }

    try:
        TsnMediaCrew().categorize_summary_crew(quality_score=score).kickoff(
            inputs=inputs
        )
    except Exception as exc:
        logger.error(
            "[cat_sum_node] ID=%d | CrewAI hatası: %s", article_id, exc
        )
        return {
            **state,
            "status": "error",
            "error_message": f"categorize_summary_crew hatası: {exc}",
        }

    # Araçlar DB'ye yazdı; state'e okuyoruz
    categories = _read_categories_from_db(article_id)
    summary = _read_summary_from_db(article_id)

    logger.info(
        "[cat_sum_node] ID=%d ✓ | Kategoriler=%s | Özet uzunluğu=%d",
        article_id,
        categories,
        len(summary or ""),
    )

    return {
        **state,
        "categories": categories,
        "summary": summary,
        "status": "approved",
    }


# ===========================================================================
# NODE 4b — discard_node
# Düşük kaliteli veya geçersiz haberler için.
# Şemadaki "Discard" kutusuna karşılık gelir.
# ===========================================================================

def discard_node(state: ArticleState) -> ArticleState:
    """
    Düşük puanlı veya geçersiz haberleri işaretler.
    Şemada bu node'dan da personalization'a ok var — negatif sinyal.
    """
    article_id = state["article_id"]
    reason = state.get("error_message") or (
        f"Kalite puanı eşiğin altında "
        f"({state.get('quality_score', '?')} < {state.get('score_threshold', SCORE_THRESHOLD)})"
    )

    logger.warning(
        "[discard_node] ID=%d | Haber REDDEDİLDİ. Neden: %s",
        article_id,
        reason,
    )

    return {
        **state,
        "status": "discarded",
        "categories": [],
        "summary": None,
        "audio_url": None,
        "video_url": None,
    }


# ===========================================================================
# NODE 7 — personalization_node (LangChain LCEL)
# Hem onaylanan hem reddedilen haberler buraya gelir.
# Şemada her iki yoldan da Personalization Agent'a ok var.
# ===========================================================================

def personalization_node(state: ArticleState) -> ArticleState:
    """
    LangChain LCEL Personalization zincirini çalıştırır.

    Demo hizalamasından sonra bu node yalnızca onaylanan haberlerde çalışır.
    Çünkü demo_simulation.py düşük puanlı haberlerde kategori/özet/ses gibi
    pahalı adımları çalıştırmadan sıradaki URL'e geçer.

    - Onaylanan haberler → pozitif kullanıcı segmenti etiketleri
    - Reddedilen haberler → discard_node üzerinden doğrudan save_results_node'a
      gider; burada personalization çalıştırılmaz.

    LCEL zinciri: ChatPromptTemplate | ChatGoogleGenerativeAI.with_structured_output()
    LangSmith LANGCHAIN_TRACING_V2=true ise bu zinciri de otomatik izler.
    """
    article_id = state["article_id"]
    status = state.get("status", "discarded")

    logger.info(
        "[personalization_node] ID=%d | LangChain LCEL zinciri çalıştırılıyor...",
        article_id,
    )

    try:
        result = run_personalization_chain(
            article_id=article_id,
            title=state["title"],
            status=status,
            categories=state.get("categories") or [],
            summary=state.get("summary"),
        )
        tags = result.tags
    except Exception as exc:
        logger.error(
            "[personalization_node] ID=%d | Personalizasyon hatası: %s",
            article_id,
            exc,
        )
        tags = []

    return {**state, "personalization_tags": tags}


# ===========================================================================
# NODE 8 — save_results_node
# Pipeline'ın son adımı: durumu loglayıp final olarak işaretler.
# İleride Cloud Storage URL'lerini backend'e bildirim olarak gönderebilir.
# ===========================================================================

def save_results_node(state: ArticleState) -> ArticleState:
    """
    Pipeline tamamlandı. Final durumu loglar.

    Önemli ayrım:
        - Onaylanan haberlerde buraya kategori, özet, ses/video ve
          personalization çıktılarıyla gelinir.
        - Reddedilen haberlerde demo_simulation.py davranışına uygun olarak
          kategori/özet/medya adımları çalışmaz; save_results_node sadece
          reddetme kararını terminal/log seviyesinde görünür kılar.

    Phase 2: audio_url / video_url'i backend API'ye POST edebilir.
    """
    article_id = state["article_id"]
    status = state.get("status", "unknown")
    score = state.get("quality_score", "—")
    tags = state.get("personalization_tags", [])
    audio = state.get("audio_url") or "—"
    video = state.get("video_url") or "—"

    logger.info(
        "[save_results_node] ID=%d ✓ TAMAMLANDI\n"
        "  Durum       : %s\n"
        "  Kalite Puanı: %s\n"
        "  Kategoriler : %s\n"
        "  Etiketler   : %s\n"
        "  Ses URL     : %s\n"
        "  Video URL   : %s",
        article_id,
        status.upper(),
        score,
        state.get("categories", []),
        tags,
        audio,
        video,
    )

    # status alanını "completed" yapmıyoruz; çünkü demo akışında düşük puanlı
    # haber "tamamlandı" değil, "reddedildi ve sıradaki URL'e geçildi" olarak
    # görünür. main_langgraph.py istatistiklerinin completed/discarded ayrımını
    # doğru görebilmesi için semantic status korunur.
    return state


# ===========================================================================
# GRAPH ASSEMBLY
# StateGraph'a node'lar ve edge'ler eklenir, ardından derlenir.
# ===========================================================================

def build_graph() -> StateGraph:
    """
    TSN Media LangGraph pipeline'ını derleyip döndürür.
    main_langgraph.py bu fonksiyonun döndürdüğü graph'ı .invoke() ile çalıştırır.
    """
    graph = StateGraph(ArticleState)

    # -- Node'ları ekle ------------------------------------------------------
    graph.add_node("ingest_article", ingest_article)
    graph.add_node("mcp_browser_node", mcp_browser_node)
    graph.add_node("scoring_node", scoring_node)
    graph.add_node("discard_node", discard_node)
    graph.add_node("categorize_and_summarize_node", categorize_and_summarize_node)
    graph.add_node("voice_synthesis_node", voice_synthesis_node)
    graph.add_node("video_generation_node", video_generation_node)
    graph.add_node("personalization_node", personalization_node)
    graph.add_node("save_results_node", save_results_node)

    # -- Giriş noktası -------------------------------------------------------
    graph.set_entry_point("ingest_article")

    # -- Koşullu edge: ingest sonrası ilk kalite kapısı ----------------------
    # demo_simulation.py içinde RSS item alınamazsa döngü "continue" eder.
    # Burada geçersiz haberler LLM'e gitmeden discard_node'a düşer.
    graph.add_conditional_edges(
        "ingest_article",
        route_after_ingest,
        {
            "mcp_browser_node": "mcp_browser_node",
            "discard_node": "discard_node",
        },
    )

    # -- Sabit edge: MCP Browser → scoring ----------------------------------
    # Canlı sayfa okuması tamamlandıktan veya kontrollü şekilde atlandıktan
    # sonra haber artık demo'daki LangChain/Ollama scoring adımına hazırdır.
    graph.add_edge("mcp_browser_node", "scoring_node")

    # -- Koşullu edge: scoring sonrası yön tayini ----------------------------
    # route_by_score() → "approved"  → categorize_and_summarize_node
    #                  → diğerleri  → discard_node
    graph.add_conditional_edges(
        "scoring_node",
        route_by_score,
        {
            "categorize_and_summarize_node": "categorize_and_summarize_node",
            "discard_node": "discard_node",
        },
    )

    # -- Sabit edge'ler: onaylanan haberler yolu -----------------------------
    graph.add_edge("categorize_and_summarize_node", "voice_synthesis_node")
    graph.add_edge("voice_synthesis_node", "video_generation_node")
    graph.add_edge("video_generation_node", "personalization_node")

    # -- Sabit edge: reddedilen haberler yolu --------------------------------
    # demo_simulation.py düşük puanlı haberde kategori/özet/ses adımlarına
    # girmeden "Skipping to next URL..." der. Bu yüzden discard doğrudan final
    # log node'una bağlanır.
    graph.add_edge("discard_node", "save_results_node")

    # -- Sabit edge: personalization → save → END ----------------------------
    graph.add_edge("personalization_node", "save_results_node")
    graph.add_edge("save_results_node", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Modül seviyesinde derlenen graph — import edildiğinde kullanıma hazır
# ---------------------------------------------------------------------------
tsn_graph = build_graph()
logger.info("tsn_graph derlendi ve hazır.")
