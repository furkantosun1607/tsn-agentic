"""
TSN Media — LangGraph Orchestration Layer  (AI Orchestration Layer)
===================================================================
Mimari şemadaki elmas (◆) şeklindeki "AI Orchestration Layer" kutusunun
tam Python implementasyonu.

Bu dosya sistemin kalbidir:
    - ArticleState: Bir haberin pipeline boyunca taşıdığı tüm veri
    - Node'lar: Her agent/adım bir Python fonksiyonu
    - Edge'ler: Node'lar arası geçiş kuralları (koşullu veya sabit)
    - tsn_graph: Derlenmiş, çalıştırılmaya hazır StateGraph

Teknoloji Stratejisi
--------------------
| Bileşen                          | Teknoloji         | Gerekçe                          |
|----------------------------------|-------------------|----------------------------------|
| Orchestration (bu dosya)         | LangGraph         | Conditional routing, state mgmt  |
| Scoring / Categorization / Özet  | CrewAI            | Multi-tool agent reasoning       |
| Personalization                  | LangChain LCEL    | Tek adım; Crew ağırlığı yok      |
| Voice / Video                    | Stub → Phase 2    | Graph yapısı hazır tutulur       |
| Gözlemleme                       | LangSmith         | Tüm sistemi otomatik trace eder  |

LangSmith Entegrasyonu
----------------------
LANGCHAIN_TRACING_V2=true ortam değişkeni set edildiğinde,
tsn_graph.invoke() çağrısı içindeki TÜM node'lar (CrewAI, LCEL, stub)
otomatik olarak LangSmith'e kaydedilir. Ek kod gerekmez.

Graph Akışı
-----------
START
  ↓
[ingest_article]  → Ham metni doğrula, state hazırla
  ↓
[scoring_node]    → CrewAI scoring_crew() | SaveQualityScoreTool → DB'ye yazar
  ↓
[route_by_score]  → CONDITIONAL EDGE (threshold: SCORE_THRESHOLD env var)
  ↙                              ↘
(score < threshold)         (score >= threshold)
[discard_node]              [categorize_and_summarize_node]
  ↓                               ↓
  └──── her ikisi de ─────→ [voice_synthesis_node]  (stub)
                                   ↓
                            [video_generation_node] (stub)
                                   ↓
                            [personalization_node]  LangChain LCEL
                                   ↓
                            [save_results_node]     DB final update
                                   ↓
                                  END

NOT: discard_node → personalization_node bağlantısı şemaya sadıktır.
     Reddedilen haberler negatif sinyal olarak personalizasyona beslenir.
"""

from __future__ import annotations

import logging
import os
import sys
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
    content: str
    available_categories: str
    score_threshold: int          # SCORE_THRESHOLD env'den gelir

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
        if article and article.categories:
            return [cat.name for cat in article.categories]
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
# NODE 2 — scoring_node
# CrewAI scoring_crew() çalıştırır.
# SaveQualityScoreTool puanı DB'ye yazar; node ardından DB'den okur.
# score >= threshold → "approved", aksi → "discarded"
# ===========================================================================

def scoring_node(state: ArticleState) -> ArticleState:
    """
    CrewAI Scoring Agent'ı çalıştırır.
    Şemadaki "Scoring / Evaluation Agent" kutusuna karşılık gelir.
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
    Puanlama sonucuna göre pipeline'ı yönlendirir.
    Bu fonksiyon LangGraph'ın koşullu kenarı (conditional edge) olarak kullanılır.

    - "approved"  → Kategorizasyon + Özetleme → Media → Personalizasyon
    - Diğer (discarded/error) → Discard → Personalizasyon (negatif sinyal)
    """
    if state.get("status") == "approved":
        return "categorize_and_summarize_node"
    return "discard_node"


# ===========================================================================
# NODE 3a — categorize_and_summarize_node
# Sadece onaylanan haberler için.
# CrewAI categorize_summary_crew() çalıştırır (quality_score ile).
# ===========================================================================

def categorize_and_summarize_node(state: ArticleState) -> ArticleState:
    """
    Kategorizasyon ve özetleme işlemlerini sırayla yürütür.
    Şemadaki "Categorization Agent" + "Summarization Agent" kutularına karşılık gelir.

    categorize_summary_crew(quality_score=...) — crew içinde db_write araçları çalışır:
        • UpdateScoreAndCategoriesTool → DB'ye kategorileri yazar
        • UpdateSummaryTool → DB'ye özeti yazar
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
# NODE 3b — discard_node
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
    }


# ===========================================================================
# NODE 4 — personalization_node (LangChain LCEL)
# Hem onaylanan hem reddedilen haberler buraya gelir.
# Şemada her iki yoldan da Personalization Agent'a ok var.
# ===========================================================================

def personalization_node(state: ArticleState) -> ArticleState:
    """
    LangChain LCEL Personalization zincirini çalıştırır.

    - Onaylanan haberler → pozitif kullanıcı segmenti etiketleri
    - Reddedilen haberler → negatif sinyal etiketleri (filtre geri bildirimi)

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
# NODE 5 — save_results_node
# Pipeline'ın son adımı: durumu loglayıp final olarak işaretler.
# İleride Cloud Storage URL'lerini backend'e bildirim olarak gönderebilir.
# ===========================================================================

def save_results_node(state: ArticleState) -> ArticleState:
    """
    Pipeline tamamlandı. Final durumu loglar.
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

    return {**state, "status": "completed"}


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
    graph.add_node("scoring_node", scoring_node)
    graph.add_node("discard_node", discard_node)
    graph.add_node("categorize_and_summarize_node", categorize_and_summarize_node)
    graph.add_node("voice_synthesis_node", voice_synthesis_node)
    graph.add_node("video_generation_node", video_generation_node)
    graph.add_node("personalization_node", personalization_node)
    graph.add_node("save_results_node", save_results_node)

    # -- Giriş noktası -------------------------------------------------------
    graph.set_entry_point("ingest_article")

    # -- Sabit edge: ingest → scoring ----------------------------------------
    # (Her haber scoring'e gider; çok kısa içerikler ingest'te zaten discard edilir)
    graph.add_edge("ingest_article", "scoring_node")

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
    # Şemada discard'dan da personalization'a ok var (negatif sinyal)
    graph.add_edge("discard_node", "personalization_node")

    # -- Sabit edge: personalization → save → END ----------------------------
    graph.add_edge("personalization_node", "save_results_node")
    graph.add_edge("save_results_node", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Modül seviyesinde derlenen graph — import edildiğinde kullanıma hazır
# ---------------------------------------------------------------------------
from langgraphics import watch
tsn_graph = watch(build_graph())
logger.info("tsn_graph derlendi ve hazır.")
