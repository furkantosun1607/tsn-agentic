"""
TSN Media AI Worker — LangGraph Entry Point
============================================
Bu dosya, mevcut main.py'nin LangGraph tabanlı alternatifidir.
Mevcut main.py KORUNUR ve hâlâ tek başına çalışabilir.

Farklar:
    main.py            → Düz for döngüsü, CrewAI sequential (3 görev birden)
    main_langgraph.py  → LangGraph state machine, koşullu dallanma,
                         LangSmith tracing, discard filtresi

Çalıştırma:
    # Kök dizinden (tsn_v0/)
    python -m ai_workers.main_langgraph

LangSmith Takibi:
    backend/.env dosyasındaki LANGCHAIN_API_KEY tanımlıysa,
    her haber için ayrı bir trace otomatik oluşturulur.
    Dashboard: https://smith.langchain.com → Proje: tsn-media-ai-pipeline
"""

import logging
import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# Path setup — main.py ile birebir aynı
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_backend_dir = os.path.join(_project_root, "backend")

for _p in [_project_root, _backend_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv

# .env yükle — LangSmith değişkenleri burada okunur
load_dotenv(os.path.join(_backend_dir, ".env"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai_worker_langgraph")

# ---------------------------------------------------------------------------
# LangSmith durum kontrolü
# ---------------------------------------------------------------------------
_LANGSMITH_VARS = [
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
]

def _check_langsmith_config() -> bool:
    """LangSmith ortam değişkenlerini kontrol eder, eksikleri loglar."""
    ok = True
    for var in _LANGSMITH_VARS:
        val = os.environ.get(var, "")
        if not val or val.startswith("<"):
            logger.warning(
                "⚠ LangSmith değişkeni eksik/geçersiz: %s=%r "
                "(backend/.env dosyasını kontrol et)",
                var, val,
            )
            ok = False
    return ok

# ---------------------------------------------------------------------------
# Gerçek importlar (.env yüklendikten sonra yapılmalı)
# ---------------------------------------------------------------------------
from sqlalchemy import or_

from app.core.database import SessionLocal
from app.models.article import Article
from app.models.category import Category
from ai_workers.langgraph_flow import SCORE_THRESHOLD, ArticleState, tsn_graph

# Rate-limit parametreleri (main.py ile aynı)
_MIN_INTERVAL_SEC = float(os.environ.get("AI_WORKER_MIN_INTERVAL_SEC", "20"))
_RATE_LIMIT_BACKOFF_SEC = float(os.environ.get("AI_WORKER_RATE_LIMIT_BACKOFF_SEC", "45"))


# ---------------------------------------------------------------------------
# Yardımcı: Rate-limit algılama (main.py'den alındı)
# ---------------------------------------------------------------------------

def _parse_retry_after_seconds(message: str) -> float | None:
    m = re.search(r"retry\s*(?:in|after)?\s*[:\s]*(\d+)\s*s", message, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*second", message, re.I)
    if m:
        return float(m.group(1))
    return None


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        kw in text
        for kw in ["429", "too many requests", "resource_exhausted", "rate limit", "rate-limit"]
    ) or ("quota" in text and "exceed" in text)


def _sleep_for_rate_limit(exc: BaseException) -> None:
    parsed = _parse_retry_after_seconds(str(exc))
    wait = max(parsed if parsed is not None else _RATE_LIMIT_BACKOFF_SEC, 5.0)
    logger.warning("Hız limiti algılandı → %.0f sn bekleniyor...", wait)
    time.sleep(wait)


# ---------------------------------------------------------------------------
# DB yardımcıları (main.py ile aynı mantık)
# ---------------------------------------------------------------------------

def fetch_pending_articles(limit: int = 10) -> list[dict]:
    """Henüz işlenmemiş haberleri veritabanından getirir."""
    session = SessionLocal()
    try:
        articles = (
            session.query(Article)
            .filter(
                or_(
                    Article.ai_quality_score.is_(None),
                    Article.summary.is_(None),
                )
            )
            .limit(limit)
            .all()
        )
        return [
            {
                "article_id": a.id,
                "title": a.title,
                "content": (a.content or a.summary or "")[:2000],
            }
            for a in articles
        ]
    except Exception as exc:
        logger.error("Bekleyen makaleler çekilirken hata: %s", exc)
        return []
    finally:
        session.close()


def fetch_available_categories() -> str:
    """Tüm kategori adlarını virgülle ayrılmış string olarak döndürür."""
    session = SessionLocal()
    try:
        cats = session.query(Category).all()
        return ", ".join(c.name for c in cats)
    except Exception as exc:
        logger.error("Kategoriler çekilirken hata: %s", exc)
        return ""
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Ana döngü
# ---------------------------------------------------------------------------

def run() -> None:
    """
    LangGraph tabanlı ana çalışma döngüsü.

    Her haber için tsn_graph.invoke() çağrılır.
    LangSmith, her invoke'u bağımsız bir trace olarak kaydeder:
        run_name = "article_{id}_{title_prefix}"
    """
    logger.info("=" * 65)
    logger.info("TSN Media AI Worker (LangGraph) başlatılıyor...")
    logger.info("=" * 65)

    # LangSmith kontrolü
    langsmith_ok = _check_langsmith_config()
    if langsmith_ok:
        logger.info(
            "LangSmith aktif → Proje: %s",
            os.environ.get("LANGCHAIN_PROJECT", "—"),
        )
    else:
        logger.warning(
            "LangSmith devre dışı (API key eksik). "
            "Tracing olmadan devam ediliyor."
        )

    logger.info("Puan eşiği (SCORE_THRESHOLD): %d", SCORE_THRESHOLD)

    # Veritabanından bekleyen haberleri çek
    pending = fetch_pending_articles(limit=10)
    if not pending:
        logger.info("İşlenecek bekleyen haber bulunamadı. Çıkış yapılıyor.")
        return

    logger.info("%d adet bekleyen haber bulundu.", len(pending))
    available_categories = fetch_available_categories()
    logger.info("Mevcut kategoriler: %s", available_categories)

    # Her haber için LangGraph çalıştır
    stats = {"completed": 0, "discarded": 0, "error": 0}

    for idx, article_data in enumerate(pending, start=1):
        if idx > 1:
            logger.info(
                "RPM koruması: %.1f sn bekleniyor...", _MIN_INTERVAL_SEC
            )
            time.sleep(_MIN_INTERVAL_SEC)

        article_id = article_data["article_id"]

        logger.info(
            "[%d/%d] LangGraph pipeline başlatılıyor → ID=%d — %s",
            idx,
            len(pending),
            article_id,
            article_data["title"][:80],
        )

        # -- Başlangıç state'i (ArticleState) --
        initial_state: ArticleState = {
            "article_id": article_id,
            "title": article_data["title"],
            "content": article_data["content"],
            "available_categories": available_categories,
            "score_threshold": SCORE_THRESHOLD,
            # Boş başlangıç değerleri
            "quality_score": None,
            "categories": [],
            "summary": None,
            "audio_url": None,
            "video_url": None,
            "personalization_tags": [],
            "status": "pending",
            "error_message": None,
            "retry_count": 0,
        }

        try:
            # -- LangGraph pipeline çalıştır --
            # config["run_name"] → LangSmith'te her haberın adı bu olur
            final_state: ArticleState = tsn_graph.invoke(
                initial_state,
                config={
                    "run_name": (
                        f"article_{article_id}_"
                        f"{article_data['title'][:30].replace(' ', '_')}"
                    ),
                },
            )

            final_status = final_state.get("status", "unknown")

            if final_status == "completed":
                stats["completed"] += 1
            elif final_status == "discarded":
                stats["discarded"] += 1
            else:
                stats["error"] += 1

            logger.info(
                "[%d/%d] ✓ ID=%d | Durum: %s | Puan: %s | Etiketler: %s",
                idx,
                len(pending),
                article_id,
                final_status.upper(),
                final_state.get("quality_score", "—"),
                final_state.get("personalization_tags", []),
            )

        except Exception as exc:
            stats["error"] += 1
            logger.error(
                "[%d/%d] ✗ Kritik hata → ID=%d | %s",
                idx,
                len(pending),
                article_id,
                exc,
            )
            if _is_rate_limit_error(exc):
                _sleep_for_rate_limit(exc)

    # -- Özet rapor --
    logger.info("=" * 65)
    logger.info(
        "AI Worker (LangGraph) tamamlandı. "
        "Tamamlanan: %d | Reddedilen: %d | Hata: %d",
        stats["completed"],
        stats["discarded"],
        stats["error"],
    )
    logger.info("=" * 65)


if __name__ == "__main__":
    run()
