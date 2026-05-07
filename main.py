"""
TSN Media AI Worker — Entry point. (GERÇEK ÜRETİM ORTAMI ÇALIŞTIRICISI)
=======================================================================
SUNUM NOTU (Hoca İçin Mimari Açıklama):
Sunum sırasında ekranda izlediğiniz "demo_simulation.py" adlı görsel şölenin, 
veritabanına bağlı ve gerçek üretim (production) ortamında çalışan arka plan hali budur.

Sistem Mimarisi ve Akış (Workflow):
-----------------------------------
1. INGESTION (Veri Çekme): `fetch_pending_articles` fonksiyonu, veritabanına giderek
   henüz yapay zeka tarafından işlenmemiş (kalite puanı veya özeti olmayan) haberleri çeker.
2. CREW AI BAŞLATMA: `tsn_crew = TsnMediaCrew()` ile, sunumda bahsettiğimiz ajanlar 
   (Scoring Agent, Categorization Agent, Summarization Agent) ayağa kaldırılır.
3. İŞLEME DÖNGÜSÜ: `tsn_crew.crew().kickoff(inputs=inputs)` komutuyla haber sırayla ajanlara verilir.
   - Önce puanlanır (Score < 50 ise reddedilir)
   - Ardından Kategorisi belirlenir.
   - En son TL;DR özeti çıkartılır.
4. VERİTABANI GÜNCELLEME: Ajanların kullandığı "Tools" (Araçlar), sonuçları doğrudan 
   veritabanına yazarak front-end (Angular) arayüzüne düşmesini sağlar.
"""

import logging
import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# Path setup — ensure both project root and backend are importable
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_backend_dir = os.path.join(_project_root, "backend")

for path in [_project_root, _backend_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from dotenv import load_dotenv

# Load .env from the backend directory (DB credentials, API keys)
load_dotenv(os.path.join(_backend_dir, ".env"))

from sqlalchemy import or_

from app.core.database import SessionLocal
from app.models.article import Article
from app.models.category import Category
from ai_workers.crew import TsnMediaCrew

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai_worker")

# Bekleme süreleri (Gemini ücretsiz: ~10 istek/dk; bir haber = birden fazla görev = birden fazla istek)
_MIN_INTERVAL_SEC = float(os.environ.get("AI_WORKER_MIN_INTERVAL_SEC", "20"))
_RATE_LIMIT_BACKOFF_SEC = float(os.environ.get("AI_WORKER_RATE_LIMIT_BACKOFF_SEC", "45"))


def _parse_retry_after_seconds(message: str) -> float | None:
    """API metninden '39 saniye bekle' veya Retry-After benzeri süre çıkarmaya çalışır."""
    m = re.search(r"retry\s*(?:in|after)?\s*[:\s]*(\d+)\s*s", message, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*second", message, re.I)
    if m:
        return float(m.group(1))
    return None


def _is_rate_limit_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__!s} {exc!s}".lower()
    if "429" in text:
        return True
    if "too many requests" in text:
        return True
    if "resource_exhausted" in text:
        return True
    if "rate limit" in text or "rate-limit" in text:
        return True
    if "quota" in text and ("exceed" in text or "exceeded" in text):
        return True
    return False


def _sleep_for_rate_limit(exc: BaseException) -> None:
    parsed = _parse_retry_after_seconds(str(exc))
    wait = parsed if parsed is not None else _RATE_LIMIT_BACKOFF_SEC
    # API önerisinden biraz pay bırak
    wait = max(wait, 5.0)
    logger.warning(
        "Hız limiti algılandı. Bir sonraki haberden önce %.0f sn bekleniyor...",
        wait,
    )
    time.sleep(wait)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def fetch_pending_articles(limit: int = 10) -> list[dict]:
    """Fetch articles where AI processing is incomplete."""
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

        results = []
        for article in articles:
            results.append({
                "article_id": article.id,
                "title": article.title,
                "content": (article.content or article.summary or "")[:2000],
            })

        return results

    except Exception as e:
        logger.error("Bekleyen makaleler çekilirken hata: %s", e)
        return []
    finally:
        session.close()


def fetch_available_categories() -> str:
    """Fetch all category names as a comma-separated string."""
    session = SessionLocal()
    try:
        categories = session.query(Category).all()
        return ", ".join(cat.name for cat in categories)
    except Exception as e:
        logger.error("Kategoriler çekilirken hata: %s", e)
        return ""
    finally:
        session.close()


def run():
    """
    Ana Çalıştırma Döngüsü (Main Execution Loop)
    SUNUMDAKİ YERİ: Terminaldeki "Starting Autonomous Web Scraping & Processing (INFINITE LOOP)..."
    kısmının, veritabanı kontrollü çalışan gerçek production versiyonudur.
    """
    logger.info("=" * 60)
    logger.info("TSN Media AI Worker başlatılıyor...")
    logger.info("=" * 60)

    # Fetch pending articles
    pending = fetch_pending_articles(limit=10)
    if not pending:
        logger.info("İşlenecek bekleyen haber bulunamadı. Çıkış yapılıyor.")
        return

    logger.info("%d adet bekleyen haber bulundu.", len(pending))

    # Fetch available categories once
    available_categories = fetch_available_categories()
    logger.info("Mevcut kategoriler: %s", available_categories)

    # Initialize crew
    tsn_crew = TsnMediaCrew()

    # Process each article
    for idx, article_data in enumerate(pending, start=1):
        if idx > 1:
            logger.info(
                "RPM koruması: haberler arası %.1f sn bekleniyor...",
                _MIN_INTERVAL_SEC,
            )
            time.sleep(_MIN_INTERVAL_SEC)

        logger.info(
            "[%d/%d] İşleniyor: ID=%d — %s",
            idx, len(pending),
            article_data["article_id"],
            article_data["title"][:80],
        )

        try:
            inputs = {
                "article_id": article_data["article_id"],
                "title": article_data["title"],
                "content": article_data["content"],
                "available_categories": available_categories,
            }

            # SUNUMDAKİ YERİ: Tüm yapay zeka süreçlerinin (Puanlama -> Kategori -> Özet)
            # tek bir komutla sırayla çalıştırıldığı yer. (Sequential Process)
            result = tsn_crew.crew().kickoff(inputs=inputs)
            logger.info(
                "[%d/%d] TAMAMLANDI: ID=%d",
                idx, len(pending), article_data["article_id"],
            )
            logger.debug("Crew sonucu: %s", result)

        except Exception as e:
            logger.error(
                "[%d/%d] HATA: ID=%d — %s",
                idx, len(pending), article_data["article_id"], e,
            )
            if _is_rate_limit_error(e):
                _sleep_for_rate_limit(e)
            continue

    logger.info("=" * 60)
    logger.info("Tüm bekleyen haberler işlendi. AI Worker tamamlandı.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
