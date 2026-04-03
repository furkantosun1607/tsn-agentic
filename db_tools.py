"""
Custom CrewAI tools for database operations.
Each tool inherits from BaseTool and uses robust try-except error handling.
"""

import os
import sys
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session

# Add the backend directory to sys.path so we can import app modules
_backend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"
)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from app.core.database import SessionLocal
from app.models.article import Article, article_categories
from app.models.category import Category


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session() -> Session:
    """Create and return a new database session."""
    return SessionLocal()


# ---------------------------------------------------------------------------
# Tool 1 — Get Pending News
# ---------------------------------------------------------------------------

class _GetPendingNewsInput(BaseModel):
    """Input schema — no parameters needed, fetches all pending articles."""
    limit: int = Field(
        default=10,
        description="Maksimum kaç adet bekleyen haber getirilsin (token tasarrufu için)"
    )


class GetPendingNewsTool(BaseTool):
    """
    Fetches articles from the database that have NOT yet been processed by AI.
    Filter: ai_quality_score IS NULL OR summary IS NULL
    Returns article id, title, and content to save tokens.
    """
    name: str = "get_pending_news"
    description: str = (
        "Veritabanından henüz AI tarafından işlenmemiş haberleri getirir. "
        "ai_quality_score veya summary alanı NULL olan makaleleri döndürür."
    )
    args_schema: Type[BaseModel] = _GetPendingNewsInput

    def _run(self, limit: int = 10) -> str:
        session = _get_session()
        try:
            pending_articles = (
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

            if not pending_articles:
                return "İşlenecek bekleyen haber bulunamadı."

            results = []
            for article in pending_articles:
                # Content yoksa summary'yi fallback olarak kullan
                text_content = article.content or article.summary or ""
                results.append(
                    f"ID: {article.id}\n"
                    f"Başlık: {article.title}\n"
                    f"İçerik: {text_content[:2000]}\n"  # Token tasarrufu: max 2000 karakter
                    f"---"
                )

            return f"Toplam {len(pending_articles)} bekleyen haber bulundu:\n\n" + "\n".join(results)

        except Exception as e:
            return f"HATA: Bekleyen haberler getirilirken sorun oluştu — {str(e)}"
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Tool 2 — Save Quality Score only (used by score_task)
# ---------------------------------------------------------------------------

class _SaveQualityScoreInput(BaseModel):
    """Input schema for saving quality score to DB."""

    article_id: int = Field(..., description="Guncellenecek makalenin ID'si")
    score: int = Field(..., ge=1, le=100, description="1-100 arasi kalite puani")


class SaveQualityScoreTool(BaseTool):
    """
    Writes the AI quality score (ai_quality_score) for an article.
    Categorization and summary are handled separately (not by this tool).
    """

    name: str = "save_quality_score"
    description: str = (
        "Puanlama taski bittikten hemen sonra ZORUNLU olarak cagir. "
        "Makalenin ai_quality_score alanini DB'ye yazar. "
        "Kategorileri veya ozet alanini degistirmez."
    )
    args_schema: Type[BaseModel] = _SaveQualityScoreInput

    def _run(self, article_id: int, score: int) -> str:
        session = _get_session()
        try:
            article = session.query(Article).filter(Article.id == article_id).first()
            if not article:
                return f"HATA: ID={article_id} olan makale bulunamadi."

            article.ai_quality_score = score
            session.commit()
            return f"BAŞARILI: Makale ID={article_id} puani kaydedildi: {score}"
        except Exception as e:
            session.rollback()
            return f"HATA: Puan kaydedilirken sorun olustu — {str(e)}"
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Tool 2 — Update Score & Categories
# ---------------------------------------------------------------------------

class _UpdateScoreAndCategoriesInput(BaseModel):
    """Input schema for updating score and categories."""
    article_id: int = Field(..., description="Güncellenecek makalenin ID'si")
    score: int = Field(..., ge=1, le=100, description="1-100 arası kalite puanı")
    category_names: list[str] = Field(
        ..., description="Atanacak kategori adları listesi (en fazla 3)"
    )


class UpdateScoreAndCategoriesTool(BaseTool):
    """
    Updates the ai_quality_score for an article and links it to categories
    via the article_categories many-to-many junction table.
    Enforces a maximum of 3 categories.
    """
    name: str = "update_score_and_categories"
    description: str = (
        "Bir haberin AI kalite puanını günceller ve en fazla 3 kategoriyi "
        "article_categories Many-to-Many tablosuna kaydeder."
    )
    args_schema: Type[BaseModel] = _UpdateScoreAndCategoriesInput

    def _run(self, article_id: int, score: int, category_names: list[str]) -> str:
        session = _get_session()
        try:
            # --- Validate article ---
            article = session.query(Article).filter(Article.id == article_id).first()
            if not article:
                return f"HATA: ID={article_id} olan makale bulunamadı."

            # --- Update quality score ---
            article.ai_quality_score = score

            # --- Enforce max 3 categories ---
            category_names = category_names[:3]

            # --- Clear existing AI categories for this article ---
            session.execute(
                article_categories.delete().where(
                    article_categories.c.article_id == article_id
                )
            )

            # --- Link new categories ---
            linked_categories = []
            for cat_name in category_names:
                category = (
                    session.query(Category)
                    .filter(Category.name == cat_name.strip())
                    .first()
                )
                if category:
                    session.execute(
                        article_categories.insert().values(
                            article_id=article_id,
                            category_id=category.id,
                        )
                    )
                    linked_categories.append(cat_name)

            session.commit()

            return (
                f"BAŞARILI: Makale ID={article_id} güncellendi. "
                f"Puan: {score}, Kategoriler: {linked_categories}"
            )

        except Exception as e:
            session.rollback()
            return f"HATA: Puan/kategori güncellenirken sorun oluştu — {str(e)}"
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Tool 3 — Update Summary
# ---------------------------------------------------------------------------

class _UpdateSummaryInput(BaseModel):
    """Input schema for updating article summary."""
    article_id: int = Field(..., description="Güncellenecek makalenin ID'si")
    summary: str = Field(..., description="3-4 maddelik TL;DR özet metni")


class UpdateSummaryTool(BaseTool):
    """
    Updates the summary column of an article with the AI-generated TL;DR.
    """
    name: str = "update_summary"
    description: str = (
        "Bir haberin summary alanını AI tarafından üretilen TL;DR özet ile günceller."
    )
    args_schema: Type[BaseModel] = _UpdateSummaryInput

    def _run(self, article_id: int, summary: str) -> str:
        session = _get_session()
        try:
            article = session.query(Article).filter(Article.id == article_id).first()
            if not article:
                return f"HATA: ID={article_id} olan makale bulunamadı."

            article.summary = summary
            session.commit()

            return f"BAŞARILI: Makale ID={article_id} özeti güncellendi."

        except Exception as e:
            session.rollback()
            return f"HATA: Özet güncellenirken sorun oluştu — {str(e)}"
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Tool 4 — Get Available Categories (helper for categorization agent)
# ---------------------------------------------------------------------------

class _GetAvailableCategoriesInput(BaseModel):
    """Input schema — no parameters needed."""
    pass


class GetAvailableCategoriesTool(BaseTool):
    """
    Fetches all available category names from the database.
    Used by the categorization agent to know which categories exist.
    """
    name: str = "get_available_categories"
    description: str = "Veritabanındaki tüm mevcut kategori adlarını getirir."
    args_schema: Type[BaseModel] = _GetAvailableCategoriesInput

    def _run(self) -> str:
        session = _get_session()
        try:
            categories = session.query(Category).all()
            if not categories:
                return "Veritabanında kategori bulunamadı."

            cat_names = [cat.name for cat in categories]
            return f"Mevcut Kategoriler ({len(cat_names)} adet): {', '.join(cat_names)}"

        except Exception as e:
            return f"HATA: Kategoriler getirilirken sorun oluştu — {str(e)}"
        finally:
            session.close()
