"""
TSN Media — Personalization Agent (LangChain LCEL)
==================================================
Şemadaki "Personalization Agent" düğümünün Python implementasyonu.

Teknoloji Seçimi: LangChain LCEL (Expression Language)
    - Neden CrewAI değil? Personalizasyon tek adımlı bir LLM görevi.
      CrewAI'nin ajan/araç/görev üçlüsüne ihtiyaç yok; basit bir
      prompt → LLM → parser zinciri yeterli.
    - Neden LCEL? Modern LangChain'in kompozisyon modeli; pipe (|)
      operatörü ile prompt, model ve parser tek satırda bağlanıyor.
    - LangSmith Entegrasyonu: LANGCHAIN_TRACING_V2=true olduğunda
      her zincir çalışması otomatik olarak LangSmith'e kaydedilir.

Görev:
    - ONAYLANAN haberler için içerik tabanlı kişiselleştirme etiketleri
    - REDDEDİLEN haberler için negatif sinyal etiketleri
      (kalite filtresi bilgisi olarak gelecekte kullanılabilir)
"""

from __future__ import annotations

import logging
import os
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Path setup
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_backend_dir = os.path.join(_project_root, "backend")
for _p in [_project_root, _backend_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger("personalization_chain")


# ---------------------------------------------------------------------------
# Output Schema
# ---------------------------------------------------------------------------

class PersonalizationOutput(BaseModel):
    """LangChain LCEL zincirinin yapılandırılmış çıktısı."""

    tags: list[str] = Field(
        ...,
        description=(
            "Habere özgü kişiselleştirme etiketleri. "
            "Örnek: ['Sabah Okuyucusu', 'Teknoloji Meraklısı', 'Borsa Takipçisi']. "
            "Maksimum 5 etiket."
        ),
    )
    sentiment: str = Field(
        ...,
        description="'positive' (yayınlanan haber) veya 'negative' (reddedilen haber).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Etiket güven skoru (0.0 – 1.0 arası).",
    )


# ---------------------------------------------------------------------------
# LLM ve Zincir Kurulumu
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Sen bir içerik kişiselleştirme uzmanısın. Görevin, verilen haber makalesini
analiz ederek hangi kullanıcı segmentlerinin bu haberle ilgileneceğini tespit etmektir.

Çıktın MUTLAKA şu JSON formatında olmalı:
{{
  "tags": ["<etiket1>", "<etiket2>", ...],  // maksimum 5 adet
  "sentiment": "positive" veya "negative",
  "confidence": <0.0 ile 1.0 arası sayı>
}}

Etiket örnekleri: "Sabah Okuyucusu", "Teknoloji Meraklısı", "Borsa Takipçisi",
"Spor Tutkunu", "Siyaset Analisti", "Dünya Haberleri", "Yerel Haber Takipçisi"

Reddedilen (negatif sentiment) haberler için etiketler negatif sinyal içermeli:
Örnek: "Düşük Kalite Sinyal", "Clickbait Risk", "Kısa İçerik"
"""

_HUMAN_PROMPT = """\
Haber Durumu: {status}
Başlık: {title}
Kategoriler: {categories}
Özet: {summary}
"""


def build_personalization_chain():
    """
    LangChain LCEL zinciri oluşturur ve döndürür.

    Zincir: ChatPromptTemplate | ChatGoogleGenerativeAI (structured) 
    LangSmith, LANGCHAIN_TRACING_V2=true ise zinciri otomatik izler.
    """
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.3,
    )

    # with_structured_output → LangChain'in modern yapılandırılmış çıktı API'si
    # Pydantic modeli LLM'e schema olarak verilir; JSON doğrulaması otomatik yapılır
    structured_llm = llm.with_structured_output(PersonalizationOutput)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ]
    )

    # LCEL pipe operatörü: prompt çıktısı → LLM → yapılandırılmış çıktı
    return prompt | structured_llm


# ---------------------------------------------------------------------------
# Ana çalıştırma fonksiyonu (LangGraph node'u tarafından çağrılır)
# ---------------------------------------------------------------------------

def run_personalization_chain(
    article_id: int,
    title: str,
    status: str,
    categories: list[str],
    summary: str | None,
) -> PersonalizationOutput:
    """
    Personalization zincirini çalıştırır ve etiketleri döndürür.

    Args:
        article_id: Makale ID'si (loglama için).
        title: Haber başlığı.
        status: "approved" (yayınlandı) veya "discarded" (reddedildi).
        categories: Atanmış kategori listesi.
        summary: AI tarafından üretilen özet (reddedilenlerde None olabilir).

    Returns:
        PersonalizationOutput: tags, sentiment, confidence alanlarını içerir.
    """
    chain = build_personalization_chain()

    categories_str = ", ".join(categories) if categories else "Belirtilmedi"
    summary_str = summary or "Özet mevcut değil (reddedilen haber)"
    status_label = "ONAYLANDI - yayınlanacak" if status == "approved" else "REDDEDİLDİ - yayınlanmayacak"

    logger.info(
        "[Personalization] Zincir çalıştırılıyor → ID=%d | Durum=%s",
        article_id,
        status,
    )

    try:
        result: PersonalizationOutput = chain.invoke(
            {
                "status": status_label,
                "title": title,
                "categories": categories_str,
                "summary": summary_str,
            }
        )
        logger.info(
            "[Personalization] ID=%d | Etiketler=%s | Güven=%.2f",
            article_id,
            result.tags,
            result.confidence,
        )
        return result

    except Exception as exc:
        logger.error(
            "[Personalization] ID=%d | Zincir hatası: %s", article_id, exc
        )
        # Hata durumunda güvenli varsayılan döndür; pipeline durmamalı
        return PersonalizationOutput(
            tags=["İşlem Hatası"],
            sentiment="neutral",
            confidence=0.0,
        )
