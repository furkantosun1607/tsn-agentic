"""
TSN Media — Media Generation Stub Nodes
=======================================
Şemadaki "Voice Synthesis Agent" ve "Video Generation Agent" düğümlerinin
LangGraph node implementasyonları.

Durum: STUB (Phase 2 Placeholder)
    Bu node'lar şu an gerçek API çağrısı yapmaz; sabırlı bir log mesajı
    bırakır ve state'i olduğu gibi geçirir. Amaç:
    1. LangGraph graph yapısının şimdi tamamlanmış olması (node eksik kalmaz)
    2. LangSmith'te voice/video adımlarının trace'de görünmesi
    3. Phase 2'de sadece _run_voice_api() ve _run_video_api() fonksiyonlarını
       doldurmak yeterli — graph yapısı değişmez.

Phase 2 Entegrasyon Hedefleri:
    - Voice Synthesis: Google Cloud Text-to-Speech API veya ElevenLabs API
    - Video Generation: RunwayML API veya Sora API

Kullanım:
    from ai_workers.media_nodes import voice_synthesis_node, video_generation_node
    graph.add_node("voice_synthesis_node", voice_synthesis_node)
    graph.add_node("video_generation_node", video_generation_node)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("media_nodes")

# Phase 2 yapılandırması (şu an kullanılmıyor, yer tutucu olarak bırakıldı)
_VOICE_API_PROVIDER = os.environ.get("VOICE_API_PROVIDER", "google_tts")  # google_tts | elevenlabs
_VIDEO_API_PROVIDER = os.environ.get("VIDEO_API_PROVIDER", "runwayml")    # runwayml | sora


# ---------------------------------------------------------------------------
# Phase 2 — Gerçek API çağrıları buraya gelecek
# ---------------------------------------------------------------------------

def _run_voice_api(text: str, article_id: int) -> str | None:
    """
    [PHASE 2 PLACEHOLDER]
    TTS API'sine metin gönderecek ve audio dosyasının URL'ini döndürecek.

    Hedef implementasyon:
        if _VOICE_API_PROVIDER == "google_tts":
            from google.cloud import texttospeech
            # ... Google TTS çağrısı
        elif _VOICE_API_PROVIDER == "elevenlabs":
            import requests
            # ... ElevenLabs API çağrısı
    """
    # TODO: Phase 2'de gerçek API entegrasyonu
    return None


def _run_video_api(summary: str, article_id: int) -> str | None:
    """
    [PHASE 2 PLACEHOLDER]
    Video generation API'sine özet gönderecek ve video URL'ini döndürecek.

    Hedef implementasyon:
        if _VIDEO_API_PROVIDER == "runwayml":
            # ... RunwayML Gen-2 API çağrısı
        elif _VIDEO_API_PROVIDER == "sora":
            # ... OpenAI Sora API çağrısı (erişim açıldığında)
    """
    # TODO: Phase 2'de gerçek API entegrasyonu
    return None


# ---------------------------------------------------------------------------
# LangGraph Node'ları
# ---------------------------------------------------------------------------

def voice_synthesis_node(state: dict) -> dict:
    """
    Voice Synthesis LangGraph Node.
    Şemadaki "Voice Synthesis Agent" kutusunun karşılığı.

    Şu an STUB — Phase 2'de gerçek TTS API entegrasyonu yapılacak.
    LangSmith bu node'u trace eder; gecikme ve durum izlenebilir.
    """
    article_id = state.get("article_id", 0)
    summary = state.get("summary") or ""

    logger.info(
        "[VoiceSynthesis] Node tetiklendi → ID=%d | [PHASE 2 STUB]",
        article_id,
    )

    # Phase 2: audio_url = _run_voice_api(summary, article_id)
    audio_url = _run_voice_api(summary, article_id)

    if audio_url:
        logger.info(
            "[VoiceSynthesis] ID=%d | Ses dosyası oluşturuldu: %s",
            article_id, audio_url
        )
    else:
        logger.info(
            "[VoiceSynthesis] ID=%d | STUB: Phase 2'de aktif olacak "
            "(Hedef API: %s)",
            article_id, _VOICE_API_PROVIDER
        )

    return {**state, "audio_url": audio_url}


def video_generation_node(state: dict) -> dict:
    """
    Video Generation LangGraph Node.
    Şemadaki "Video Generation Agent" kutusunun karşılığı.

    Şu an STUB — Phase 2'de gerçek video generation API entegrasyonu yapılacak.
    LangSmith bu node'u trace eder; gecikme ve durum izlenebilir.
    """
    article_id = state.get("article_id", 0)
    summary = state.get("summary") or ""

    logger.info(
        "[VideoGeneration] Node tetiklendi → ID=%d | [PHASE 2 STUB]",
        article_id,
    )

    # Phase 2: video_url = _run_video_api(summary, article_id)
    video_url = _run_video_api(summary, article_id)

    if video_url:
        logger.info(
            "[VideoGeneration] ID=%d | Video dosyası oluşturuldu: %s",
            article_id, video_url
        )
    else:
        logger.info(
            "[VideoGeneration] ID=%d | STUB: Phase 2'de aktif olacak "
            "(Hedef API: %s)",
            article_id, _VIDEO_API_PROVIDER
        )

    return {**state, "video_url": video_url}
