"""
TSN Media — Media Generation Nodes
==================================
Bu dosya demo_simulation.py içindeki şu bölümü production graph'a taşır:

    [LangGraph] Voice Synthesis Node generating audio from title...
    [Voice Node] Audio file actually generated: audio_outputs/voice_1043.mp3
    [Voice Node] Playing synthesized audio...

Demo dosyası kullanıcıya sahnede gerçek bir deneyim göstermek için gTTS ile
başlığı MP3'e çevirip pygame ile oynatır. Production worker tarafında aynı
akışı iki katmana ayırıyoruz:

1. voice_synthesis_node
   Haberin başlığını ses üretim girdisi kabul eder. Varsayılan olarak dış API
   çağırmaz; ancak MEDIA_ENABLE_LOCAL_TTS=true yapılırsa demo'daki gTTS
   mantığına benzer şekilde audio_outputs klasörüne MP3 yazmayı dener.

2. video_generation_node
   Demo'da henüz görsel olarak işlenmeyen fakat mimaride gösterilen genişletme
   noktasıdır. Şimdilik güvenli placeholder olarak kalır.

Bu görevde kodu çalıştırmak istenmediği için node'lar bol açıklamalı ve
opsiyonel çalışacak şekilde düzenlenmiştir.

Kullanım:
    from ai_workers.media_nodes import voice_synthesis_node, video_generation_node
    graph.add_node("voice_synthesis_node", voice_synthesis_node)
    graph.add_node("video_generation_node", video_generation_node)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("media_nodes")

# Phase 2 yapılandırması (şu an kullanılmıyor, yer tutucu olarak bırakıldı)
_VOICE_API_PROVIDER = os.environ.get("VOICE_API_PROVIDER", "local_gtts")  # local_gtts | google_tts | elevenlabs
_VIDEO_API_PROVIDER = os.environ.get("VIDEO_API_PROVIDER", "runwayml")    # runwayml | sora
_MEDIA_ENABLE_LOCAL_TTS = os.environ.get("MEDIA_ENABLE_LOCAL_TTS", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Phase 2 — Gerçek API çağrıları buraya gelecek
# ---------------------------------------------------------------------------

def _run_voice_api(text: str, article_id: int) -> str | None:
    """
    Ses üretim katmanı.

    demo_simulation.py doğrudan gTTS(text=title, lang="tr") çağırır. Burada
    aynı davranışı opsiyonel yapıyoruz, çünkü production worker her çalıştığında
    yerel dosya üretmek veya internet üzerinden gTTS'e gitmek istenmeyebilir.

    Dönüş değeri:
        - Başarılıysa audio_outputs/voice_<article_id>.mp3 gibi göreli yol.
        - Kapalıysa veya hata olursa None.

    Phase 2 hedefleri:
        - local_gtts  : demo ile birebir uyumlu, hızlı prototip modu
        - google_tts  : Google Cloud Text-to-Speech entegrasyonu
        - elevenlabs  : daha doğal stüdyo sesi entegrasyonu
    """
    if not text:
        return None

    if _VOICE_API_PROVIDER == "local_gtts" and _MEDIA_ENABLE_LOCAL_TTS:
        try:
            from gtts import gTTS

            project_root = Path(__file__).resolve().parents[1]
            audio_dir = project_root / "audio_outputs"
            audio_dir.mkdir(exist_ok=True)

            audio_path = audio_dir / f"voice_{article_id}.mp3"
            tts = gTTS(text=text, lang="tr")
            tts.save(str(audio_path))
            return f"audio_outputs/{audio_path.name}"
        except Exception as exc:
            logger.warning(
                "[VoiceSynthesis] ID=%d | local_gtts başarısız: %s",
                article_id,
                exc,
            )
            return None

    # TODO: Phase 2'de google_tts / elevenlabs gerçek API entegrasyonu
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

    Demo ile bilinçli uyum:
        - Ses metni olarak summary değil title kullanılır.
        - Çünkü demo_simulation.py yalnızca haber başlığını seslendirir.
        - Summary daha uzun ve haber spikeri metni gibi okunabilir; fakat bu
          demo akışının gösterdiği davranış değildir.

    MEDIA_ENABLE_LOCAL_TTS=false iken node sadece log bırakır ve audio_url=None
    döndürür. Bu, kodu çalıştırmadan mimariyi göstermek için güvenlidir.
    """
    article_id = state.get("article_id", 0)
    title = state.get("title") or ""

    logger.info(
        "[VoiceSynthesis] Node tetiklendi → ID=%d | [PHASE 2 STUB]",
        article_id,
    )

    audio_url = _run_voice_api(title, article_id)

    if audio_url:
        logger.info(
            "[VoiceSynthesis] ID=%d | Ses dosyası oluşturuldu: %s",
            article_id, audio_url
        )
    else:
        logger.info(
            "[VoiceSynthesis] ID=%d | STUB: Phase 2'de aktif olacak "
            "(Provider: %s, local_tts_enabled=%s)",
            article_id, _VOICE_API_PROVIDER, _MEDIA_ENABLE_LOCAL_TTS
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
