import os
import hashlib
import threading
import torch
import soundfile as sf
from playsound import playsound
import time

SAMPLE_RATE = 48000
SPEAKER = "eugene"
DEVICE = "cpu"
CACHE_DIR = "tts_cache"

os.makedirs(CACHE_DIR, exist_ok=True)

_model = None
_lock = threading.Lock()
_play_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                try:
                    print("🔊 Загрузка Silero TTS...")
                    _model, _ = torch.hub.load(
                        repo_or_dir="snakers4/silero-models",
                        model="silero_tts",
                        language="ru",
                        speaker="v4_ru",
                    )
                    _model.to(DEVICE)
                    print(f"   ✅ Silero TTS загружен. Голос: {SPEAKER}")
                except Exception as e:
                    print(f"   ⚠️ Silero TTS не загружен: {e}")
                    _model = None
    return _model


def _normalize(text: str) -> str:
    import re
    text = re.sub(r"[*#_`~]", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _cache_path(text: str) -> str:
    key = hashlib.md5(f"{SPEAKER}:{text}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.wav")


def _synthesize(text: str) -> str | None:
    model = _get_model()
    if not model:
        return None

    text = _normalize(text)
    if not text:
        return None

    path = _cache_path(text)
    
   
    if os.path.exists(path):
        return path

    try:
        with _lock:
            audio = model.apply_tts(text=text, speaker=SPEAKER, sample_rate=SAMPLE_RATE)
        
        
        sf.write(path, audio.numpy(), SAMPLE_RATE)
        
     
        if os.path.exists(path) and os.path.getsize(path) > 0:
            time.sleep(0.05)  
            return path
        else:
            return None
            
    except Exception as e:
        print(f"⚠️ Ошибка синтеза: {e}")
        return None


def _play(path: str):
    """Воспроизведение"""
    if not os.path.exists(path):
        print(f"⚠️ Файл не найден: {path}")
        return
    
    with _play_lock:
        for attempt in range(3):
            try:
                playsound(path)
                return
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.1)
                    continue
                print(f"⚠️ Ошибка воспроизведения: {e}")


def speak(text: str):
    """Озвучить синхронно"""
    if not text:
        return
    path = _synthesize(text)
    if path:
        _play(path)


def speak_async(text: str):
    """Озвучить асинхронно"""
    if not text:
        return
    threading.Thread(target=speak, args=(text,), daemon=True).start()