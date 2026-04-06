from faster_whisper import WhisperModel
from functools import lru_cache


@lru_cache(maxsize=1)
def get_whisper_model():
    try:
        import torch
        if torch.cuda.is_available():
            return WhisperModel("base", device="cuda", compute_type="float16")
        else:
            return WhisperModel("base", device="cpu", compute_type="int8")
    except ImportError:
        return WhisperModel("base", device="cpu", compute_type="int8")


def parse_media(file_path: str) -> str:
    model = get_whisper_model()
    segments, info = model.transcribe(file_path, task="translate")
    text = " ".join([segment.text for segment in segments])
    return text.strip()