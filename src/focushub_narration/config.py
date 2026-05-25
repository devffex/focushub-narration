import torch

# Default settings for XTTS v2
DEFAULT_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_LANG_CODE = "es"
DEFAULT_REFERENCE_VOICE = "voices/alberto_rodriguez.mp3"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SPEED = 1.0

# Mastering chain defaults (Phase 2: Automated Audio Mastering Pipeline)
MASTERING_HPF_CUTOFF_HZ = 80
MASTERING_COMPRESSOR_THRESHOLD_DB = -20.0
MASTERING_COMPRESSOR_RATIO = 2.0
MASTERING_COMPRESSOR_ATTACK_MS = 15.0
MASTERING_COMPRESSOR_RELEASE_MS = 150.0
MASTERING_LIMITER_THRESHOLD_DB = -1.0
MASTERING_TARGET_LUFS = -14.0


def get_device() -> str:
    """Determine the best available hardware device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_vram_info() -> str:
    """Return memory details of the primary GPU if available."""
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device_idx)
        total_mem = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
        return f"GPU: {name} ({total_mem:.2f} GB VRAM)"
    return "CPU Only"
