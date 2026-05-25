import torch

# Default settings
DEFAULT_LANG_CODE = "es"
DEFAULT_VOICE = "ef_dora"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SPEED = 1.0

# Supported Spanish voice styles
SPANISH_VOICES = {
    "ef_dora": "Spanish Female - highly recommended for clear, expressive inflections",
    "em_alex": "Spanish Male",
    "em_santa": "Spanish Male",
}


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
