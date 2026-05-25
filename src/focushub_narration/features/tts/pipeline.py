import logging
from typing import Generator, Optional

import numpy as np
import torch
from TTS.api import TTS

# Patch torch.load to bypass weights_only=True default in PyTorch 2.6+ for legacy Coqui models
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from focushub_narration.config import (  # noqa: E402
    DEFAULT_LANG_CODE,
    DEFAULT_MODEL_NAME,
    DEFAULT_REFERENCE_VOICE,
    DEFAULT_SPEED,
    get_device,
    get_vram_info,
)

logger = logging.getLogger(__name__)


class NarrationPipeline:
    """Wrapper around Coqui TTS (XTTS v2) for premium zero-shot Spanish voice cloning."""

    def __init__(self, lang_code: str = DEFAULT_LANG_CODE):
        self.lang_code = lang_code
        self.device = get_device()
        self._pipeline: Optional[TTS] = None

    @property
    def pipeline(self) -> TTS:
        """Lazily initialize TTS to avoid heavy startup penalty when not needed."""
        if self._pipeline is None:
            logger.info("Initializing Coqui TTS (XTTS v2) on %s...", self.device.upper())
            logger.info("System hardware: %s", get_vram_info())

            # Lazily initialize and load the model weights
            # XTTS v2 automatically downloads to local cache (~1.8GB) on first execution
            self._pipeline = TTS(model_name=DEFAULT_MODEL_NAME).to(self.device)
        return self._pipeline

    def generate(
        self,
        text: str,
        reference_voice: str = DEFAULT_REFERENCE_VOICE,
        speed: float = DEFAULT_SPEED,
    ) -> Generator[np.ndarray, None, None]:
        """Synthesize Spanish text into high-fidelity audio cloning a reference speaker.

        Args:
            text: The full text string to be spoken.
            reference_voice: Path to the reference 10-20s WAV/MP3 speaker audio clip.
            speed: The speed multiplier (1.0 is standard).

        Yields:
            Numpy array containing the synthesized high-fidelity 24kHz waveform.
        """
        logger.info("Processing text with XTTS v2 using reference voice: %s...", reference_voice)

        # Coqui's tts() returns a list of floats representing the audio waveform at 24000Hz
        wav = self.pipeline.tts(
            text=text,
            speaker_wav=reference_voice,
            language=self.lang_code,
            speed=speed,
        )

        if wav is not None:
            # Convert list of floats to a contiguous numpy array
            yield np.array(wav)
        else:
            logger.error("XTTS v2 synthesis failed to return audio waveform.")
