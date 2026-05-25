import logging
from typing import Generator, Optional

import numpy as np
from kokoro import KPipeline

from focushub_narration.config import (
    DEFAULT_LANG_CODE,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    get_device,
    get_vram_info,
)

logger = logging.getLogger(__name__)


class NarrationPipeline:
    """Wrapper around Kokoro KPipeline for robust Spanish TTS generation."""

    def __init__(self, lang_code: str = DEFAULT_LANG_CODE):
        self.lang_code = lang_code
        self.device = get_device()
        self._pipeline: Optional[KPipeline] = None

    @property
    def pipeline(self) -> KPipeline:
        """Lazily initialize KPipeline to avoid heavy startup penalty when not needed."""
        if self._pipeline is None:
            logger.info("Initializing KPipeline on %s...", self.device.upper())
            logger.info("System hardware: %s", get_vram_info())
            # This automatically downloads the weight files (~300MB) on first run
            self._pipeline = KPipeline(lang_code=self.lang_code, device=self.device)
        return self._pipeline

    def generate(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_SPEED,
        split_pattern: str = r"\n+|\.",
    ) -> Generator[np.ndarray, None, None]:
        """Generate audio segments from the input text.

        Splits the text using the specified split_pattern to prevent OOM errors
        on systems with limited VRAM (e.g. 4GB limits).

        Args:
            text: The full text string to be spoken.
            voice: The voice style character to use (e.g., 'ef_dora').
            speed: The speed multiplier for the voice.
            split_pattern: Regex pattern to split sentences (default is newlines and periods).

        Yields:
            Numpy arrays containing high-fidelity 24kHz audio segments.
        """
        logger.info("Processing text with voice '%s' at speed %s...", voice, speed)

        # Generator yields (graphemes, phonemes, audio)
        generator = self.pipeline(
            text,
            voice=voice,
            speed=speed,
            split_pattern=split_pattern,
        )

        for i, (_graphemes, _phonemes, audio) in enumerate(generator):
            if audio is not None:
                logger.debug("Segment %d: processed successfully.", i + 1)
                yield audio
            else:
                logger.warning("Segment %d: returned empty audio.", i + 1)
