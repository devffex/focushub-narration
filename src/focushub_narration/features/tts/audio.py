import logging
import os
from typing import Sequence

import numpy as np
import soundfile as sf

from focushub_narration.config import DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)


def save_audio(
    segments: Sequence[np.ndarray],
    output_filename: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bool:
    """Concatenate audio segments and save them as an uncompressed high-fidelity WAV file.

    Args:
        segments: Sequence of numpy arrays containing audio.
        output_filename: The target filepath to save the WAV file.
        sample_rate: The sample rate to output (Kokoro native is 24000Hz).

    Returns:
        True if the file was saved successfully, False otherwise.
    """
    if not segments:
        logger.error("Failed to generate audio: No audio segments available to save.")
        return False

    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(output_filename))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Concatenate numpy array segments
        final_audio = np.concatenate(segments)

        # Write WAV file
        sf.write(output_filename, final_audio, sample_rate)
        logger.info(
            "[✓] High-quality audio saved successfully to: %s",
            os.path.abspath(output_filename),
        )
        return True
    except Exception as e:
        logger.error("Failed to write audio file: %s", e, exc_info=True)
        return False
