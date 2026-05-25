"""Audio I/O and professional mastering pipeline for FocusHub Narration.

Handles audio concatenation, professional mastering (HPF → Compressor → Limiter),
and WAV file output. The mastering chain uses Spotify's pedalboard library to
produce broadcast-ready audio at -14 LUFS.
"""

import logging
import os
from typing import Sequence

import numpy as np
import soundfile as sf
from pedalboard import (
    Compressor,
    Gain,
    HighpassFilter,
    Limiter,
    Pedalboard,
)

from focushub_narration.config import (
    DEFAULT_SAMPLE_RATE,
    MASTERING_COMPRESSOR_ATTACK_MS,
    MASTERING_COMPRESSOR_RATIO,
    MASTERING_COMPRESSOR_RELEASE_MS,
    MASTERING_COMPRESSOR_THRESHOLD_DB,
    MASTERING_HPF_CUTOFF_HZ,
    MASTERING_LIMITER_THRESHOLD_DB,
    MASTERING_TARGET_LUFS,
)

logger = logging.getLogger(__name__)


def _estimate_lufs(audio: np.ndarray) -> float:
    """Estimate integrated loudness in LUFS using an RMS-based approximation.

    This is a simplified estimation. True LUFS (ITU-R BS.1770) requires
    K-weighting and gated measurement, but for voice-only content the RMS
    approximation with a speech-specific offset is within ~1 dB of the true value.

    For speech, true LUFS typically reads ~3 dB lower than raw RMS because
    gating excludes silence between phrases and K-weighting de-emphasizes
    low frequencies.

    Args:
        audio: 1D numpy array of audio samples (float32/float64).

    Returns:
        Estimated loudness in LUFS.
    """
    # Speech-specific offset: true LUFS reads ~3 dB lower than raw RMS for voice
    SPEECH_LUFS_OFFSET = -3.0

    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return float("-inf")
    return 20 * np.log10(rms) + SPEECH_LUFS_OFFSET


def master_audio(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Run the audio through a professional 3-stage mastering chain.

    Pipeline:
        1. High-Pass Filter (80 Hz) — remove low-end rumble
        2. Compressor (-16 dB, 3:1) — even out vocal dynamics
        3. Peak Limiter (-1 dB) — prevent clipping
        4. Gain adjustment — boost to -14 LUFS broadcast standard

    Args:
        audio: 1D numpy array of raw synthesized audio (float32).
        sample_rate: Sample rate of the audio (default: 24000 Hz).

    Returns:
        Mastered audio as a 1D numpy array.
    """
    logger.info("Running professional mastering chain...")

    # Ensure float32 for pedalboard compatibility
    audio = audio.astype(np.float32)

    # Stage 1-3: HPF → Compressor → Limiter
    board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=MASTERING_HPF_CUTOFF_HZ),
            Compressor(
                threshold_db=MASTERING_COMPRESSOR_THRESHOLD_DB,
                ratio=MASTERING_COMPRESSOR_RATIO,
                attack_ms=MASTERING_COMPRESSOR_ATTACK_MS,
                release_ms=MASTERING_COMPRESSOR_RELEASE_MS,
            ),
            Limiter(threshold_db=MASTERING_LIMITER_THRESHOLD_DB),
        ]
    )

    # Pedalboard expects shape (num_channels, num_samples) — reshape for mono
    audio_2d = audio.reshape(1, -1)
    mastered = board(audio_2d, sample_rate)
    mastered = mastered.flatten()

    # Stage 4: LUFS-targeted gain adjustment
    current_lufs = _estimate_lufs(mastered)
    if current_lufs > float("-inf"):
        gain_db = MASTERING_TARGET_LUFS - current_lufs
        # Clamp gain to avoid extreme amplification on very quiet audio
        gain_db = max(-20.0, min(gain_db, 20.0))

        gain_board = Pedalboard(
            [
                Gain(gain_db=gain_db),
                Limiter(threshold_db=MASTERING_LIMITER_THRESHOLD_DB),  # Re-limit after gain
            ]
        )
        mastered_2d = mastered.reshape(1, -1)
        mastered = gain_board(mastered_2d, sample_rate).flatten()

        final_lufs = _estimate_lufs(mastered)
        logger.info(
            "[✓] Mastering complete: %.1f LUFS → %.1f LUFS (gain: %+.1f dB).",
            current_lufs,
            final_lufs,
            gain_db,
        )
    else:
        logger.warning("Audio is silent — skipping LUFS gain adjustment.")

    return mastered


def save_audio(
    segments: Sequence[np.ndarray],
    output_filename: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    apply_mastering: bool = True,
) -> bool:
    """Concatenate audio segments, apply mastering, and save as high-fidelity WAV.

    Args:
        segments: Sequence of numpy arrays containing audio.
        output_filename: The target filepath to save the WAV file.
        sample_rate: The sample rate to output (XTTS native is 24000Hz).
        apply_mastering: If True, run the professional mastering chain before saving.

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

        # Apply professional mastering chain
        if apply_mastering:
            final_audio = master_audio(final_audio, sample_rate)
        else:
            logger.info("Mastering bypassed (--no-master flag).")

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
