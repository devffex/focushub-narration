"""Reference audio preparation for optimal XTTS v2 zero-shot cloning.

Zero-shot cloning models like XTTS v2 are mirror-reflections of the reference clip.
This module ensures the reference is completely dry, peak-normalized to -3dB,
and has all silent gaps removed so the model receives pure, uninterrupted,
high-fidelity vocal data.
"""

import logging
import os

import numpy as np
import soundfile as sf

from focushub_narration.config import DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)

# Voice prep constants
SILENCE_THRESHOLD_DB = -40.0
PEAK_TARGET_DB = -3.0


def _db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10 ** (db / 20.0)


def _trim_silence(audio: np.ndarray, threshold_db: float = SILENCE_THRESHOLD_DB) -> np.ndarray:
    """Strip leading and trailing silence from audio using an energy-threshold algorithm.

    Args:
        audio: 1D numpy array of audio samples.
        threshold_db: Silence threshold in dB (relative to peak). Default: -40 dB.

    Returns:
        Trimmed audio array.
    """
    threshold_linear = _db_to_linear(threshold_db)
    abs_audio = np.abs(audio)

    # Find first and last samples above threshold
    above_threshold = abs_audio > threshold_linear
    if not np.any(above_threshold):
        logger.warning("Audio is entirely below the silence threshold — returning as-is.")
        return audio

    nonzero_indices = np.nonzero(above_threshold)[0]
    start = max(0, nonzero_indices[0] - 512)  # Small margin (~21ms at 24kHz)
    end = min(len(audio), nonzero_indices[-1] + 512)

    trimmed = audio[start:end]
    trimmed_ms = (len(audio) - len(trimmed)) / DEFAULT_SAMPLE_RATE * 1000
    logger.info("[✓] Trimmed %.0f ms of silence (start=%d, end=%d).", trimmed_ms, start, end)
    return trimmed


def _normalize_peak(audio: np.ndarray, target_db: float = PEAK_TARGET_DB) -> np.ndarray:
    """Normalize audio peak amplitude to a target dB level.

    Args:
        audio: 1D numpy array of audio samples.
        target_db: Target peak level in dB. Default: -3 dB.

    Returns:
        Peak-normalized audio array.
    """
    current_peak = np.max(np.abs(audio))
    if current_peak == 0:
        logger.warning("Audio is completely silent — cannot normalize.")
        return audio

    target_linear = _db_to_linear(target_db)
    gain = target_linear / current_peak
    normalized = audio * gain

    current_db = 20 * np.log10(current_peak) if current_peak > 0 else float("-inf")
    logger.info(
        "[✓] Peak normalized: %.1f dB → %.1f dB (gain: %.2fx).",
        current_db,
        target_db,
        gain,
    )
    return normalized


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to a target sample rate using linear interpolation.

    For voice reference clips this is sufficient quality. For production music
    mastering, a proper sinc-based resampler would be preferred.

    Args:
        audio: 1D numpy array of audio samples.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio array.
    """
    if orig_sr == target_sr:
        return audio

    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(audio)), audio)
    logger.info("[✓] Resampled from %d Hz → %d Hz.", orig_sr, target_sr)
    return resampled


def prepare_reference(
    input_path: str,
    output_path: str | None = None,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    target_peak_db: float = PEAK_TARGET_DB,
    silence_threshold_db: float = SILENCE_THRESHOLD_DB,
) -> str:
    """Prepare a reference audio clip for optimal XTTS v2 zero-shot cloning.

    Pipeline: Load → Mono → Resample → Trim Silence → Peak Normalize → Save WAV

    Args:
        input_path: Path to the source reference audio (MP3, WAV, FLAC, etc.).
        output_path: Path for the prepared output WAV. If None, auto-generates
                     a ``_prepped.wav`` sibling file next to the input.
        target_sr: Target sample rate (XTTS native: 24000 Hz).
        target_peak_db: Target peak normalization in dB (-3 dB recommended).
        silence_threshold_db: Threshold for silence trimming in dB.

    Returns:
        Absolute path to the prepared output WAV file.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Reference audio not found: {input_path}")

    logger.info("Preparing reference audio: %s", input_path)

    # 1. Load audio
    audio, orig_sr = sf.read(input_path, dtype="float64")
    logger.info("Loaded: %d samples at %d Hz (%.2fs).", len(audio), orig_sr, len(audio) / orig_sr)

    # 2. Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        logger.info("[✓] Converted stereo → mono.")

    # 3. Resample to target sample rate
    audio = _resample(audio, orig_sr, target_sr)

    # 4. Trim leading/trailing silence
    audio = _trim_silence(audio, threshold_db=silence_threshold_db)

    # 5. Peak normalize to target dB
    audio = _normalize_peak(audio, target_db=target_peak_db)

    # 6. Build output path
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_prepped.wav"

    # 7. Save as lossless WAV
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sf.write(output_path, audio.astype(np.float32), target_sr)
    abs_path = os.path.abspath(output_path)
    duration = len(audio) / target_sr
    logger.info("[✓] Prepared reference saved: %s (%.2fs at %d Hz).", abs_path, duration, target_sr)

    return abs_path
