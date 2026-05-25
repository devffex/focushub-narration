import argparse
import datetime
import logging
import os
import sys
import warnings

from focushub_narration.config import DEFAULT_LANG_CODE, DEFAULT_REFERENCE_VOICE, DEFAULT_SPEED
from focushub_narration.features.tts import NarrationPipeline, save_audio
from focushub_narration.features.tts.voice_prep import prepare_reference

# Suppress third-party PyTorch and model warnings (UserWarning & FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("focushub_narration")


def register_in_catalog(audio_path: str, text: str, reference_voice: str, speed: float):
    """Automatically logs the successful generation inside the markdown catalog file."""
    catalog_path = "audios_catalog.md"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format absolute links and snippets
    basename = os.path.basename(audio_path)
    ref_basename = os.path.basename(reference_voice)
    text_snippet = text if len(text) <= 80 else text[:77] + "..."

    # Create file if it doesn't exist
    if not os.path.exists(catalog_path):
        with open(catalog_path, "w", encoding="utf-8") as f:
            f.write("# 🎙️ FocusHub Audios Catalog\n\n")
            f.write(
                "A detailed registry of locally generated premium voice"
                " narrations using Coqui XTTS v2 zero-shot cloning.\n\n"
            )
            f.write("| Timestamp | Audio File | Reference Voice | Speed | Text Snippet |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")

    # Append the new row
    with open(catalog_path, "a", encoding="utf-8") as f:
        row = (
            f"| {timestamp} | [{basename}]({audio_path})"
            f" | `{ref_basename}` | {speed}x"
            f' | *"{text_snippet}"* |\n'
        )
        f.write(row)
    logger.info("[✓] Successfully registered audio in %s", catalog_path)


def cmd_narrate(args):
    """Execute the narration TTS pipeline (default command)."""
    # Automatically construct unique output path if none is supplied
    if args.output is None:
        outputs_dir = "outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(outputs_dir, f"narration_{timestamp}.wav")

    logger.info("Starting FocusHub Narration Pipeline (XTTS v2)...")
    logger.info("Target Output: %s", args.output)
    logger.info("Reference Voice: %s", args.reference)

    # 1. Initialize the modular Pipeline wrapper
    narrator = NarrationPipeline(lang_code=DEFAULT_LANG_CODE)

    # 2. Generate the audio segments (clones the reference voice dynamically)
    logger.info("Running zero-shot text-to-speech inference...")
    segments = []
    try:
        for segment in narrator.generate(
            text=args.text,
            reference_voice=args.reference,
            speed=args.speed,
        ):
            segments.append(segment)
    except Exception as e:
        logger.error("An error occurred during inference: %s", e, exc_info=True)
        sys.exit(1)

    # 3. Save the high fidelity WAV file
    success = save_audio(
        segments=segments,
        output_filename=args.output,
        apply_mastering=args.master,
    )
    if success:
        logger.info("FocusHub Narration completed successfully! [✓]")
        # 4. Catalog the audio entry in a premium markdown file
        register_in_catalog(
            audio_path=args.output, text=args.text, reference_voice=args.reference, speed=args.speed
        )
    else:
        logger.error("FocusHub Narration failed to complete. [X]")
        sys.exit(1)


def cmd_prep_voice(args):
    """Execute the reference voice preparation pipeline."""
    logger.info("Starting Reference Voice Preparation...")
    try:
        output_path = prepare_reference(
            input_path=args.input,
            output_path=args.output,
        )
        logger.info("Reference voice preparation completed successfully! [✓]")
        logger.info("Prepared file: %s", output_path)
    except FileNotFoundError as e:
        logger.error("Reference voice file not found: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to prepare reference voice: %s", e, exc_info=True)
        sys.exit(1)


_KNOWN_SUBCOMMANDS = {"narrate", "prep-voice"}

# Default narration text
_DEFAULT_TEXT = (
    "Mira hacia atrás... no para lamentarte..."
    " sino para entender quién eres hoy. "
    "El dolor es inevitable..."
    " pero el sufrimiento es una elección."
)


def _build_narrate_parser(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add narration-specific arguments to a parser."""
    parser.add_argument(
        "--text",
        type=str,
        default=_DEFAULT_TEXT,
        help="Text to convert into speech (in Spanish).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Path to the output high-fidelity WAV file."
            " If omitted, uniquely generated under outputs/."
        ),
    )
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        default=DEFAULT_REFERENCE_VOICE,
        help=(
            "Path to the 10-20 second reference audio WAV/MP3 file to clone the speaker's voice."
        ),
    )
    parser.add_argument(
        "--speed",
        "-s",
        type=float,
        default=DEFAULT_SPEED,
        help=("Inference speed multiplier (e.g. 1.0 is standard, 0.8 is slower, 1.2 is faster)."),
    )
    parser.add_argument(
        "--master",
        action="store_true",
        default=False,
        help=("Apply automated audio mastering chain (HPF → Compressor → Limiter) to the output."),
    )
    return parser


def _build_prep_voice_parser(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add voice-prep-specific arguments to a parser."""
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help=("Path to the source reference audio file (MP3, WAV, FLAC, etc.)."),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=(
            "Path for the prepared output WAV. If omitted, saves as <input_basename>_prepped.wav."
        ),
    )
    return parser


def _detect_subcommand() -> str | None:
    """Check sys.argv for a known subcommand name.

    We inspect argv manually so that argparse doesn't reject
    top-level narration flags (--reference, --text, etc.) as
    invalid subcommand names when no subcommand is given.
    """
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        if arg in _KNOWN_SUBCOMMANDS:
            return arg
        # First non-flag token isn't a known subcommand → default mode
        break
    return None


def main():
    subcommand = _detect_subcommand()

    if subcommand == "prep-voice":
        # --- Voice preparation subcommand ---
        parser = argparse.ArgumentParser(
            description=("Prepare a reference audio clip for optimal XTTS v2 zero-shot cloning."),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable detailed debug logging.",
        )
        _build_prep_voice_parser(parser)

        # Remove the subcommand token so argparse doesn't choke
        filtered_argv = [a for a in sys.argv[1:] if a != "prep-voice"]
        args = parser.parse_args(filtered_argv)

        if args.debug:
            logger.setLevel(logging.DEBUG)
            logging.getLogger("focushub_narration.features").setLevel(logging.DEBUG)

        cmd_prep_voice(args)

    else:
        # --- Narration (default, or explicit "narrate") ---
        parser = argparse.ArgumentParser(
            description=(
                "FocusHub Narration - High-quality Spanish"
                " Text-to-Speech using XTTS v2"
                " Zero-Shot Cloning."
            ),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable detailed debug logging.",
        )
        _build_narrate_parser(parser)

        # Remove the optional subcommand token
        filtered_argv = [a for a in sys.argv[1:] if a != "narrate"]
        args = parser.parse_args(filtered_argv)

        if args.debug:
            logger.setLevel(logging.DEBUG)
            logging.getLogger("focushub_narration.features").setLevel(logging.DEBUG)

        cmd_narrate(args)


if __name__ == "__main__":
    main()
