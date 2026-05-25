import argparse
import logging
import sys

from focushub_narration.config import DEFAULT_VOICE, SPANISH_VOICES
from focushub_narration.features.tts import NarrationPipeline, save_audio

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("focushub_narration")


def main():
    parser = argparse.ArgumentParser(
        description="FocusHub Narration - High-quality Spanish Text-to-Speech using Kokoro-82M.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--text",
        type=str,
        default=(
            "Hola. ¿Cómo estás? Me alegra mucho saludarte hoy. "
            "Esta es una prueba de voz generada localmente en tu computadora. "
            "Como puedes notar, la entonación no es robótica; tiene pausas naturales "
            "y un ritmo fluido ideal para narraciones de alta calidad."
        ),
        help="Text to convert into speech (in Spanish).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="resultado_latam.wav",
        help="Path to the output high-fidelity WAV file.",
    )
    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        choices=list(SPANISH_VOICES.keys()),
        default=DEFAULT_VOICE,
        help=f"Voice character to use. Available voices: {list(SPANISH_VOICES.keys())}",
    )
    parser.add_argument(
        "--speed",
        "-s",
        type=float,
        default=1.0,
        help="Inference speed multiplier (e.g. 1.0 is standard, 0.8 is slower, 1.2 is faster).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging.",
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("focushub_narration.features").setLevel(logging.DEBUG)

    logger.info("Starting FocusHub Narration Pipeline...")
    logger.info("Target Output: %s", args.output)
    logger.info("Selected Voice: %s (%s)", args.voice, SPANISH_VOICES[args.voice])

    # 1. Initialize the modular Pipeline wrapper
    narrator = NarrationPipeline(lang_code="es")

    # 2. Generate the audio segments (utilizes Kokoro's safety chunking for 4GB GPU/VRAM limits)
    logger.info("Running text-to-speech inference...")
    segments = []
    try:
        for segment in narrator.generate(
            text=args.text,
            voice=args.voice,
            speed=args.speed,
        ):
            segments.append(segment)
    except Exception as e:
        logger.error("An error occurred during inference: %s", e, exc_info=True)
        sys.exit(1)

    # 3. Concatenate and save high fidelity WAV file
    success = save_audio(segments=segments, output_filename=args.output)
    if success:
        logger.info("FocusHub Narration completed successfully! [✓]")
    else:
        logger.error("FocusHub Narration failed to complete. [X]")
        sys.exit(1)


if __name__ == "__main__":
    main()
