"""Advanced punctuation engineering for XTTS v2 cadence control.

XTTS v2 translates punctuation marks into physical speech commands:
pitch drops, intakes of breath, and pause lengths. This module normalizes
raw input text into XTTS-optimized punctuation before inference.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Maximum sentence length (in characters) before inserting a natural clause break.
MAX_SENTENCE_LENGTH = 120

# Common Spanish conjunctions/prepositions that serve as natural clause boundaries.
_CLAUSE_BREAK_WORDS = [
    " pero ",
    " sino ",
    " aunque ",
    " porque ",
    " ya que ",
    " mientras ",
    " cuando ",
    " donde ",
    " como ",
    " para ",
    " que ",
    " y ",
    " o ",
    " ni ",
    " sin embargo ",
    " no obstante ",
    " además ",
    " también ",
    " entonces ",
]


def _normalize_ellipses(text: str) -> str:
    """Normalize 3+ consecutive dots to exactly 2 dots for consistent reflective pauses.

    XTTS v2 interprets `..` as a controlled reflective pause. Three or more dots
    produce inconsistent, sometimes overly long pauses.
    """
    # Replace 3+ dots with exactly 2
    return re.sub(r"\.{3,}", "..", text)


def _normalize_em_dashes(text: str) -> str:
    """Normalize em-dashes and double-hyphens to spaced em-dashes for conversational shifts.

    XTTS v2 uses em-dashes as sudden, conversational thought breaks.
    Ensuring consistent spacing prevents the model from rushing or merging words.
    """
    # Normalize double-hyphens to em-dash
    text = text.replace("--", "—")
    # Ensure em-dashes have surrounding spaces
    text = re.sub(r"\s*—\s*", " — ", text)
    return text


def _add_exclamation_breathing_room(text: str) -> str:
    """Add spacing around exclamation marks to prevent clipping on opening syllables.

    Without spacing, XTTS v2 can rush or clip the first phoneme after ¡ and
    the last phoneme before !.
    """
    # Add space after opening ¡ if followed by a letter
    text = re.sub(r"¡(\w)", r"¡ \1", text)
    # Add space before closing ! if preceded by a letter
    text = re.sub(r"(\w)!", r"\1 !", text)
    return text


def _add_question_breathing_room(text: str) -> str:
    """Add spacing around question marks to prevent rushed question starts.

    Same principle as exclamation breathing room.
    """
    # Add space after opening ¿ if followed by a letter
    text = re.sub(r"¿(\w)", r"¿ \1", text)
    # Add space before closing ? if preceded by a letter
    text = re.sub(r"(\w)\?", r"\1 ?", text)
    return text


def _break_long_sentences(text: str, max_length: int = MAX_SENTENCE_LENGTH) -> str:
    """Insert commas at natural clause boundaries in overly long sentences.

    Long sentences without punctuation cause XTTS v2 to produce a monotone drone.
    This function identifies sentences exceeding max_length characters and inserts
    a comma at the first natural clause boundary (conjunction/preposition).

    Args:
        text: Input text.
        max_length: Maximum sentence length before attempting to insert a break.

    Returns:
        Text with clause breaks inserted where needed.
    """
    # Split on sentence-ending punctuation while keeping the delimiters
    sentences = re.split(r"(?<=[.!?])\s+", text)
    result = []

    for sentence in sentences:
        if len(sentence) > max_length:
            # Try to find a natural clause boundary near the middle
            best_pos = -1
            middle = len(sentence) // 2
            best_dist = len(sentence)

            for word in _CLAUSE_BREAK_WORDS:
                pos = sentence.lower().find(word.lower())
                while pos != -1:
                    dist = abs(pos - middle)
                    # Only break if the clause boundary is reasonably centered
                    if dist < best_dist and pos > 20:
                        best_dist = dist
                        best_pos = pos + len(word) - 1  # After the word + trailing space
                    pos = sentence.lower().find(word.lower(), pos + 1)

            if best_pos > 0:
                # Insert a comma before the clause boundary word
                # Find the actual word start (the space before it)
                insert_pos = best_pos - len(sentence[: best_pos + 1].split()[-1])
                # Only insert if there isn't already punctuation nearby
                if insert_pos > 0 and sentence[insert_pos - 1] not in ",.;:!?—":
                    sentence = sentence[:insert_pos].rstrip() + "," + sentence[insert_pos:]
                    logger.debug("Inserted clause break at position %d.", insert_pos)

        result.append(sentence)

    return " ".join(result)


def preprocess_text(text: str) -> str:
    """Transform raw input text into XTTS v2-optimized punctuation.

    Applies the following transformations in order:
    1. Normalize ellipses (3+ dots → exactly 2)
    2. Normalize em-dashes (double-hyphens → spaced em-dash)
    3. Add exclamation breathing room (spacing around ¡ and !)
    4. Add question breathing room (spacing around ¿ and ?)
    5. Break overly long sentences (insert commas at clause boundaries)
    6. Clean up any double spaces introduced by transformations

    Args:
        text: Raw input text in Spanish.

    Returns:
        Preprocessed text optimized for XTTS v2 inference.
    """
    original = text

    text = _normalize_ellipses(text)
    text = _normalize_em_dashes(text)
    text = _add_exclamation_breathing_room(text)
    text = _add_question_breathing_room(text)
    text = _break_long_sentences(text)

    # Clean up any double/triple spaces introduced by transformations
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    if text != original:
        logger.info("[✓] Text preprocessed for XTTS v2 cadence optimization.")
        logger.debug("Original: %s", original)
        logger.debug("Processed: %s", text)

    return text
