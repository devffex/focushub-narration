# FocusHub Narration 🎙️

Premium, high-fidelity local Spanish text-to-speech engine powered by **XTTS v2**. FocusHub Narration provides studio-grade, expressive, and deeply natural Latin American (LATAM) voice synthesis using state-of-the-art **zero-shot voice cloning**. 

Instead of generic pre-packaged digital models, you can clone any deep, robust, "god-like" speaker's voice using a simple 10-20 second audio reference clip.

---

## ⚡ Quick Start

### 1. Prerequisites
Ensure you have [uv](https://github.com/astral-sh/uv) installed (a fast Python package installer and resolver).

### 2. Setup the Environment
Coqui TTS is compiled for **Python 3.11**. Create the local virtual environment targeting Python 3.11 and synchronize dependencies:

```bash
# Initialize and sync packages
uv venv --python 3.11
uv sync
```

*Note: XTTS v2 model weights (~1.8 GB) will be automatically downloaded from the Hugging Face Hub on your very first synthesis run.*

---

## 🚀 Execution & Usage

Because the package is structured under a `src/` directory, you must define the `PYTHONPATH` when executing the script.

### Basic Generation
Synthesize audio using the default stoic text and the default reference voice (`voices/epic_male.wav`):

```bash
PYTHONPATH=src uv run --active python main.py
```

### Custom Text & Zero-Shot Voice Cloning
Pass a custom text string and clone the premium narrator voice:

```bash
PYTHONPATH=src uv run --active python main.py \
  --reference voices/alberto_rodriguez.mp3 \
  --text "El dolor es inevitable... pero el sufrimiento es una elección."
```

### Unique Output Audios & Audios Catalog
To prevent overriding previous results, FocusHub Narration **automatically generates unique, timestamped filenames** inside the `outputs/` directory (e.g. `outputs/narration_20260525_150551.wav`) if you do not specify an explicit output filename via `--output` / `-o`.

Additionally, every successful audio synthesis is dynamically logged into the **[audios_catalog.md](file:///home/julio/Work/devffex/quirky-shannon/audios_catalog.md)** index file in your root folder. This provides an organized, professional history tracking:
*   Timestamp
*   Markdown link to play/access the exact WAV file
*   Reference voice used
*   Speed
*   Text snippet spoken

### Premium Stoic & Biographical Narration
Here is the optimized motivational narration script, showcasing the voice's emotional range, natural pacing, and authoritative delivery using the default voice (`voices/alberto_rodriguez.mp3`):

```bash
PYTHONPATH=src uv run --active python main.py --voice voices/alberto_rodriguez.mp3 --text "Mira hacia atrás... no para lamentarte... sino para entender quién eres hoy. Él lo había perdido todo. La fortuna, su estatus, su patria... todo se desvaneció en un solo día frente a las costas del mar Egeo. Pero en ese abismo de silencio y desesperación, en lugar de maldecir su destino, eligió la calma. Recordó las palabras de los antiguos sabios: no podemos controlar lo que nos pasa, pero sí gobernamos la respuesta de nuestra propia alma. Fue así como Zenón de Citio transformó la ruina en virtud, dando origen al estoicismo. Su vida nos enseña que el dolor es inevitable, pero el sufrimiento es una elección. Levántate... respira hondo... y abraza tu destino con valentía. Porque tu mayor victoria no consiste en no caer jamás, sino en levantarte con más fuerza y dignidad después de cada tormenta. El fuego no destruye al oro... simplemente revela su verdadera pureza."
```

### Reference Audio Guidelines
To get clean, professional, "god-like" vocals:
1. Find a high-quality audio clip of the target voice.
2. Ensure there is **no background music, noise, echo, or sound effects**.
3. Place it under the `voices/` directory.

The following custom, robust voice is configured in your folder:
*   `voices/alberto_rodriguez.mp3` (Default - Spanish male, deep narration tone)

Call the script specifying this voice:
```bash
PYTHONPATH=src uv run --active python main.py -r voices/alberto_rodriguez.mp3
```

### Optional Speed Adjustment
Tweak the speed multiplier using `--speed` (e.g. `0.95` for slightly slower and more dramatic pacing):

```bash
PYTHONPATH=src uv run --active python main.py --speed 0.95
```

### Debug Mode
To inspect device selection (CUDA/VRAM) or deep learning synthesizer operations, append `--debug`:

```bash
PYTHONPATH=src uv run --active python main.py --debug
```
