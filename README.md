# Attention Visualization

This is a simple implementation of attention visualization for image captioning per token using the VisionEncoderDecoderModel from the Hugging Face Transformers library.

# Installation

make sure you have `uv` installed
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

then clone the repository and sync the dependencies
```bash
git clone https://github.com/Slipstream-Max/dl_attn_vis.git
cd dl_attn_vis
uv sync
```

# Usage
just run the script
```bash
uv run main.py -i path/to/image.jpg
```
or
```bash
source .venv/bin/activate
python main.py -i path/to/image.jpg
```


