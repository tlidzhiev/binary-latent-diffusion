# Binary Latent Diffusion

## Installation

### Requirements
- Python 3.12.11
- CUDA (optional, for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/tlidzhiev/binary-latent-diffusion.git
cd binary-latent-diffusion

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.12.11
source .venv/bin/activate

# Install dependencies via uv
uv sync --all-groups

# Install pre-commit
pre-commit install
```

## Credits

The code of this project is based on the [PyTorch project template](https://github.com/Blinorot/pytorch_project_template).
