# Multimodal learning through the eyes and ears of a single child

This repository contains code and models from the following papers:
- Vong, W. K., Wang, W., Orhan, A. E., & Lake, B. M (2024). Grounded language acquisition through the eyes and ears of a single child. *Science*.
- Wang, W., Vong, W. K., Kim, N., & Lake, B. M. (2023). Finding Structure in One Child's Linguistic Experience. *Cognitive Science*.

## Installation

This project uses Python 3.8, and [`uv`](https://docs.astral.sh/uv/) for dependency management. Follow these steps to set up the environment:

1. Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone git@github.com:wkvong/multimodal-baby.git
cd multimodal-baby
```

3. Create and set up the virtual environment:
```bash
# Create a virtual environment in this folder
uv venv

# Optional: If you need to create the environment in a custom location, run the following commands instead:
uv venv ${UV_CACHE_DIR}/multimodal_baby_env
export VIRTUAL_ENV=${UV_CACHE_DIR}/multimodal_baby_env

# Install dependencies
uv sync
```

4. Install additional requirements:
```bash
# Install CLIP from source
uv pip install git+https://github.com/openai/CLIP.git

# Download spaCy language model
uv run -- spacy download en_core_web_sm
```

5. Test the installation:
```bash
# Run the demo script to verify everything is working
uv run demo.py
```

## Usage
Usage of CVCL follows the [CLIP](https://github.com/openai/CLIP) API. The following code downloads the pre-trained CVCL model (trained on the SAYCam-S dataset) from HuggingFace Hub, and then encodes images and utterances using the model:

```python
import torch
from multimodal.multimodal_lit import MultiModalLitModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cvcl, preprocess = MultiModalLitModel.load_model(model_name="cvcl")
cvcl = cvcl.to(device)
cvcl.eval()

# create random image to encode
images = torch.rand(4, 3, 224, 224).to(device)
image_features = cvcl.encode_image(images)

# create texts to encode
texts = ["ball", "puzzle", "car"]
texts, texts_len = cvcl.tokenize(texts)
texts, texts_len = texts.to(device), texts_len.to(device)
texts_features = cvcl.encode_text(texts, texts_len)

# get logits from a batch of images and texts
logits_per_image, logits_per_text = cvcl(images, texts, texts_len)

print("Logits per image shape:", logits_per_image.shape)
print("Logits per text shape:", logits_per_text.shape)
```

## Figures
The code in `analysis_cvcl/figures.R` can be run to reproduce the main figures from the paper.

## Datasets
This project uses the SAYCam dataset described in the following paper: 

Sullivan J, Mei M, Perfors A, Wojcik EH, Frank MC (2021) [SAYCam: A large, longitudinal audiovisual dataset recorded from the infant's perspective.](https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00039/97495/SAYCam-A-Large-Longitudinal-Audiovisual-Dataset) Open Mind.

The original dataset is hosted on the [Databrary](https://nyu.databrary.org/) repository for behavioral science, along with the SAYCam-S and Labeled-S subsets used in this project. Unfortunately, we are unable to publicly share the SAYCam dataset here due to the terms of use. Interested researchers can apply for access to the dataset with approval from their institution's IRB.

## Citation

Thank you for checking out our work! If you use models or code from repo, please cite either:
- Vong, W. K., Wang, W., Orhan, A. E., and Lake, B. M (2024). Grounded language acquisition through the eyes and ears of a single child. Science.

or:

- Wang, W., Vong, W. K., Kim, N., and Lake, B. M. (2023). Finding Structure in One Child's Linguistic Experience. Cognitive Science, 47, e13305. 

## Acknowledgments
We are grateful for the authors of the SAYCam article, and the volunteers who contributed to the data set, for making our article possible. This work was supported by the DARPA Machine Common Sense program and NSF Award 1922658 NRT-HDR: FUTURE Foundations, Translation, and Responsibility for Data Science. 
