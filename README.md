# Multimodal learning through the eyes and ears of a single child

This repository contains code and models from the following papers:
- Vong, W. K., Wang, W., Orhan, A. E., & Lake, B. M (2024). Grounded language acquisition through the eyes and ears of a single child. *Science*.
- Wang, W., Vong, W. K., Kim, N., & Lake, B. M. (2023). Finding Structure in One Child's Linguistic Experience. *Cognitive Science*.

## Requirements
* torch==2.0.1
* torchvision==0.15.2
* pytorch-lightning==1.6.0
* spacy==3.0.0
* clip==1.0
* huggingface_hub==0.17.3
* Some other packages for preprocessing, evaluation and visualization may be required, see `requirements.txt`

Slightly older or newer versions will probably work as well.

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
