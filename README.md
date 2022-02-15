# Multimodal Learning on SAYCam Dataset

This is a project for multimodal learning on SAYCam dataset ([Sullivan et al., 2020](https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00039/97495/SAYCam-A-Large-Longitudinal-Audiovisual-Dataset)).
The original data is on [Databrary](https://nyu.databrary.org/volume/564).

## Requirements
- Python 3
- PyTorch (along with torchvision)
- PyTorch Lightning
- other packages for preprocessing, data loading, logging, evaluation, and visualization (see [requirements.txt](requirements.txt))

## Files and Modules
- [train.py](train.py): entrance for training.
- [multimodal/](multimodal): modules for models, data preprocessing and loading, and other utilities.
  - [multimodal_lit.py](multimodal/multimodal_lit.py): the overarching PyTorch Lightning model `MultiModalLitModel`, which contains main code for building the model, configuring the optimizer, computing the loss, training and validation.
  - [multimodal.py](multimodal/multimodal.py): model components (`VisionEncoder`, `TextEncoder`) and sub-models (`MultiModalModel`, `LanguageModel`).
  - [multimodal_data_module.py](multimodal/multimodal_data_module.py): abstract code for loading datasets.
  - [multimodal_saycam_data_module.py](multimodal/multimodal_saycam_data_module.py): code for preprocessing and loading SAYCam dataset.
  - [coco_captions_data_module.py](multimodal/coco_captions_data_module.py): code for preprocessing and loading COCO Captions dataset.
  - [utils.py](multimodal/utils.py): miscellaneous utilities.
  - [beam_search.py](multimodal/beam_search.py): beam search.
  - [textgen_eval.py](multimodal/textgen_eval.py): evaluate generated text by [pycocoevalcap](https://github.com/salaniz/pycocoevalcap) (COCO Caption evaluation tools).
- [ngram.py](ngram.py): N-gram language model.
- [tests/](tests): testing.
- [notebooks/](notebooks): Jupyter notebooks for search and visualization.
  - [search_engine.ipynb](notebooks/search_engine.ipynb): search a word in the dataset and visualize found examples. Also some dataset statistics.
  - [lm_clustering.ipynb](notebooks/lm_clustering.ipynb): following [Elman (1990)](https://crl.ucsd.edu/~elman/Papers/fsit.pdf), investigating the representations of words by clustering and plotting dendrograms and T-SNE.
- [runner.py](runner.py): create Slurm scripts (and submit as Slurm jobs) given configurations.

## Models and Objectives
### VisionEncoder
The visual encoder for images. The architecture is based on ResNeXt ([Xie et al, 2017](https://arxiv.org/abs/1611.05431)), though other architectures in [torchvision.models](https://pytorch.org/vision/stable/models.html) are also possible (`--cnn_model <model>`).

For the SAYCam dataset, we use pre-trained self-supervised ResNeXt CNN from [Orhan et al. (2020)](https://arxiv.org/abs/2007.16189) (by `--pretrained_cnn`). You have to download this pre-trained model into `models/TC-S-resnext.tar` or designate the path to the model by `--cnn_model <path>`.

For the COCO Captioning dataset, we use pre-trained ResNeXt (by `--pretrained_cnn --cnn_model resnext50_32x4d`). Using the aforementioned pre-trained model results in significantly inferior performance.

### TextEncoder
The text encoder for text. Currently supported architectures are LSTM, bi-LSTM, CBOW, simple embedding (by `--text_encoder <model>`).
To generate text (captions), the text encoder must be uni-diretional. The uni-directional model can condition on the image features so that it turns into an image-captioning model (`--captioning`).

### MultimodalModel
The muitimodal contrastive model. Encode pairs of image and text, compute pair-wise similarities, and compute InfoNCE loss ([van den Oord, et al., 2018](https://arxiv.org/abs/1807.03748)).

### LanguageModel
The language model. Encode the text, project the hidden representations to distributions over the vocabulary, and compute the cross-entropy loss with ground-truth text.

When the text encoder is unidirectional, it can generate text by beam search decoding.

### MultiModalLitModel
  Contains:
  - a `VisionEncoder` `vision_encoder`,
  - a `TextEncoder` `text_encoder`,
  - a `MultimodalModel` `model` built over `vision_encoder` and `text_encoder`, and
  - a `LanguageModel` `language_model` built over `text_encoder`.

  These sub-models share the parameters, so `MultiModalLitModel` does not contain duplicated parameters.

#### Joint Loss Objective
The final joint loss objective is:

loss_joint = lambda_mm * loss_mm + lambda_lm * loss_lm

where loss_mm is the multimodal contrastive loss, loss_lm is the language modeling cross-entropy loss, and lambda_mm and lambda_lm are the weights respectively.

You can train only one of the loss by setting the weight of the other to 0. If you would like to save computation for the other loss, add `--optimize_unused`.

## Datasets and Data Modules
There is an abstract PyTorch Lightning data module `MultiModalDataModule` and two data modules, `MultiModalSAYCamDataModule` and `COCOCaptionsDataModule`, inheriting from the abstract module for each of the datasets.
Each instance of `MultiModalDataModule` contains train/val/test splits of the dataset in its `datasets` dict. Each split is a dataset object described below.

Similarly, there is an abstract PyTorch Dataset `MultiModalDataset` class and two dataset classes, `MultiModalSAYCamDataset` and `COCOCaptionsDataset`, inheriting from the abstract dataset class.
There is an additional `LabeledSEvalDataset` which returns a set of referents and a target word for evaluation.

By `--augment_frames`, images are augmented by a set of transforms during training.

### SAYCam Dataset
Since there are multiple frames paired with one utterance in each example of the SAYCam dataset, we have to choose one to return every time. The default is to choose the first frame; adding `--multiple_frames` chooses a random frame when training.

### COCO Captions Dataset
We use the widely adopted Karpathy splits ([Karpathy & Fei-Fei, 2017](https://arxiv.org/abs/1412.2306)).
We prune the vocabulary by dropping any word that has count less than 5 and end up with a vocabulary size of 9490.

Similarly, since there are multiple captions paired with one image in each example of the COCO Captions dataset, we have to choose one to return every time. The default is to choose the first caption; adding `--multiple_captions` chooses a random caption when training.

## Training
This is an example to train a joint objective model:
```bash
python train.py --dataset saycam \
--lambda_mm 0.5 --lambda_lm 0.5 \
--sim mean --embedding_type flat --normalize_features \
--pretrained_cnn --multiple_frames --augment_frames \
--text_encoder lstm --embedding_dim 512 --tie True --bias True --dropout_i 0.5 --dropout_o 0.0 \
--fix_temperature --temperature 0.07 \
--batch_size 8 --optimizer AdamW --lr 0.003 --lr_scheduler --weight_decay 0.05 --val_batch_size 16 --drop_last \
--seed 0 \
--optimize_unused \
--max_epochs 200 \
--gpus 1 --num_workers 8 \
--checkpoint_callback True --logger True \
--exp_name joint_dataset_saycam_lambda_mm_0.5_lambda_lm_0.5_sim_mean_embedding_type_flat_text_encoder_lstm_embedding_dim_512_dropout_i_0.5_dropout_o_0.0_batch_size_8_optimizer_AdamW_lr_0.003_lr_scheduler_True_weight_decay_0.05_val_batch_size_16_seed_0
```

For the detailed explanation of all arguments, run `python train.py -h`.

## [runner.py](runner.py)
This is to make running multiple experiments and writing Slurm scripts/submitting Slurm jobs easier.

It takes a configuration file, generate Slurm scripts (and optionally submit as Slurm jobs) with organized names for all configurations designated by the configuration file.

The configuration file is a Python file containing two entities:
- `grids`:
  A list of dicts. Each dict is a grid (for grid search), containing str keys and list of possible values.
  The runner will turn each str key into an command-line argument (flag) with one possible value in the corresponding list.
  For bool values, the runner will turn `True` values into a flag without argument.
  There is one special key `"main_file"` for the Python file name to run; in this case it should map to a single value `["train"]`.
  The runner will enumerate all possible combinations of the values in a grid (i.e., grid search) and generate one Slurm script for each combination.
- `flags`:
  A list of flags.
  We need a unique and easy-to-recognize name for the Slurm script, job, output files, etc. for each script.
  The runner build the name in the form `{basename}_{flag1}_{value1}_{flag2}_{value2}...`.
  `flags` list designates which flags to include in the name and their orders.
  `flags` need not contain all flags. You may intentionally omit some flags in `flags` if they are unimportant for distinguishing runs, so the name is shorter and easier to read.
 
