from pathlib import Path
import json
import glob
import os
from PIL import Image
import shutil
from collections import Counter
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import clip

from multimodal.multimodal_data_module import MAX_LEN_UTTERANCE, PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN_ID, UNK_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID, IMAGE_H, IMAGE_W, normalizer, multiModalDataset_collate_fn
from multimodal.multimodal_saycam_data_module import MultiModalSAYCamDataModule, DATA_DIR, TRAIN_METADATA_FILENAME

# directories and filenames
OBJECT_CATEGORIES_DATA_DIR = Path(
    "/home/wv9/code/WaiKeen/multimodal-baby/data/object_categories")
OBJECT_CATEGORIES_EVAL_METADATA_FILENAME = DATA_DIR / \
    "eval_object_categories.json"


class ObjectCategoriesEvalDataset(Dataset):
    def __init__(self, data, vocab, eval_include_sos_eos=False, clip_eval=False):
        self.data = data
        self.vocab = vocab
        self.eval_include_sos_eos = eval_include_sos_eos
        self.clip_eval = clip_eval

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_H, IMAGE_W),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalizer,
        ])

    def __getitem__(self, idx):
        # read trial information
        trial = self.data[idx]

        # read in images (target and foils)
        # target image is always the first index
        n_imgs = len(trial["foil_img_filenames"]) + 1
        imgs = torch.zeros((n_imgs, 3, IMAGE_H, IMAGE_W))

        target_img_filename = trial["target_img_filename"]
        imgs[0] = self.transform(Image.open(
            target_img_filename).convert("RGB"))

        for i, foil_img_filename in enumerate(trial["foil_img_filenames"]):
            imgs[i +
                 1] = self.transform(Image.open(foil_img_filename).convert("RGB"))

        # get target category index from vocab as a single utterance
        raw_label = trial["target_category"]

        if not self.clip_eval:
            # use SAYCam vocab/tokenizer
            label = [self.vocab[raw_label]]
            if self.eval_include_sos_eos:
                # label is [<sos>, label, <eos>] to match LM training
                label = [SOS_TOKEN_ID] + label + [EOS_TOKEN_ID]

            label = torch.LongTensor(label)
            label_len = len(label)
        else:
            # use CLIP tokenizer
            label = clip.tokenize(raw_label)
            label_len = len(label)

        return imgs, label, label_len, [raw_label]

    def __len__(self):
        return len(self.data)


class ObjectCategoriesDataModule(pl.LightningDataModule):
    """
    The data module associated with evaluation using naturalistic object categories.
    """

    def __init__(self) -> None:
        super().__init__()
        pass

    def prepare_data(self, *args, **kwargs) -> None:
        print("Calling prepare_data!")
        self.vocab = _get_vocab()
        self.object_categories = _get_object_categories(self.vocab)
        _move_test_items(self.object_categories)
        _generate_object_category_eval_metadata(self.object_categories)

    def setup(self, *args, **kwargs) -> None:
        print("Calling setup!")
        self.vocab = _get_vocab()
        with open(OBJECT_CATEGORIES_EVAL_METADATA_FILENAME) as f:
            data = json.load(f)
            self.data = data['data']

        self.eval_dataset = ObjectCategoriesEvalDataset(self.data, self.vocab)

    def test_dataloader(self, shuffle=False):
        eval_dataloader = DataLoader(
            self.eval_dataset,
            collate_fn=multiModalDataset_collate_fn,
            shuffle=shuffle,
            batch_size=1,
            num_workers=4,
            pin_memory=False,
        )
        return eval_dataloader


def _get_vocab():
    """Get vocab dict from SAYCam"""
    multimodal_dm = MultiModalSAYCamDataModule()
    vocab = multimodal_dm.read_vocab()
    return vocab


def _get_object_categories(vocab):
    """Get list of object categories"""
    object_categories = list(OBJECT_CATEGORIES_DATA_DIR.glob("*"))
    object_categories = [
        object_category.name for object_category in object_categories]

    # get object categories in saycam vocab
    object_categories_in_vocab = []
    for object_category in object_categories:
        if object_category in vocab:
            object_categories_in_vocab.append(object_category)

    return object_categories_in_vocab


def _move_test_items(object_categories):
    """Move test items into the parent dir for each object category"""
    for object_category in object_categories:
        test_dir = os.path.join(
            OBJECT_CATEGORIES_DATA_DIR / f"{object_category}/TestItems")
        test_items = glob.glob(f"{test_dir}/*.jpg")
        for test_item in test_items:
            test_filename = test_item.split("/")[-1]
            shutil.move(test_item, "..")


def _generate_object_category_eval_trial(idx, target_img_filename, target_category, object_categories, n_foils):
    """Generate a single evaluation trial for object category evaluation."""
    foil_categories = object_categories.copy()
    foil_categories.remove(target_category)
    foil_categories = np.random.choice(
        foil_categories, size=n_foils, replace=False)

    foil_img_filenames = []
    for i in range(n_foils):
        foil_imgs = list(Path(OBJECT_CATEGORIES_DATA_DIR /
                              f"{foil_categories[i]}").glob("*.jpg"))
        foil_img_filename = np.random.choice(foil_imgs).as_posix()
        foil_img_filenames.append(foil_img_filename)

    # save trial info as a dict
    eval_trial = {}
    eval_trial["trial_num"] = idx
    eval_trial["target_category"] = target_category
    eval_trial["target_img_filename"] = target_img_filename.as_posix()
    eval_trial["foil_categories"] = list(foil_categories)
    eval_trial["foil_img_filenames"] = foil_img_filenames
    return eval_trial


def _generate_object_category_eval_metadata(object_categories):
    """Create splits for evaluating Multimodal SAYCam models on object categories"""

    if os.path.exists(OBJECT_CATEGORIES_EVAL_METADATA_FILENAME):
        print("Object categories evaluation metadata files have already been created. Skipping this step.")
    else:
        print("Creating metadata files for object categories evaluation.")

        n_evaluations_per_example = 5
        n_foils = 3

        eval_dataset = []
        for target_category in object_categories:
            target_img_filenames = Path(
                OBJECT_CATEGORIES_DATA_DIR / f"{target_category}").glob("*.jpg")
            for target_img_filename in target_img_filenames:
                for i in range(n_evaluations_per_example):
                    eval_trial = _generate_object_category_eval_trial(
                        i, target_img_filename, target_category,
                        object_categories, n_foils)
                    eval_dataset.append(eval_trial)

        # save dataset as JSON
        eval_dict = {"data": eval_dataset}
        with open(OBJECT_CATEGORIES_EVAL_METADATA_FILENAME, "w") as f:
            json.dump(eval_dict, f)


def _get_object_category_word_counts():
    """Get word frequency for each object category."""
    with open(TRAIN_METADATA_FILENAME) as f:
        train_data = json.load(f)
        train_data = train_data["data"]

    words = []
    for data in train_data:
        utterance = data['utterance'].split(' ')
        for word in utterance:
            words.append(word)

    counts = Counter(words)
    object_category_counts = {}
    for object_category in object_categories_in_vocab:
        object_category_counts[object_category] = counts[object_category]
    return object_category_counts
