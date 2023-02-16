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
DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal")
OBJECT_CATEGORIES_DATA_DIR = DATA_DIR / "object_categories"
OBJECT_CATEGORIES_RESIZED_DATA_DIR = DATA_DIR / "object_categories_resized"
OBJECT_CATEGORIES_EVAL_METADATA_FILENAME = DATA_DIR / \
    "eval_object_categories.json"


class ObjectCategoriesEvalDataset(Dataset):
    def __init__(self, data, vocab, eval_include_sos_eos=False, clip_eval=False):
        self.data = data
        self.vocab = vocab
        self.eval_include_sos_eos = eval_include_sos_eos
        self.clip_eval = clip_eval

        if self.clip_eval:
            print("Using CLIP transforms for evaluation")
            # use CLIP transforms
            self.transform = transforms.Compose([
                transforms.Resize(
                    IMAGE_H, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(IMAGE_H),
                # _convert_image_to_rgb,  # commeting out since we convert to RGB
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            print("Using base transforms for evaluation")
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

class ObjectCategoriesTextEvalDataset(Dataset):
    """
    Dataset that returns a single target image and multiple category labels for evaluation
    """

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

        # read in target image
        img = torch.zeros((1, 3, IMAGE_H, IMAGE_W))
        target_img_filename = trial["target_img_filename"]
        img[0] = self.transform(Image.open(target_img_filename).convert("RGB"))

        # get target category and foil categories
        raw_target_label = trial["target_category"]
        raw_foil_labels = trial["foil_categories"]
        raw_labels = [raw_target_label] + raw_foil_labels
        labels = []
        labels_len = []
        for raw_label in raw_labels:
            if not self.clip_eval:
                # use SAYCam vocab/tokenizer
                label = [self.vocab[raw_label]]
                if self.eval_include_sos_eos:
                    label = [SOS_TOKEN_ID] + label + [EOS_TOKEN_ID]
                labels.append(label)
                labels_len.append(len(label))
            else:
                # use CLIP tokenizer
                label = clip.tokenize(raw_label)
                labels.append(label)
                labels_len.append(len(label))

        if not self.clip_eval:
            # convert list of labels to tensor
            labels = torch.LongTensor(labels)
        else:
            # labels are already tensors, so need to concatenate
            labels = torch.cat(labels, dim=0)

        return img, labels, labels_len, [raw_target_label]
        

    def __len__(self):
        return len(self.data)




class ObjectCategoriesDataModule(pl.LightningDataModule):
    """
    The data module associated with evaluation using naturalistic object categories.
    """

    def __init__(self, args=None) -> None:
        super().__init__()
        self.eval_type = args.eval_type
        self.clip_eval = args.clip_eval
        
    def prepare_data(self, *args, **kwargs) -> None:
        print("Calling prepare_data!")
        self.vocab = _get_vocab()
        self.object_categories = _get_object_categories(self.vocab)
        # _move_test_items(self.object_categories)  # skip this step
        # _resize_images(self.object_categories)  # skip this step
        _generate_object_category_eval_metadata(self.object_categories)

    def setup(self, *args, **kwargs) -> None:
        print("Calling setup!")
        self.vocab = _get_vocab()
        with open(OBJECT_CATEGORIES_EVAL_METADATA_FILENAME) as f:
            data = json.load(f)
            self.data = data['data']

        if self.eval_type == "image":
            self.eval_dataset = ObjectCategoriesEvalDataset(self.data, self.vocab, clip_eval=self.clip_eval)
        elif self.eval_type == "text":
            self.eval_dataset = ObjectCategoriesTextEvalDataset(self.data, self.vocab, clip_eval=self.clip_eval)

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
    print("Moving test items!")
    for object_category in object_categories:
        test_dir = os.path.join(
            OBJECT_CATEGORIES_DATA_DIR / f"{object_category}/TestItems")
        test_items = glob.glob(f"{test_dir}/*.jpg")
        for test_item in test_items:
            test_filename = test_item.split("/")[-1]
            shutil.move(test_item, "..")

def _resize_images(object_categories):
    """Resize Brady stimuli to be 50% smaller"""
    print("Resizing object category images!")
    os.makedirs(OBJECT_CATEGORIES_RESIZED_DATA_DIR, exist_ok=True)

    for object_category in object_categories:
        # create dir for object category
        category_dir = OBJECT_CATEGORIES_DATA_DIR / f"{object_category}"
        category_resized_dir = OBJECT_CATEGORIES_RESIZED_DATA_DIR / f"{object_category}"
        os.makedirs(category_resized_dir, exist_ok=True)

        # resize images and save to new dir
        for img_filename in glob.glob(f"{category_dir}/*.jpg"):
            img = Image.open(img_filename)
            img = img.resize((int(IMAGE_W / 2), int(IMAGE_H / 2)), Image.BICUBIC)
            new_img = Image.new('RGB', (IMAGE_W, IMAGE_H), 'white')
            new_img.paste(img, (int(IMAGE_W / 4), int(IMAGE_H / 4)))
            new_img.save(f"{category_resized_dir}/{img_filename.split('/')[-1]}")
            

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
    """Get word frequency for each object category from the training set."""
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
