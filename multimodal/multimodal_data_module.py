from pathlib import Path
from typing import Any, Tuple
import json
import argparse

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import pytorch_lightning as pl

from multimodal.utils import GaussianBlur

import clip

# directories and filenames
# must be consistent with multimodal_saycam_data_module
EVAL_DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal")
EVAL_METADATA_FILENAME = "eval_dev.json"
# EVAL_DEV_METADATA_FILENAME = EVAL_DATA_DIR / "eval_dev.json"
# EVAL_TEST_METADATA_FILENAME = EVAL_DATA_DIR / "eval_test.json"

# default arguments
# dataloader arguments
BATCH_SIZE = 4
VAL_BATCH_SIZE = 16
NUM_WORKERS = 4
EVAL_INCLUDE_SOS_EOS = False

# evaluation arguments
N_VAL_DATALOADERS_PER_SPLIT = 2
TEST_WHILE_VAL = False
EVAL_TYPE = "image"

# sampling arguments
MAX_LEN_UTTERANCE = 25

# training arguments
AUGMENT_FRAMES = False

# special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
SOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3

# image arguments
IMAGE_H = 224
IMAGE_W = 224

# image transforms
normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIP_EVAL = False


def read_vocab(vocab_filename):
    with open(vocab_filename) as f:
        return json.load(f)


def load_data(filename):
    with open(filename) as f:
        data = json.load(f)
        return data['data']


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class MultiModalDataset(Dataset):
    """
    Abstract Dataset that returns paired image-utterances.
    """

    def __init__(self):
        super().__init__()

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Returns an image-utterance pair in tuple
        (img, utterance_idxs, utterance_length, raw_utterances).
        raw_utterances: a list of str, each of which is a sentence with
        space-separated tokens.
        """
        raise NotImplementedError


def multiModalDataset_collate_fn(batch):
    img, utterance_idxs, utterance_length, raw_utterance = zip(*batch)
    img = torch.stack(img, 0)
    utterance_idxs = pad_sequence(
        utterance_idxs, batch_first=True, padding_value=PAD_TOKEN_ID)
    utterance_length = torch.tensor(utterance_length, dtype=torch.long)
    if utterance_idxs.size(1) > MAX_LEN_UTTERANCE:
        utterance_idxs = utterance_idxs[:, :MAX_LEN_UTTERANCE]
        utterance_length = torch.minimum(
            utterance_length, torch.tensor(MAX_LEN_UTTERANCE, dtype=torch.long))
    raw_utterance = list(raw_utterance)
    return img, utterance_idxs, utterance_length, raw_utterance


class LabeledSEvalDataset(Dataset):
    """
    Dataset that returns a set of referents and a target word for evaluation
    """

    def __init__(self, data, vocab, transform, eval_include_sos_eos=False, clip_eval=False):
        self.data = data
        self.vocab = vocab
        self.transform = transform
        self.eval_include_sos_eos = eval_include_sos_eos
        self.clip_eval = clip_eval

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


class LabeledSTextEvalDataset(Dataset):
    """
    Dataset that returns a single referent and multiple target words for evaluation
    """

    def __init__(self, data, vocab, transform, eval_include_sos_eos=False, clip_eval=False):
        self.data = data
        self.vocab = vocab
        self.transform = transform
        self.eval_include_sos_eos = eval_include_sos_eos
        self.clip_eval = clip_eval

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


class MultiModalDataModule(pl.LightningDataModule):
    """
    The abstract data module consisting of images and the associated utterances.
    """

    def __init__(self, args=None) -> None:
        super().__init__()

        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.drop_last = self.args.get("drop_last", False)
        self.val_batch_size = self.args.get("val_batch_size", VAL_BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))
        self.augment_frames = self.args.get("augment_frames", AUGMENT_FRAMES)
        self.eval_include_sos_eos = self.args.get("eval_include_sos_eos",
                                                  EVAL_INCLUDE_SOS_EOS)
        self.test_while_val = self.args.get("test_while_val", TEST_WHILE_VAL)
        self.eval_type = self.args.get("eval_type", EVAL_TYPE)
        self.eval_metadata_filename = self.args.get(
            "eval_metadata_filename", EVAL_METADATA_FILENAME)
        self.clip_eval = self.args.get(
            "clip_eval", CLIP_EVAL)

        # check which metadata file is being used
        print(f"Using metadata file: {self.eval_metadata_filename}")

        if self.augment_frames:
            # add same augmentations as emin used
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    (IMAGE_H, IMAGE_W), scale=(0.2, 1.)),
                # transforms.RandomApply([transforms.ColorJitter(0.9, 0.9, 0.9, 0.5)], p=0.9),
                # transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalizer,
            ])
        elif self.clip_eval:
            print("Using CLIP transforms for evaluation")
            # use CLIP transforms (for CLIP evaluation only)
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
            print("Using base transforms")
            # just convert to tensor and normalize
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalizer,
            ])

        # keep base transform for val and test
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            normalizer,
        ])

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per train step."
        )
        parser.add_argument(
            "--drop_last", action="store_true", help="Drop the last not full batch."
        )
        parser.add_argument(
            "--val_batch_size", type=int, default=VAL_BATCH_SIZE, help="Number of examples to operate on per forward step during validation."
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="Number of additional processes to load data."
        )
        parser.add_argument(
            "--augment_frames", action="store_true", help="Apply data augmentation to images."
        )
        parser.add_argument(
            "--eval_include_sos_eos", action="store_true", help="Add <sos> and <eos> tokens during evaluation"
        )
        parser.add_argument("--test_while_val", action="store_true",
                            help="Evaluate test set during validation (for COCO only!)")
        parser.add_argument("--eval_type", type=str, default="image", choices=[
                            "image", "text"], help="Run evaluation using multiple images or multiple labels")
        parser.add_argument("--eval_metadata_filename", type=str,
                            default="eval_filtered_dev.json",
                            help="JSON file with metadata for (dev) evaluation split to use")
        parser.add_argument("--clip_eval", action="store_true",
                            help="Perform evaluation using CLIP")
        return parser

    # TODO: add relevant config details
    # def config(self):
    #     """Return important settings of the dataset, which will be passed to instantiate models."""
    #     return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}

    def prepare_data(self, *args, **kwargs) -> None:
        print("Calling prepare_data!")

    def setup(self, *args, **kwargs) -> None:
        print("Calling setup!")

        # read vocab
        vocab = self.read_vocab()

        # read and create image-text data splits (train/val/test)
        self.datasets = self.create_datasets(vocab)

        # read and create eval data splits (val/test)
        self.eval_datasets = self.create_eval_datasets(vocab)

    def read_vocab(self):
        raise NotImplementedError

    def create_datasets(self, vocab):
        raise NotImplementedError

    def create_eval_datasets(self, vocab):
        eval_datasets = {}

        eval_dev_metadata_filename = EVAL_DATA_DIR / self.eval_metadata_filename
        eval_test_metadata_filename = EVAL_DATA_DIR / \
            self.eval_metadata_filename.replace("dev", "test")

        for split, filename in [
                ("val", eval_dev_metadata_filename),
                ("test", eval_test_metadata_filename)]:
            data = load_data(filename)

            if self.eval_type == "image":
                dataset = LabeledSEvalDataset(
                    data, vocab, self.transform, self.eval_include_sos_eos, self.clip_eval)
            elif self.eval_type == "text":
                dataset = LabeledSTextEvalDataset(
                    data, vocab, self.transform, self.eval_include_sos_eos, self.clip_eval)

            eval_datasets[split] = dataset

        return eval_datasets

    def train_dataloader(self, batch_size=None, shuffle=True, drop_last=None):
        if batch_size is None:
            batch_size = self.batch_size
        if drop_last is None:
            drop_last = self.drop_last

        return DataLoader(
            self.datasets['train'],
            collate_fn=multiModalDataset_collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_test_dataloader(self, dataset, eval_dataset, batch_size=None,
                            shuffle=False, drop_last=False):
        if batch_size is None:
            batch_size = self.val_batch_size

        dataloader = DataLoader(
            dataset,
            collate_fn=multiModalDataset_collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=multiModalDataset_collate_fn,
            shuffle=shuffle,
            # batch_size=self.batch_size // 4,  # divide by 4 here since eval trials have 4 images
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        return [dataloader, eval_dataloader]

    def val_dataloader(self, batch_size=None, shuffle=False, drop_last=False):
        dataloaders = self.val_test_dataloader(
            self.datasets['val'],
            self.eval_datasets['val'],
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        if self.test_while_val:
            dataloaders += self.test_dataloader(
                batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

        return dataloaders

    def test_dataloader(self, batch_size=None, shuffle=False, drop_last=False):
        return self.val_test_dataloader(
            self.datasets['test'],
            self.eval_datasets['test'],
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )


def load_and_print_info(data_module_class):
    # parse args
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()

    # set up data module
    data = data_module_class(args)
    data.prepare_data()

    print(data)
