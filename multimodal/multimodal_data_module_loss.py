from pathlib import Path
from typing import Any, Tuple
import json
import argparse
import re

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import pytorch_lightning as pl

from multimodal.utils import GaussianBlur

# directories and filenames
# must be consistent with multimodal_saycam_data_module
#EVAL_DATA_DIR = Path("/saycam/S_multimodal")
EVAL_DATA_DIR = Path("/saycam")
EVAL_DEV_METADATA_FILENAME = EVAL_DATA_DIR / "eval_dev.json"
EVAL_TEST_METADATA_FILENAME = EVAL_DATA_DIR / "eval_test.json"

# default arguments
# dataloader arguments
BATCH_SIZE = 4
VAL_BATCH_SIZE = 16
NUM_WORKERS = 4
EVAL_INCLUDE_SOS_EOS = False

# evaluation arguments
N_VAL_DATALOADERS_PER_SPLIT = 2
TEST_WHILE_VAL = True

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


def read_vocab(vocab_filename):
    with open(vocab_filename) as f:
        return json.load(f)


def load_data(filename):
    with open(filename) as f:
        data = json.load(f)
        return data['data']


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

    def __init__(self, data, vocab, eval_include_sos_eos=False):
        self.data = data
        self.vocab = vocab
        self.eval_include_sos_eos = eval_include_sos_eos
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalizer,
        ])

    def __getitem__(self, idx):
        # read trial information
        trial = self.data[idx]

        # read in images (target and foils)
        # target image is always the first index
        imgs = torch.zeros((4, 3, IMAGE_H, IMAGE_W))
        target_img_filename = trial["target_img_filename"]
        target_img_filename = re.sub('\/misc\/vlgscratch4\/LakeGroup\/shared\_data', '/saycam', target_img_filename)
        #target_img_filename = re.sub('\/misc\/vlgscratch4\/LakeGroup\/WaiKeen\/multimodal\-baby\/data', '/saycam/S_multimodal', target_img_filename)
        target_img_filename = re.sub('\/misc\/vlgscratch4\/LakeGroup\/WaiKeen\/multimodal\-baby\/data', '/saycam', target_img_filename)
        target_img_filename = re.sub('\/saycam\/S_multimodal', '/saycam', target_img_filename)
        imgs[0] = self.transform(Image.open(
            target_img_filename).convert("RGB"))

        for i, foil_img_filename in enumerate(trial["foil_img_filenames"]):
            foil_img_filename = re.sub('\/misc\/vlgscratch4\/LakeGroup\/shared\_data', '/saycam', foil_img_filename)
            #foil_img_filename = re.sub('\/misc\/vlgscratch4\/LakeGroup\/WaiKeen\/multimodal\-baby\/data', '/saycam/S_multimodal', foil_img_filename)    
            foil_img_filename = re.sub('\/misc\/vlgscratch4\/LakeGroup\/WaiKeen\/multimodal\-baby\/data', '/saycam', foil_img_filename)          
            foil_img_filename = re.sub('\/saycam\/S_multimodal', '/saycam', foil_img_filename)    
            imgs[i +
                 1] = self.transform(Image.open(foil_img_filename).convert("RGB"))

        # get target category index from vocab as a single utterance
        raw_label = trial["target_category"]
        label = [self.vocab[raw_label]]
        if self.eval_include_sos_eos:
            # label is [<sos>, label, <eos>] to match LM training
            label = [SOS_TOKEN_ID] + label + [EOS_TOKEN_ID]

        label = torch.LongTensor(label)
        label_len = len(label)

        return imgs, label, label_len, [raw_label]

    def __len__(self):
        return len(self.data)


class MultiModalDataModule(pl.LightningDataModule):
    """
    The abstract data module consisting of images and the associated utterances.
    """

    def __init__(self, args=None) -> None:
        super().__init__(args)

        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.drop_last = self.args.get("drop_last", False)
        self.val_batch_size = self.args.get("val_batch_size", VAL_BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))
        self.augment_frames = self.args.get("augment_frames", AUGMENT_FRAMES)
        self.eval_include_sos_eos = self.args.get("eval_include_sos_eos",
                                                  EVAL_INCLUDE_SOS_EOS)

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
        else:
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
        # self.eval_datasets = self.create_eval_datasets(vocab)

    def read_vocab(self):
        raise NotImplementedError

    def create_datasets(self, vocab):
        raise NotImplementedError

    def create_eval_datasets(self, vocab):
        eval_datasets = {}

        for split, filename in [
                ("val", EVAL_DEV_METADATA_FILENAME),
                ("test", EVAL_TEST_METADATA_FILENAME)]:
            data = load_data(filename)
            dataset = LabeledSEvalDataset(
                data, vocab, self.eval_include_sos_eos)
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

    def val_test_dataloader(self, dataset, batch_size=None,
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

        return dataloader

    def val_dataloader(self, batch_size=None, shuffle=False, drop_last=False):
        dataloaders = self.val_test_dataloader(
            self.datasets['val'],
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return dataloaders

    def test_dataloader(self, batch_size=None, shuffle=False, drop_last=False):
        return self.val_test_dataloader(
            self.datasets['test'],
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
