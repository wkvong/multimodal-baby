from pathlib import Path
from typing import Any, Tuple
from collections import Counter
import json

import numpy as np
import torch

from multimodal.multimodal_data_module import MultiModalDataset, \
    MultiModalDataModule, read_vocab, load_and_print_info, \
    PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, \
    PAD_TOKEN_ID, UNK_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID
from multimodal.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# directories and filenames
# use CHILDES single child data
DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/CHILDES/sarah")

# default arguments
# dataset arguments
TRAIN_FRAC = 0.8
VAL_FRAC = 0.2


def read_lines(filename):
    with open(filename, 'r') as f:
        yield from map(str.strip, f)


class TextOnlyDataset(MultiModalDataset):
    """
    Dataset that returns utterances, but in the form as MultiModalDataset:
    paired image-utterances where image is None.
    """

    def __init__(self, data, vocab):
        super().__init__()
        self.data = data
        self.vocab = vocab

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Returns an image-utterance pair in tuple
        (img, utterance_idxs, utterance_length, raw_utterances)
        where img is None
        """

        # get utterance and convert to indices
        utterance = self.data[idx]
        utterance_words = utterance.split()
        utterance_words = [SOS_TOKEN] + utterance_words + [EOS_TOKEN]
        utterance_length = len(utterance_words)
        utterance_idxs = torch.tensor(
            [self.vocab.get(word, UNK_TOKEN_ID) for word in utterance_words],
            dtype=torch.long
        )

        return None, utterance_idxs, utterance_length, [utterance]


class TextOnlyDataModule(MultiModalDataModule):
    """
    A data module consisting of text utterances.
    """

    def __init__(self, args=None) -> None:
        super().__init__(args)
        self.data_dir = self.args.get("data_dir", DATA_DIR)
        self.vocab_filename = self.data_dir / "vocab.json"
        self.raw_filename = self.data_dir / f"{self.data_dir.stem}.txt"
        self.split_filenames = {
            "train": self.data_dir / "train.txt",
            "val": self.data_dir / "valid.txt",
            "test": self.data_dir / "test.txt",
        }

    @staticmethod
    def add_additional_to_argparse(parser):
        return parser

    @staticmethod
    def add_to_argparse(parser):
        parser = super(TextOnlyDataModule, TextOnlyDataModule).add_to_argparse(
            parser)
        parser = TextOnlyDataModule.add_additional_to_argparse(parser)
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        super().prepare_data(*args, **kwargs)
        _split_data(self.raw_filename, self.split_filenames)
        _create_vocab(self.vocab_filename, self.raw_filename)

    def read_vocab(self):
        return read_vocab(self.vocab_filename)

    def create_datasets(self, vocab):
        datasets = {}

        for split, filename in self.split_filenames.items():
            if filename.exists():
                data = list(read_lines(filename))
                dataset = TextOnlyDataset(data, vocab)
                datasets[split] = dataset

        return datasets


def _split_data(
        raw_filename: Path, split_filenames: dict[str, Path],
        train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC,
        seed=0,
):
    """Creates data split files"""

    for split, filename in split_filenames.items():
        if filename.exists():
            print("Data split files have already been created. Skipping this step.")
            return

    print("Creating files for train, validation and test split.")

    utterances = list(read_lines(raw_filename))

    # shuffle utterances
    np.random.seed(seed)
    idxs = np.arange(len(utterances))
    np.random.shuffle(idxs)

    # split utterances into train/val/test
    train_n = int(len(utterances) * train_frac)
    val_n = int(round((len(utterances) - train_n) * (val_frac / (1 - train_frac))))
    split_points = [train_n, train_n + val_n]
    split_idxs = np.split(idxs, split_points)
    split_utterances = []
    for _idxs in split_idxs:
        _idxs.sort()
        _utterances = [utterances[i] for i in _idxs]
        split_utterances.append(_utterances)

    for split, _utterances in zip(
        ["train", "val", "test"], split_utterances
    ):
        filename = split_filenames[split]
        with open(filename, "w") as f:
            f.write('\n'.join(_utterances))


def _create_vocab(vocab_filename: Path, filename: Path, freq_threshold=0):
    """Create vocabulary object and save to file"""

    if vocab_filename.exists():
        print("Vocabulary file already exists. Skipping this step.")
        return

    print("Creating vocab.json file!")

    counter = Counter()

    # load utterances
    utterances = list(read_lines(filename))

    # get token frequency
    for utterance in utterances:
        tokens = utterance.split()
        counter.update(tokens)

    # sort by frequency
    vocab = sorted(counter.most_common(),
                    key=lambda item: (-item[1], item[0]))

    # create vocab
    special_token_and_ids = [
        (PAD_TOKEN, PAD_TOKEN_ID),
        (UNK_TOKEN, UNK_TOKEN_ID),
        (SOS_TOKEN, SOS_TOKEN_ID),
        (EOS_TOKEN, EOS_TOKEN_ID),
    ]
    special_tokens = [token for token, token_id in special_token_and_ids]
    vocab = special_tokens + \
        [token for token, freq in vocab
            if token not in special_tokens and freq >= freq_threshold]
    # check consistency of special tokens
    for token, token_id in special_token_and_ids:
        assert vocab[token_id] == token

    # create vocab dict
    vocab_dict = {token: idx for idx, token in enumerate(vocab)}

    # save as JSON file
    with open(vocab_filename, "w") as f:
        json.dump(vocab_dict, f)


if __name__ == "__main__":
    load_and_print_info(TextOnlyDataModule)
