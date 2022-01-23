from typing import Any, Tuple
from pathlib import Path
import json
import argparse
from collections import Counter

from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader

from multimodal.multimodal_data_module import MultiModalDataset, \
    multiModalDataset_collate_fn, MultiModalDataModule, \
    PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, \
    PAD_TOKEN_ID, UNK_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID, \
    IMAGE_H, IMAGE_W

# directories and filenames
DATA_DIR = Path("/misc/vlgscratch5/LakeGroup/shared_data/coco")
ANNOTATIONS_DATA_DIR = DATA_DIR / "annotations"
RAW_TRAIN_DATA_FILENAME = ANNOTATIONS_DATA_DIR / "captions_train2017.json"
RAW_VAL_DATA_FILENAME = ANNOTATIONS_DATA_DIR / "captions_val2017.json"
KARPATHY_CAPTION_DATASETS_DIR = DATA_DIR / "karpathy_caption_datasets"
KARPATHY_CAPTION_DATASET_FILENAME = KARPATHY_CAPTION_DATASETS_DIR / "dataset_coco.json"
VOCAB_FILENAME = KARPATHY_CAPTION_DATASETS_DIR / "vocab.json"
TRAIN_IMAGE_DIR = DATA_DIR / "train2017"
VAL_IMAGE_DIR = DATA_DIR / "val2017"
TRAIN_DATA_FILENAME = KARPATHY_CAPTION_DATASETS_DIR / "preprocessed_captions_train2017.json"
VAL_DATA_FILENAME = KARPATHY_CAPTION_DATASETS_DIR / "preprocessed_captions_val2017.json"


def load_dataset(filename):
    with open(filename, 'r') as f:
        dataset = json.load(f)
    return dataset


class COCOCaptionsDataset(MultiModalDataset):
    """
    Dataset that returns paired image-captions from MS COCO Captions Dataset.
    """

    def __init__(self, dataset, image_dir, transform):
        super().__init__()
        self.dataset = dataset
        self.id2image = {image['id']: image for image in dataset['images']}
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.dataset['annotations'])

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        """
        Returns an image-caption pair in tuple (image, caption_idxs, caption_length)
        """

        annotation = self.dataset['annotations'][idx]

        # get caption
        caption_idxs = annotation['token_ids']
        caption_idxs = [SOS_TOKEN_ID] + caption_idxs + [EOS_TOKEN_ID]
        caption_length = len(caption_idxs)
        caption_idxs = torch.tensor(caption_idxs, dtype=torch.long)

        # get image
        image = self.id2image[annotation['image_id']]
        image_filename = self.image_dir / image['file_name']
        image = Image.open(image_filename).convert("RGB")

        # apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, caption_idxs, caption_length


class COCOCaptionsDataModule(MultiModalDataModule):
    """
    A data module for MS COCO Captions dataset consisting of images and the 
    associated captions.
    """

    def __init__(self, args=None) -> None:
        super().__init__(args)

        resizer = transforms.Resize((IMAGE_H, IMAGE_W))
        self.transform = transforms.Compose([
            resizer,
            self.transform,
        ])
        self.base_transform = transforms.Compose([
            resizer,
            self.base_transform,
        ])

    @staticmethod
    def add_additional_to_argparse(parser):
        return parser

    @staticmethod
    def add_to_argparse(parser):
        parser = super(COCOCaptionsDataModule, COCOCaptionsDataModule)\
            .add_to_argparse(parser)
        parser = COCOCaptionsDataModule.add_additional_to_argparse(parser)
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        super().prepare_data(*args, **kwargs)
        _prepare_data()

    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)

        self.datasets = {}

        for split, filename, image_dir, transform in [
                ("train", TRAIN_DATA_FILENAME, TRAIN_IMAGE_DIR, self.transform),
                ("val", VAL_DATA_FILENAME, VAL_IMAGE_DIR, self.base_transform)]:
            dataset = load_dataset(filename)
            dataset = COCOCaptionsDataset(
                dataset,
                image_dir=image_dir,
                transform=transform,
            )
            self.datasets[split] = dataset

    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'],
            collate_fn=multiModalDataset_collate_fn,
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets['val'],
            collate_fn=multiModalDataset_collate_fn,
            shuffle=False,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )


def _prepare_data(count_threshold=5):
    """
    count_threshold: build vocabulary for all tokens with count > count_threshold
    """

    # check if everything has been preprocessed
    if all(filename.exists() for filename in [
            VOCAB_FILENAME, TRAIN_DATA_FILENAME, VAL_DATA_FILENAME]):
        print("All data have already existed. Skipping this step.")
        return
    print("Preparing data...")

    with open(KARPATHY_CAPTION_DATASET_FILENAME, 'r') as f:
        karpathy_dataset = json.load(f)

    id2sent = {
        sentence['sentid']: sentence
        for image in karpathy_dataset['images']
        for sentence in image['sentences']
    }

    # following https://github.com/ruotianluo/self-critical.pytorch/blob/master/scripts/prepro_labels.py
    token_counts = Counter()
    length_counts = Counter()
    for sentence in id2sent.values():
        tokens = sentence['tokens']
        token_counts.update(tokens)
        length_counts[len(tokens)] += 1
    count_tokens = [(count, token) for token, count in token_counts.items()]
    count_tokens.sort(reverse=True)
    print('Top tokens and their counts:')
    for count, token in count_tokens[:20]:
        print(f'{token:>10}: {count:>7}')
    # print some stats
    total_tokens = sum(token_counts.values())
    print(f'Total tokens: {total_tokens}')
    vocab = [token for count, token in count_tokens if count > count_threshold]
    unk_count = sum((count for count, token in count_tokens if count <= count_threshold))
    print(f'vocab size (excluding special tokens): {len(vocab)}')
    frac_str = lambda a, b: f'{a}/{b} = {a/b:.2%}'
    print('OOV tokens: ' + frac_str(len(count_tokens) - len(vocab), len(count_tokens)))
    print('UNK rate: ' + frac_str(unk_count, total_tokens))
    max_length = max(length_counts.keys())
    print(f'Max length: {max_length}')
    print(f'Length distribution: (count: number of sequences)')
    for length in range(max_length + 1):
        count = length_counts.get(length, 0)
        print(f'{length:>2}: {count:>7} {count/len(id2sent):7.2%}')

    # add special tokens to vocab
    vocab = [None] * 4 + vocab
    for token, token_id in [
            (PAD_TOKEN, PAD_TOKEN_ID),
            (UNK_TOKEN, UNK_TOKEN_ID),
            (SOS_TOKEN, SOS_TOKEN_ID),
            (EOS_TOKEN, EOS_TOKEN_ID)]:
        vocab[token_id] = token

    idx2token = {idx: token for idx, token in enumerate(vocab)}
    token2idx = {token: idx for idx, token in enumerate(vocab)}

    with open(VOCAB_FILENAME, 'w') as f:
        json.dump(idx2token, f)
    VOCAB_FILENAME.chmod(0o644)

    # lookup tokens
    for sentence in id2sent.values():
        sentence['token_ids'] = [
            token2idx.get(token, UNK_TOKEN_ID) for token in sentence['tokens']]

    # preprocess data splits
    for split, filename, raw_filename, image_dir in [
            ('train', TRAIN_DATA_FILENAME, RAW_TRAIN_DATA_FILENAME, TRAIN_IMAGE_DIR),
            ('val', VAL_DATA_FILENAME, RAW_VAL_DATA_FILENAME, VAL_IMAGE_DIR)]:
        with open(raw_filename, 'r') as f:
            dataset = json.load(f)

        id2image = {image['id']: image for image in dataset['images']}

        for annotation in dataset['annotations']:
            # sanity checks
            assert annotation['image_id'] in id2image, \
                f"annotation id={annotation['id']} with image id={annotation['image_id']} not found in images"
            assert annotation['id'] in id2sent, \
                f"annotation id={annotation['id']} not found in Karpathy dataset"
            sentence = id2sent[annotation['id']]
            assert sentence['raw'] == annotation['caption'], \
                f"id={annotation['id']}, raw sentence in Karpathy dataset \"{sentence['raw']}\" differs from caption in annotation \"{annotation['caption']}\""
            # use the tokenized tokens and token_ids
            annotation['tokens'] = sentence['tokens']
            annotation['token_ids'] = sentence['token_ids']

        with open(filename, 'w') as f:
            json.dump(dataset, f)
        filename.chmod(0o644)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    COCOCaptionsDataModule.add_to_argparse(parser)
    args = parser.parse_args()

    # set up data module
    _prepare_data()
    data = COCOCaptionsDataModule(args)
