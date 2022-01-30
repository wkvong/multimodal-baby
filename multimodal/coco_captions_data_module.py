from typing import Any, Tuple
from pathlib import Path
import json
import argparse
from collections import Counter
import random

from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader

from multimodal.multimodal_data_module import MultiModalDataset, \
    MultiModalDataModule, read_vocab, \
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
IMAGE_DIR = DATA_DIR / "all_images"
TRAIN_DATA_FILENAME = KARPATHY_CAPTION_DATASETS_DIR / "preprocessed_captions_train.json"
VAL_DATA_FILENAME = KARPATHY_CAPTION_DATASETS_DIR / "preprocessed_captions_val.json"
TEST_DATA_FILENAME = KARPATHY_CAPTION_DATASETS_DIR / "preprocessed_captions_test.json"

# default arguments
# training arguments
MULTIPLE_CAPTIONS = False


def load_dataset(filename):
    with open(filename, 'r') as f:
        dataset = json.load(f)
    return dataset


class COCOCaptionsDataset(MultiModalDataset):
    """
    Dataset that returns paired image-captions from MS COCO Captions Dataset.
    """

    def __init__(self, dataset, image_dir, multiple_captions, transform):
        super().__init__()
        self.dataset = dataset
        self.image_dir = image_dir
        self.multiple_captions = multiple_captions
        self.transform = transform

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.dataset['images'])

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Returns an image-caption pair in tuple
        (image, caption_idxs, caption_length, raw_captions)
        """

        image = self.dataset['images'][idx]

        # get caption
        captions = image['sentences']
        # not using 'raw' because that is untokenized and contains punctuations
        raw_captions = [" ".join(caption['tokens']) for caption in captions]
        caption = random.choice(captions) if self.multiple_captions else \
                  captions[0]
        caption_idxs = caption['token_ids']
        caption_idxs = [SOS_TOKEN_ID] + caption_idxs + [EOS_TOKEN_ID]
        caption_length = len(caption_idxs)
        caption_idxs = torch.tensor(caption_idxs, dtype=torch.long)

        # get image
        image_filename = self.image_dir / image['filename']
        image = Image.open(image_filename).convert("RGB")

        # apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, caption_idxs, caption_length, raw_captions


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

        self.multiple_captions = self.args.get(
            "multiple_captions", MULTIPLE_CAPTIONS)

    @staticmethod
    def add_additional_to_argparse(parser):
        parser.add_argument(
            "--multiple_captions", action="store_true",
            help="Randomly sample captions per image."
        )
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

    def read_vocab(self):
        return read_vocab(VOCAB_FILENAME)

    def create_datasets(self, vocab):
        datasets = {}

        for split, filename, multiple_captions, transform in [
                ("train", TRAIN_DATA_FILENAME, self.multiple_captions,
                 self.transform),
                ("val", VAL_DATA_FILENAME, False, self.base_transform),
                ("test", TEST_DATA_FILENAME, False, self.base_transform)]:
            dataset = load_dataset(filename)
            dataset = COCOCaptionsDataset(
                dataset,
                image_dir=IMAGE_DIR,
                multiple_captions=multiple_captions,
                transform=transform,
            )
            datasets[split] = dataset

        return datasets


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

    id2img = {
        img['imgid']: img
        for img in karpathy_dataset['images']
    }

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
        json.dump(token2idx, f)
    VOCAB_FILENAME.chmod(0o644)

    # lookup tokens
    for sentence in id2sent.values():
        sentence['token_ids'] = [
            token2idx.get(token, UNK_TOKEN_ID) for token in sentence['tokens']]

    # here we do not use the data splits provided by Emin, but use them for sanity checks
    split_mappings = set()
    for split, raw_filename in [
            ('train', RAW_TRAIN_DATA_FILENAME),
            ('val', RAW_VAL_DATA_FILENAME)]:
        with open(raw_filename, 'r') as f:
            dataset = json.load(f)

        id2image = {image['id']: image for image in dataset['images']}
        print(f"{split} 2017 images: {len(dataset['images'])}")

        for annotation in dataset['annotations']:
            # sanity checks
            image = id2image[annotation['image_id']]
            sentence = id2sent[annotation['id']]
            assert sentence['raw'] == annotation['caption'], \
                f"id={annotation['id']}, raw sentence in Karpathy dataset \"{sentence['raw']}\" differs from caption in annotation \"{annotation['caption']}\""
            img = id2img[sentence['imgid']]
            assert img['cocoid'] == image['id'], f"cocoid={img['cocoid']} different from image['id']={image['id']}"
            assert img['filename'][-16:] == image['file_name'], f"img filename={img['filename']} different from image file_name={image['file_name']}"
            split_mappings.add((img['split'], split))
    print('split mappings (karpathy, 2017):', split_mappings)

    # check all images exist
    for image in karpathy_dataset['images']:
        image['filename'] = image['filename'][-16:]
        image_path = DATA_DIR / "all_images" / image['filename']
        assert image_path.exists(), f"{image['filename']} does not exist"

    # save preprocessed splits
    for split, karpathy_splits, filename in [
            ('train', ('train', 'restval'), TRAIN_DATA_FILENAME),
            ('val', ('val',), VAL_DATA_FILENAME),
            ('test', ('test',), TEST_DATA_FILENAME)]:
        images = list(filter(lambda image: image['split'] in karpathy_splits,
                             karpathy_dataset['images']))
        print(f"{split} images: {len(images)}")
        dataset = {
            key: value
            for key, value in karpathy_dataset.items() if key != 'images'
        }
        dataset['images'] = images

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
