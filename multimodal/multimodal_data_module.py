from pathlib import Path
from typing import Any, Callable, Collection, Dict, Optional, Tuple, Union
import os
import glob
import json
import random
import re
import shutil
import time
import argparse
import cv2 as cv

import imageio
from PIL import Image
import numpy as np
import pandas as pd
from gsheets import Sheets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# from multimodal_saycam.base_data_module import BaseDataModule, load_and_print_info
from multimodal.utils import *

# directories and filenames
DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal")
GSHEETS_CREDENTIALS_FILENAME = DATA_DIR / "credentials.json"
TRANSCRIPT_LINKS_FILENAME = DATA_DIR / "SAYCam_transcript_links.csv"
TRANSCRIPTS_DIRNAME = DATA_DIR / "transcripts"
PREPROCESSED_TRANSCRIPTS_DIRNAME = DATA_DIR / "preprocessed_transcripts_5fps"
RAW_VIDEO_DIRNAME = "/misc/vlgscratch4/LakeGroup/shared_data/S_videos_annotations/S_videos/"
LABELED_S_DIRNAME = "/misc/vlgscratch4/LakeGroup/shared_data/S_clean_labeled_data_1fps_5"
EXTRACTED_FRAMES_DIRNAME = DATA_DIR / "train_5fps"
EVAL_FRAMES_DIRNAME = DATA_DIR / "eval"
ANIMATED_FRAMES_DIRNAME = DATA_DIR / "train_animated_5fps"
TRAIN_METADATA_FILENAME = DATA_DIR / "train.json"
VAL_METADATA_FILENAME = DATA_DIR / "val.json"
TEST_METADATA_FILENAME = DATA_DIR / "test.json"
EVAL_DEV_METADATA_FILENAME = DATA_DIR / "eval_dev.json"
EVAL_TEST_METADATA_FILENAME = DATA_DIR / "eval_test.json"
VOCAB_FILENAME = DATA_DIR / "vocab.json"

# default arguments
# dataloader arguments
BATCH_SIZE = 4
NUM_WORKERS = 4
TRAIN_FRAC = 0.9
VAL_FRAC = 0.05

# sampling arguments
MAX_FRAMES_PER_UTTERANCE = 32
MAX_LEN_UTTERANCE = 25

# training arguments
AUGMENT_FRAMES = False
MULTIPLE_FRAMES = False

# special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
SOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3

def read_vocab(vocab_filename=VOCAB_FILENAME):
    with open(vocab_filename) as f:
        return json.load(f)

class MultiModalSAYCamDataset(Dataset):
    """
    Dataset that returns paired image-utterances from baby S of the SAYCam Dataset
    """
    
    def __init__(self, data, vocab, multiple_frames, transform):
        super().__init__()
        self.data = data
        self.vocab = vocab
        self.multiple_frames = multiple_frames
        self.transform = transform

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        """
        Returns an image-utterance pair
        """

        # get utterance and convert to indices
        utterance = self.data[idx]["utterance"]
        utterance_words = utterance.split(" ")
        #utterance_words = utterance_words + [EOS_TOKEN]
        utterance_length = len(utterance_words)
        utterance_idxs = torch.tensor([self.vocab.get(word, UNK_TOKEN_ID) for word in utterance_words], dtype=torch.int)

        # get image
        img_filenames = self.data[idx]["frame_filenames"]

        if self.multiple_frames:
            # sample a random image associated with this utterance
            img_filename = Path(EXTRACTED_FRAMES_DIRNAME, random.choice(img_filenames))
        else:
            # otherwise, sample the first frame
            img_filename = Path(EXTRACTED_FRAMES_DIRNAME, img_filenames[0])

        img = Image.open(img_filename).convert("RGB")

        # apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, utterance_idxs, utterance_length

def multiModalSAYCamDataset_collate_fn(batch):
    img, utterance_idxs, utterance_length = zip(*batch)
    img = torch.stack(img, 0)
    utterance_idxs = pad_sequence(utterance_idxs, batch_first=True, padding_value=PAD_TOKEN_ID)
    utterance_length = torch.tensor(utterance_length, dtype=torch.int)
    if utterance_idxs.size(1) > MAX_LEN_UTTERANCE:
        utterance_idxs = utterance_idxs[:, :MAX_LEN_UTTERANCE]
        utterance_length = torch.minimum(utterance_length, torch.tensor(MAX_LEN_UTTERANCE, dtype=torch.int))
    return img, utterance_idxs, utterance_length

    
class LabeledSEvalDataset(Dataset):
    """
    Dataset that returns a set of referents and a target word for evaluation
    """
    
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        # read trial information
        trial = self.data[idx]

        # read in images (target and foils)
        # target image is always the first index
        imgs = torch.zeros((4, 3, 224, 224))
        target_img_filename = trial["target_img_filename"]
        imgs[0] = self.transform(Image.open(target_img_filename).convert("RGB"))

        for i, foil_img_filename in enumerate(trial["foil_img_filenames"]):
            imgs[i+1] = self.transform(Image.open(foil_img_filename).convert("RGB"))

        # get target category index from vocab as a single utterance
        label = self.vocab[trial["target_category"]]
        label = torch.LongTensor([label])
        label_len = len(label)
            
        return imgs, label, label_len

    def __len__(self):
        return len(self.data)

    
class MultiModalSAYCamDataModule(pl.LightningDataModule):
    """
    The MultiModal SAYCam Dataset is a dataset created from baby S of the SAYCam Dataset consisting of
    image frames and the associated child-directed utterances.
    """

    def __init__(self, args=None) -> None:
        super().__init__(args)

        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)
        self.multiple_frames = self.args.get("multiple_frames", MULTIPLE_FRAMES)
        self.augment_frames = self.args.get("augment_frames", AUGMENT_FRAMES)
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))        

        if self.augment_frames:
            # add same augmentations as emin used
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                # transforms.RandomApply([transforms.ColorJitter(0.9, 0.9, 0.9, 0.5)], p=0.9),
                # transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),            
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            # just convert to tensor and normalize
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # keep base transform for val and test
        self.base_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="Number of additional processes to load data."
        )
        parser.add_argument(
            "--multiple_frames", action="store_true", help="Randomly sample frames per utterance."
        )
        parser.add_argument(
            "--augment_frames", action="store_true", help="Apply data augmentation to images."
        )
        return parser

    # TODO: add relevant config details
    # def config(self):
    #     """Return important settings of the dataset, which will be passed to instantiate models."""
    #     return {"input_dims": self.dims, "output_dims": self.output_dims, "mapping": self.mapping}
            
    def prepare_data(self, *args, **kwargs) -> None:
        print("Calling prepare_data!")
        _download_transcripts()
        _rename_transcripts()
        _preprocess_transcripts()
        _extract_train_frames()
        _create_train_metadata()
        _extract_eval_frames()
        _create_eval_metadata()
        _create_vocab()
        # _create_animations()  # TODO: add extra argument to generate this?
    
    def setup(self, *args, **kwargs) -> None:
        print("Calling setup!")
        # read image-text data splits
        with open(TRAIN_METADATA_FILENAME) as f:
            train_data = json.load(f)
            train_data = train_data["data"]

        with open(VAL_METADATA_FILENAME) as f:
            val_data = json.load(f)
            val_data = val_data["data"]

        with open(TEST_METADATA_FILENAME) as f:
            test_data = json.load(f)
            test_data = test_data["data"]

        # read eval data splits
        with open(EVAL_DEV_METADATA_FILENAME) as f:
            eval_dev_data = json.load(f)
            eval_dev_data = eval_dev_data["data"]

        with open(EVAL_TEST_METADATA_FILENAME) as f:
            eval_test_data = json.load(f)
            eval_test_data = eval_test_data["data"]
            
        # read vocab
        vocab = read_vocab()

        # create image-text datasets
        self.train_dataset = MultiModalSAYCamDataset(train_data, vocab,
                                                     multiple_frames=self.multiple_frames,
                                                     transform=self.transform)
        self.val_dataset = MultiModalSAYCamDataset(val_data, vocab,
                                                   multiple_frames=False,
                                                   transform=self.base_transform)
        self.test_dataset = MultiModalSAYCamDataset(test_data, vocab,
                                                    multiple_frames=False,
                                                    transform=self.base_transform)

        # create eval datasets
        self.eval_dev_dataset = LabeledSEvalDataset(eval_dev_data, vocab)
        self.eval_test_dataset = LabeledSEvalDataset(eval_test_data, vocab)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=multiModalSAYCamDataset_collate_fn,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )
        
    def val_dataloader(self):
        contrastive_val_dataloader = DataLoader(
            self.val_dataset,
            collate_fn=multiModalSAYCamDataset_collate_fn,
            shuffle=False,
            # batch_size=self.batch_size,
            batch_size=64,  # fixing this so that validation sets are equal across runs
            num_workers=self.num_workers,
            pin_memory=False,
        )

        eval_dev_dataloader = DataLoader(
            self.eval_dev_dataset,
            collate_fn=multiModalSAYCamDataset_collate_fn,
            shuffle=False,
            # batch_size=self.batch_size // 4,  # divide by 4 here since eval trials have 4 images
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=False
        )

        return [contrastive_val_dataloader,
                eval_dev_dataloader]
        
def _download_transcripts():
    """Download SAYCam transcripts."""
    
    # check if transcripts have already been downloaded
    if os.path.exists(TRANSCRIPTS_DIRNAME):
        print("SAYCam transcripts have already been downloaded. Skipping this step.")
    else:
        print("Downloading SAYCam transcripts from Google Sheets")
     
        # create transcript folder
        if not os.path.exists(TRANSCRIPTS_DIRNAME):
            os.makedirs(TRANSCRIPTS_DIRNAME)
            
        # set up google sheets object
        sheets = Sheets.from_files(GSHEETS_CREDENTIALS_FILENAME)
            
        # get urls of saycam files to download
        df = pd.read_csv(TRANSCRIPT_LINKS_FILENAME)
        urls = df["GoogleSheets Link"].unique()
            
        for i, url in enumerate(urls):
            print(f"Downloading SAYCam transcript {i+1}/{len(urls)}: {url}")
            s = sheets.get(url)
            title = s.title.split("_")
            title = "_".join(title[:3])

            # read all sheets (skipping the first one since it is blank)
            for j in range(1, len(s.sheets)):
                try:
                    # try and parse this sheet as a data frame
                    df = s.sheets[j].to_frame()  # convert worksheet to data frame
                    filename = f"{TRANSCRIPTS_DIRNAME}/{title}_{s.sheets[j].title}.csv"  # get filename of dataframe
                    df.to_csv(filename, index=False)  # save as CSV
                except pd.errors.ParserError:
                    continue  # move onto the next file
                    
            # sleep for 30 seconds to prevent rate limiting
            time.sleep(30)

def _rename_transcripts():
    """Manually rename a few of the transcripts that don't match naming scheme."""
    
    if os.path.exists(TRANSCRIPTS_DIRNAME / "S_20141029_2412_part 2.csv"):
        print("Renaming transcripts")
        os.rename(TRANSCRIPTS_DIRNAME / "S_20141029_2412_part 2.csv",
                  TRANSCRIPTS_DIRNAME / "S_20141029_2412_02.csv")
        os.rename(TRANSCRIPTS_DIRNAME / "S_20141029_2412_part 3.csv",
                  TRANSCRIPTS_DIRNAME / "S_20141029_2412_03.csv")
        os.rename(TRANSCRIPTS_DIRNAME / "S_20141029_2412_part 4.csv",
                  TRANSCRIPTS_DIRNAME / "S_20141029_2412_04.csv")
        os.rename(TRANSCRIPTS_DIRNAME / "S_20141029_2412_part 5.csv",
                  TRANSCRIPTS_DIRNAME / "S_20141029_2412_05.csv")
        os.rename(TRANSCRIPTS_DIRNAME / "S_20141029_2412_part 6.csv",
                  TRANSCRIPTS_DIRNAME / "S_20141029_2412_06.csv")

    if os.path.exists(TRANSCRIPTS_DIRNAME / "S_20141122_2505_part 1.csv"):
        print("Renaming transcripts")
        os.rename(TRANSCRIPTS_DIRNAME / "S_20141122_2505_part 1.csv",
                  TRANSCRIPTS_DIRNAME / "S_20141122_2505_01.csv")
        os.rename(TRANSCRIPTS_DIRNAME / "S_20141122_2505_part 2.csv",
                  TRANSCRIPTS_DIRNAME / "S_20141122_2505_02.csv")
    else:
        print("Transcripts have already been renamed. Skipping this step.")
            
def _preprocess_transcripts():
    """Preprocess transcripts by cleaning the text and extracting frame timings."""

    # check if transcripts have already been downloaded
    if os.path.exists(PREPROCESSED_TRANSCRIPTS_DIRNAME):
        print("Transcripts have already been preprocessed. Skipping this step.")
    else:
        print("Preprocessing transcripts")

        # create preprocessed transcripts folder
        if not os.path.exists(PREPROCESSED_TRANSCRIPTS_DIRNAME):
            os.makedirs(PREPROCESSED_TRANSCRIPTS_DIRNAME)

        # get all transcripts and allowed speakers
        transcripts = sorted(Path(TRANSCRIPTS_DIRNAME).glob("*.csv"))
        allowed_speakers = ["M", "Mom", "mom", "m", "mother", "Mother", "papa", "the mom"]

        # preprocess each transcript
        for transcript_idx, transcript_filename in enumerate(transcripts):
            # empty list to store processed transcript information
            preprocessed_transcript = []
            preprocessed_transcript_filename = PREPROCESSED_TRANSCRIPTS_DIRNAME / transcript_filename.name
     
            # read transcript CSV
            print(f"Preprocessing transcript: {transcript_filename.name} ({transcript_idx+1}/{len(transcripts)})")
            transcript = pd.read_csv(transcript_filename)
     
            # skip empty transcripts
            if len(transcript) <= 1:
                continue
            
            # create new column of timestamps converted to seconds
            new_timestamps = convert_timestamps_to_seconds(transcript["Time"])
            transcript["Time (Seconds)"] = new_timestamps
     
            # reset utterance count
            utterance_num = 1
     
            # extract unique video filename from transcript
            video_filename = pd.unique(transcript["Video Name"])
     
            # drop any missing filenames, or any filenames with "part" in them
            video_filename = [x for x in video_filename if not pd.isnull(x)]
            video_filename = [x for x in video_filename if "part" not in x]
     
            # skip if video filename is not unique
            if len(video_filename) != 1:
                continue
     
            # extract video filename and replace suffix
            video_filename = video_filename[0]
            video_filename = Path(video_filename).with_suffix(".mp4")
     
            # check video and transcript filenames match
            assert video_filename.stem == transcript_filename.stem
     
            for transcript_row_idx, row in transcript.iterrows():
                # get information from current utterance
                utterance = str(row["Utterance"])  # convert to string
                speaker = str(row["Speaker"])
                start_timestamp = row["Time (Seconds)"]
     
                # get end timestamp
                # hack: if last timestamp, just set end timestamp to be start time
                # this means we don't have to read the video file for this to work
                if transcript_row_idx < len(transcript) - 1:
                    end_timestamp = transcript["Time (Seconds)"][transcript_row_idx+1]
                else:
                    end_timestamp = start_timestamp  # this will sample a single frame for the last utterance
     
                # skip processing utterance if start or end timestamps are null,
                # or if speaker is not in the list of allowed speakers
                if pd.isnull(start_timestamp) or pd.isnull(end_timestamp) or speaker not in allowed_speakers:
                    continue
     
                # preprocess utterance to extract sub-utterances and timestamps
                utterances, timestamps, num_frames = _preprocess_utterance(
                    utterance, start_timestamp, end_timestamp)
     
                # skip if preprocessed utterance is empty
                if len(utterances) == 0:
                    continue
     
                # create dataset based on preprocessed utterances
                for (curr_utterance, curr_timestamps, curr_num_frames) in zip(utterances, timestamps, num_frames):
                    # loop over all possible frames for the current utterance
                    for frame_num, curr_timestamp in enumerate(curr_timestamps):
                        frame_filename = f"{video_filename.stem}_{utterance_num:03}_{frame_num:02}.jpg"
                        preprocessed_transcript.append([transcript_filename.name,
                            video_filename.name, curr_utterance, curr_timestamp,
                            utterance_num, frame_num, frame_filename])
     
                    utterance_num += 1
     
            # save preprocessed transcript as CSV
            if len(preprocessed_transcript) > 0:
                preprocessed_transcript_columns = ["transcript_filename", "video_filename",
                    "utterance", "timestamp", "utterance_num", "frame_num", "frame_filename"]
                preprocessed_transcript_df = pd.DataFrame(preprocessed_transcript,
                                                          columns=preprocessed_transcript_columns)
                preprocessed_transcript_df.to_csv(preprocessed_transcript_filename, index=False)


def _preprocess_utterance(utterance, start_timestamp, end_timestamp):
    """Preprocesses a single utterance, splitting it into multiple clean utterances with separate timestamps"""

    # check start timestamp is before end timestamp
    assert start_timestamp <= end_timestamp

    # remove special characters, anything in asterisks or parentheses etc.
    utterance = re.sub(r"\*[^)]*\*", "", utterance)
    utterance = re.sub(r"\[[^)]*\]", "", utterance)
    utterance = re.sub(r"\([^)]*\)", "", utterance)
    utterance = re.sub(r" +", " ", utterance)
    utterance = utterance.replace("--", " ")
    utterance = utterance.replace("-", "")
    utterance = utterance.replace("'", "")
    utterance = utterance.replace("*", "")
    utterance = utterance.replace("_", "")
    utterance = utterance.replace(",", "")
    utterance = utterance.replace("…", "")
    utterance = utterance.lower().strip()

    # split utterance based on certain delimeters, strip and remove empty utterances
    utterances = msplit(utterance, (".", "?", "!"))
    utterances = [utterance.strip() for utterance in utterances if len(utterance) > 0]

    if len(utterances) > 0:
        # get interpolated timestamps, including end timestamp (which we remove later)
        timestamps = np.linspace(start_timestamp, end_timestamp, len(utterances)+1, endpoint=True)
        timestamps = [int(timestamp) for timestamp in timestamps]
        all_timestamps = []
        num_frames = []

        # calculate number of frames to extract per utterance (max: 32 frames at 5fps)
        for i in range(len(timestamps)-1):
            curr_num_frames = max(min(int((timestamps[i+1] - timestamps[i]) / 0.2), MAX_FRAMES_PER_UTTERANCE), 1)
            curr_timestamps = np.linspace(timestamps[i], timestamps[i] + (curr_num_frames / 5),
                                          curr_num_frames, endpoint=False)
            # check same length
            assert len(curr_timestamps) == curr_num_frames

            # append information
            num_frames.append(curr_num_frames)
            all_timestamps.append(curr_timestamps)

        timestamps = timestamps[:-1]  # remove end timestamp
    else:
        all_timestamps = []
        num_frames = []

    # check everything is the same length
    assert len(utterances) == len(all_timestamps)
    assert len(all_timestamps) == len(num_frames)

    return utterances, all_timestamps, num_frames
            
def _extract_train_frames():
    """Extract aligned frames from SAYCam videos"""

    if os.path.exists(EXTRACTED_FRAMES_DIRNAME):
        print("Training frames have already been extracted. Skipping this step.")
    else:
        print("Extracting training frames")

        # create directory to store extracted frames
        if not os.path.exists(EXTRACTED_FRAMES_DIRNAME):
            os.makedirs(EXTRACTED_FRAMES_DIRNAME)

        # get all preprocessed transcripts
        transcripts = sorted(Path(PREPROCESSED_TRANSCRIPTS_DIRNAME).glob("*.csv"))

        for idx, transcript in enumerate(transcripts):
            # get video filename associated with this transcript
            transcript_df = pd.read_csv(transcript)
            video_filename = Path(RAW_VIDEO_DIRNAME, pd.unique(transcript_df["video_filename"]).item())

            # skip if video doesn"t exist
            if not video_filename.exists():
                print(f"{video_filename} missing! Skipping")
                continue

            # otherwise continue extraction process
            print(f"Extracting frames: {video_filename.name} ({idx+1}/{len(transcripts)})")

            # read in video and get information
            cap = cv.VideoCapture(str(video_filename))
            video_info = _get_video_info(cap)
            frame_count, frame_width, frame_height, frame_rate, frame_length = video_info

            for transcript_row_idx, row in transcript_df.iterrows():
                # get information for frame extraction
                frame_filename = Path(EXTRACTED_FRAMES_DIRNAME, str(row["frame_filename"]))
                timestamp = float(row["timestamp"])  # keep as float
                framestamp = int(timestamp * frame_rate)

                # extract frame based on timestamp
                cap.set(1, framestamp)  # set frame to extract from 
                ret, frame = cap.read()  # read frame
                frame = _extract_frame(frame, frame_height, frame_width)

                # save frame
                if frame is not None:
                    cv.imwrite(str(frame_filename), frame)

def _get_video_info(cap):
    """Returns video information"""
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv.CAP_PROP_FPS)  # leave this as a float
    frame_length = frame_count // frame_rate
    return frame_count, frame_width, frame_height, frame_rate, frame_length
    
    
def _extract_frame(frame, frame_height, frame_width):
    """Extract a single frame"""
    
    # settings for frame extraction
    final_size = 224
    resized_minor_length = 256
    new_height = frame_height * resized_minor_length // min(frame_height, frame_width)
    new_width = frame_width * resized_minor_length // min(frame_height, frame_width)
    
    # function to resize frame and recolor
    try:
        resized_frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_CUBIC)
    except Exception as e:
        print(str(e))
        return None

    # crop
    height, width, _ = resized_frame.shape
    startx = width // 2 - (final_size // 2)
    starty = height // 2 - (final_size // 2) - 16
    cropped_frame = resized_frame[starty:starty + final_size, startx:startx + final_size]
    assert cropped_frame.shape[0] == final_size and cropped_frame.shape[1] == final_size, \
        (cropped_frame.shape, height, width)

    # reverse x/y axes
    cropped_frame = np.array(cropped_frame)
    cropped_frame = cropped_frame[::-1, ::-1, :]
    # cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    return cropped_frame


def _extract_eval_frames():
    """Extract evaluation frames from labeled S dataset, splitting evenly for dev and test"""

    if os.path.exists(EVAL_FRAMES_DIRNAME):
        print("Evaluation frames have already been extracted. Skipping this step.")
    else:
        print("Extracting evaluation frames")

        # create directory to store evaluation frames
        if not os.path.exists(EVAL_FRAMES_DIRNAME):
            os.makedirs(EVAL_FRAMES_DIRNAME)
            os.makedirs(EVAL_FRAMES_DIRNAME / "dev")
            os.makedirs(EVAL_FRAMES_DIRNAME / "test")
     
        # get original set of evaluation categories
        eval_categories = os.listdir(LABELED_S_DIRNAME)
        for eval_category in eval_categories:
            eval_category_dirname = os.path.join(LABELED_S_DIRNAME, eval_category)
            eval_category_frames = sorted(os.listdir(eval_category_dirname))
     
            # get indices to split original labeled s dataset into dev and test
            split_idxs = np.arange(len(eval_category_frames))
            np.random.shuffle(split_idxs)
            dev_idxs = split_idxs[:int(len(eval_category_frames) * 0.5)]
            test_idxs = split_idxs[int(len(eval_category_frames) * 0.5):]
            
            # check dataset has been split correct
            assert len(dev_idxs) + len(test_idxs) == len(split_idxs)
     
            # copy over dev frames into a new directory
            print(f"copying {eval_category} frames for dev set")
     
            # check if directory exists, and if not, create it
            if not os.path.exists(os.path.join(EVAL_FRAMES_DIRNAME, "dev", eval_category)):
                os.makedirs(os.path.join(EVAL_FRAMES_DIRNAME, "dev", eval_category))
     
            for dev_idx in dev_idxs:
                # get path to original frame
                original_filename = os.path.join(LABELED_S_DIRNAME, eval_category, eval_category_frames[dev_idx])
     
                # copy frame
                shutil.copyfile(original_filename, os.path.join(EVAL_FRAMES_DIRNAME, "dev", eval_category, eval_category_frames[dev_idx]))
     
            # copy over test frames into a new directory
            print(f"copying {eval_category} frames for test set")
     
            # check if directory exists, and if not, create it
            if not os.path.exists(os.path.join(EVAL_FRAMES_DIRNAME, "test", eval_category)):
                os.makedirs(os.path.join(EVAL_FRAMES_DIRNAME, "test", eval_category))
     
            for test_idx in test_idxs:
                # get path to original frame
                original_filename = os.path.join(LABELED_S_DIRNAME, eval_category, eval_category_frames[test_idx])
     
                # copy frame
                shutil.copyfile(original_filename, os.path.join(EVAL_FRAMES_DIRNAME, "test", eval_category, eval_category_frames[test_idx]))
    
    
def _create_train_metadata():
    """Creates JSON files with image-utterance information"""
    
    if os.path.exists(TRAIN_METADATA_FILENAME) and os.path.exists(VAL_METADATA_FILENAME) and os.path.exists(TEST_METADATA_FILENAME):
        print("Training metadata files have already been created . Skipping this step.")
    else:
        print("Creating metadata files for train, validation and test.")

        # get all preprocessed transcripts
        transcripts = sorted(Path(PREPROCESSED_TRANSCRIPTS_DIRNAME).glob("*.csv"))

        utterances = []

        for idx, transcript in enumerate(transcripts):            
            # read in preprocessed transcript
            transcript_df = pd.read_csv(transcript)
            
            # group by utterances
            utterance_groups = transcript_df.groupby("utterance_num")
            for utterance, utterance_group in utterance_groups:
                # extract relevant information
                curr_utterance = {}
                curr_utterance["utterance"] = pd.unique(utterance_group["utterance"]).item()
                curr_utterance["transcript_filename"] = pd.unique(utterance_group["transcript_filename"]).item()
                curr_utterance["video_filename"] = pd.unique(utterance_group["video_filename"]).item()
                curr_utterance["utterance_num"] = pd.unique(utterance_group["utterance_num"]).item()
                curr_utterance["num_frames"] = len(utterance_group)
                curr_utterance["timestamps"] = list(utterance_group["timestamp"])

                # extract filenames separately
                curr_utterance["frame_filenames"] = []  # initialize as empty list
                curr_utterance_filenames = sorted(list(utterance_group["frame_filename"]))

                # skip over any nan utterances
                if not isinstance(curr_utterance["utterance"], str):
                    continue

                # check frame filenames and append all frames that exist
                for frame_filename in curr_utterance_filenames:
                    if (EXTRACTED_FRAMES_DIRNAME / frame_filename).exists():
                        curr_utterance["frame_filenames"].append(frame_filename)
                    else:
                        print(f"{frame_filename} does not exist, removing it from this list")

                # skip utterance completely if no frames were extracted
                if len(curr_utterance["frame_filenames"]) == 0:
                    print("No corresponding frames found, skipping this utterance")
                    continue
                
                # append details of remaining utterances to metadata list
                utterances.append(curr_utterance)

        # shuffle utterances
        random.shuffle(utterances)
                
        # split utterances into train/val/test
        train_n = int(len(utterances) * TRAIN_FRAC)
        val_n = int(len(utterances) * VAL_FRAC)
        test_n = int(len(utterances) - train_n - val_n)
        idxs = np.arange(len(utterances))
        train_idxs = idxs[:train_n]
        val_idxs = idxs[train_n:train_n+val_n]
        test_idxs = idxs[train_n+val_n:]
        train_utterances = [utterances[i] for i in train_idxs]
        val_utterances = [utterances[i] for i in val_idxs]
        test_utterances = [utterances[i] for i in test_idxs]
                
        # put utterances into a dictionary
        train_dict = {"data": train_utterances}
        val_dict = {"data": val_utterances}
        test_dict = {"data": test_utterances}

        # save as JSON files
        with open(TRAIN_METADATA_FILENAME, "w") as f:
            json.dump(train_dict, f)

        with open(VAL_METADATA_FILENAME, "w") as f:
            json.dump(val_dict, f)

        with open(TEST_METADATA_FILENAME, "w") as f:
            json.dump(test_dict, f)

def _create_eval_metadata():
    """Creates files for evaluating multimodal SAYCam model"""

    if os.path.exists(EVAL_DEV_METADATA_FILENAME) and os.path.exists(EVAL_TEST_METADATA_FILENAME):
        print("Evaluation metadata files have already been created . Skipping this step.")
    else:
        print("Creating metadata files for evaluation.")

        n_foils = 3  # number of foil referents
        n_evaluations = 100  # number of evaluations per category
        eval_dev_dataset = []
        eval_test_dataset = []

        # get evaluation categories and remove ones not in vocab
        eval_dev_dirname = EVAL_FRAMES_DIRNAME / "dev"
        eval_test_dirname = EVAL_FRAMES_DIRNAME / "test"
        eval_categories = sorted(os.listdir(eval_dev_dirname)) 
        eval_categories.remove("carseat")
        eval_categories.remove("couch")
        eval_categories.remove("greenery")
        eval_categories.remove("plushanimal")

        # generate dev evaluation trials
        for target_category in eval_categories:
            for i in range(n_evaluations):
                # sample item from target category
                target_category_dirname = os.path.join(eval_dev_dirname, target_category)
                target_img_filename = os.path.join(target_category_dirname,
                                                   np.random.choice(os.listdir(target_category_dirname)))
     
                foil_categories = eval_categories.copy()
                foil_categories.remove(target_category)
                foil_categories = np.random.choice(foil_categories, size=n_foils, replace=False)
                foil_img_filenames = []
     
                for j in range(n_foils):
                    foil_category_dirname = os.path.join(eval_dev_dirname, foil_categories[j])
                    foil_img_filename = os.path.join(foil_category_dirname,
                                                 np.random.choice(os.listdir(foil_category_dirname)))
                    foil_img_filenames.append(foil_img_filename)

                # save trial info as a dict
                eval_trial = {}
                eval_trial["trial_num"] = i
                eval_trial["target_category"] = target_category
                eval_trial["target_img_filename"] = target_img_filename
                eval_trial["foil_categories"] = list(foil_categories)
                eval_trial["foil_img_filenames"] = foil_img_filenames
                eval_dev_dataset.append(eval_trial)

        # generate test evaluation trials
        for target_category in eval_categories:
            for i in range(n_evaluations):
                # sample item from target category
                target_category_dirname = os.path.join(eval_test_dirname, target_category)
                target_img_filename = os.path.join(target_category_dirname,
                                                   np.random.choice(os.listdir(target_category_dirname)))
     
                foil_categories = eval_categories.copy()
                foil_categories.remove(target_category)
                foil_categories = np.random.choice(foil_categories, size=n_foils, replace=False)
                foil_img_filenames = []
     
                for j in range(n_foils):
                    foil_category_dirname = os.path.join(eval_test_dirname, foil_categories[j])
                    foil_img_filename = os.path.join(foil_category_dirname,
                                                 np.random.choice(os.listdir(foil_category_dirname)))
                    foil_img_filenames.append(foil_img_filename)

                # save trial info as a dict
                eval_trial = {}
                eval_trial["trial_num"] = i
                eval_trial["target_category"] = target_category
                eval_trial["target_img_filename"] = target_img_filename
                eval_trial["foil_categories"] = list(foil_categories)
                eval_trial["foil_img_filenames"] = foil_img_filenames
                eval_test_dataset.append(eval_trial)
                    
        # put eval trials into dictionaries
        eval_dev_dict = {"data": eval_dev_dataset}
        eval_test_dict = {"data": eval_test_dataset}

        # save as JSON files
        with open(EVAL_DEV_METADATA_FILENAME, "w") as f:
            json.dump(eval_dev_dict, f)

        with open(EVAL_TEST_METADATA_FILENAME, "w") as f:
            json.dump(eval_test_dict, f)
        
            
def _create_vocab():
    """Create vocabulary object and save to file"""

    if VOCAB_FILENAME.exists():
        print("Vocabulary file already exists. Skipping this step.")
    else:
        print("Creating vocab.json file!")

        # create vocab dictionary
        vocab_dict = {PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}
        num_words = 4

        # load utterances from training set
        with open(TRAIN_METADATA_FILENAME) as f:
            train_dict = json.load(f)

        # fill vocab with all words from utterances
        train = train_dict["data"]
        for i in range(len(train)):
            curr_utterance = str(train[i]["utterance"])
            words = curr_utterance.split(" ")
            for word in words:
                if word not in vocab_dict:
                    vocab_dict[word] = num_words
                    num_words += 1

        # save as JSON file
        with open(VOCAB_FILENAME, "w") as f:
            json.dump(vocab_dict, f)
        

def _create_animations():
    """Create animated GIFs of extracted frames paired with each utterance"""

    if os.path.exists(ANIMATED_FRAMES_DIRNAME):
        print("Animated gifs have already been created. Skipping this step.")
    else:
        print("Creating animated gifs")
    
        # create directory to store extracted frames
        if not os.path.exists(ANIMATED_FRAMES_DIRNAME):
            os.makedirs(ANIMATED_FRAMES_DIRNAME)
     
        # get list of preprocessed transcripts
        transcripts = sorted(Path(PREPROCESSED_TRANSCRIPTS_DIRNAME).glob("*.csv"))[:5]
     
        for idx, transcript in enumerate(transcripts):
            print(f"Creating animated gifs: {transcript} ({idx+1}/{len(transcripts)})")
            
            # read in preprocessed transcript
            transcript_df = pd.read_csv(transcript)
            
            # group by utterances
            utterance_groups = transcript_df.groupby("utterance_num")
     
            # create gif
            for utterance, utterance_group in utterance_groups:
                utterance_num = pd.unique(utterance_group["utterance_num"]).item()
                gif_filename = f"{pd.unique(utterance_group['transcript_filename']).item()[:-4]}_{utterance_num:03}.gif"
                gif_filepath = Path(ANIMATED_FRAMES_DIRNAME, gif_filename)
                frame_filenames = utterance_group["frame_filename"]
     
                frames = []
                for frame_filename in frame_filenames:
                    frame_filepath = EXTRACTED_FRAMES_DIRNAME / frame_filename
     
                    try:
                        img = imageio.imread(frame_filepath)
                    except FileNotFoundError:
                        continue
                        
                    frames.append(img)
     
                if len(frames) > 0:
                    print(f"Saving {gif_filepath}, with {len(frames)} frames")
                    imageio.mimsave(gif_filepath, frames, fps=10)

            
if __name__ == "__main__":
    load_and_print_info(MultiModalSAYCamDataModule)
