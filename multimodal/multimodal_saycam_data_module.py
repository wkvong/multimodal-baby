from pathlib import Path
from typing import Any, Tuple
from collections import Counter
import os
import glob
import itertools
import json
import random
import re
import shutil
import time
import cv2 as cv

import imageio
from PIL import Image
import numpy as np
import pandas as pd
from gsheets import Sheets
import torch

import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from multimodal.multimodal_data_module import MultiModalDataset, \
    MultiModalDataModule, read_vocab, load_data, load_and_print_info, \
    PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, \
    PAD_TOKEN_ID, UNK_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID, \
    IMAGE_H, IMAGE_W
from multimodal.utils import *

import spacy
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# directories and filenames
DATA_DIR = Path("/misc/vlgscratch4/LakeGroup/shared_data/S_multimodal")
GSHEETS_CREDENTIALS_FILENAME = DATA_DIR / "credentials.json"
TRANSCRIPT_LINKS_FILENAME = DATA_DIR / "SAYCam_transcript_links.csv"
TRANSCRIPTS_DIRNAME = DATA_DIR / "transcripts"
PREPROCESSED_TRANSCRIPTS_DIRNAME = DATA_DIR / "preprocessed_transcripts_5fps"
RAW_VIDEO_DIRNAME = "/misc/vlgscratch4/LakeGroup/shared_data/S_videos_annotations/S_videos/"
LABELED_S_DIRNAME = "/misc/vlgscratch4/LakeGroup/shared_data/S_clean_labeled_data_1fps_5"
FILTERED_LABELED_S_DIRNAME = "/misc/vlgscratch4/LakeGroup/shared_data/S_clean_labeled_data_clip_filtered"
EXTRACTED_FRAMES_DIRNAME = DATA_DIR / "train_5fps"
EVAL_FRAMES_DIRNAME = DATA_DIR / "eval"
FILTERED_EVAL_FRAMES_DIRNAME = DATA_DIR / "eval_filtered"
MANUAL_FILTERED_EVAL_FRAMES_DIRNAME = DATA_DIR / "eval_manual_filtered"
ANIMATED_FRAMES_DIRNAME = DATA_DIR / "train_animated_5fps"
TRAIN_METADATA_FILENAME = DATA_DIR / "train.json"
TRAIN_SHUFFLED_METADATA_FILENAME = DATA_DIR / "train_shuffled.json"
VAL_METADATA_FILENAME = DATA_DIR / "val.json"
TEST_METADATA_FILENAME = DATA_DIR / "test.json"
EVAL_DEV_METADATA_FILENAME = DATA_DIR / "eval_dev.json"
EVAL_TEST_METADATA_FILENAME = DATA_DIR / "eval_test.json"
FILTERED_EVAL_DEV_METADATA_FILENAME = DATA_DIR / "eval_filtered_dev.json"
FILTERED_EVAL_TEST_METADATA_FILENAME = DATA_DIR / "eval_filtered_test.json"
MANUAL_FILTERED_EVAL_TEST_METADATA_FILENAME = DATA_DIR / "eval_manual_filtered_test.json"
VOCAB_FILENAME = DATA_DIR / "vocab.json"

# default arguments
# dataset arguments
TRAIN_FRAC = 0.9
VAL_FRAC = 0.05

# sampling arguments
MAX_FRAMES_PER_UTTERANCE = 32

# training arguments
MULTIPLE_FRAMES = False
SHUFFLE_UTTERANCES = False


class MultiModalSAYCamDataset(MultiModalDataset):
    """
    Dataset that returns paired image-utterances from baby S of the SAYCam Dataset.
    """

    def __init__(self, data, vocab, multiple_frames, transform):
        super().__init__()
        self.data = data
        self.vocab = vocab
        self.multiple_frames = multiple_frames
        self.transform = transform

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Returns an image-utterance pair in tuple
        (img, utterance_idxs, utterance_length, raw_utterances)
        """

        # get utterance and convert to indices
        utterance = self.data[idx]["utterance"]
        utterance_words = utterance.split()
        utterance_words = [SOS_TOKEN] + utterance_words + [EOS_TOKEN]
        utterance_length = len(utterance_words)
        utterance_idxs = torch.tensor([self.vocab.get(
            word, UNK_TOKEN_ID) for word in utterance_words], dtype=torch.long)

        # get image
        img_filenames = self.data[idx]["frame_filenames"]

        if self.multiple_frames:
            # sample a random image associated with this utterance
            img_filename = Path(EXTRACTED_FRAMES_DIRNAME,
                                random.choice(img_filenames))
        else:
            # otherwise, sample the first frame
            img_filename = Path(EXTRACTED_FRAMES_DIRNAME, img_filenames[0])

        img = Image.open(img_filename).convert("RGB")

        # apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, utterance_idxs, utterance_length, [utterance]


class MultiModalSAYCamDataModule(MultiModalDataModule):
    """
    A data module created from baby S of the SAYCam Dataset consisting of
    image frames and the associated child-directed utterances.
    """

    def __init__(self, args=None) -> None:
        super().__init__(args)

        self.multiple_frames = self.args.get(
            "multiple_frames", MULTIPLE_FRAMES)
        self.shuffle_utterances = self.args.get(
            "shuffle_utterances", SHUFFLE_UTTERANCES)

    @staticmethod
    def add_additional_to_argparse(parser):
        parser.add_argument(
            "--multiple_frames", action="store_true", help="Randomly sample frames per utterance."
        )
        parser.add_argument(
            "--shuffle_utterances", action="store_true",
            help="Use shuffled utterances during training rather than matched utterances"
        )
        return parser

    @staticmethod
    def add_to_argparse(parser):
        parser = super(MultiModalSAYCamDataModule,
                       MultiModalSAYCamDataModule).add_to_argparse(parser)
        parser = MultiModalSAYCamDataModule.add_additional_to_argparse(parser)
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        super().prepare_data(*args, **kwargs)
        _download_transcripts()
        _rename_transcripts()
        _preprocess_transcripts()
        _extract_train_frames()
        _create_train_metadata()
        _create_train_shuffled_metadata()
        _filter_eval_frames()
        _extract_eval_frames()
        _extract_filtered_eval_frames()
        _create_eval_metadata()
        _create_filtered_eval_metadata()
        _create_manual_filtered_eval_metadata()
        _create_extra_eval_metadata()
        _create_extra_filtered_eval_metadata()
        _create_vocab()
        # _create_animations()  # TODO: add extra argument to generate this?

    def read_vocab(self):
        return read_vocab(VOCAB_FILENAME)

    def create_datasets(self, vocab):
        datasets = {}

        if self.shuffle_utterances:
            # use shuffled training data
            print("Training using shuffled utterances!")
            stage_splits = [("train", TRAIN_SHUFFLED_METADATA_FILENAME, self.multiple_frames,
                             self.transform),
                            ("val", VAL_METADATA_FILENAME,
                             False, self.base_transform),
                            ("test", TEST_METADATA_FILENAME, False, self.base_transform)]
        else:
            # use matched training data
            print("Training using matched utterances!")
            stage_splits = [("train", TRAIN_METADATA_FILENAME, self.multiple_frames,
                             self.transform),
                            ("val", VAL_METADATA_FILENAME,
                             False, self.base_transform),
                            ("test", TEST_METADATA_FILENAME, False, self.base_transform)]

        for split, filename, multiple_frames, transform in stage_splits:
            data = load_data(filename)
            dataset = MultiModalSAYCamDataset(
                data,
                vocab,
                multiple_frames=multiple_frames,
                transform=transform,
            )
            datasets[split] = dataset

        return datasets


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
                    # convert worksheet to data frame
                    df = s.sheets[j].to_frame()
                    # get filename of dataframe
                    filename = f"{TRANSCRIPTS_DIRNAME}/{title}_{s.sheets[j].title}.csv"
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
        allowed_speakers = ["M", "Mom", "mom", "m",
                            "mother", "Mother", "papa", "the mom"]

        # build spacy model
        nlp = spacy.load("en_core_web_sm")

        # preprocess each transcript
        for transcript_idx, transcript_filename in enumerate(transcripts):
            # empty list to store processed transcript information
            preprocessed_transcript = []
            preprocessed_transcript_filename = PREPROCESSED_TRANSCRIPTS_DIRNAME / \
                transcript_filename.name

            # read transcript CSV
            print(
                f"Preprocessing transcript: {transcript_filename.name} ({transcript_idx+1}/{len(transcripts)})")
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
                    # this will sample a single frame for the last utterance
                    end_timestamp = start_timestamp

                # skip processing utterance if start or end timestamps are null,
                # or if speaker is not in the list of allowed speakers
                if pd.isnull(start_timestamp) or pd.isnull(end_timestamp) or speaker not in allowed_speakers:
                    continue

                # preprocess utterance to extract sub-utterances and timestamps
                utterances, timestamps, num_frames = _preprocess_utterance(
                    nlp, utterance, start_timestamp, end_timestamp)

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
                preprocessed_transcript_df.to_csv(
                    preprocessed_transcript_filename, index=False)


def _preprocess_utterance(nlp, utterance, start_timestamp, end_timestamp):
    """Preprocesses a single utterance, splitting it into multiple clean utterances with separate timestamps"""

    # check start timestamp is before end timestamp
    assert start_timestamp <= end_timestamp

    # remove anything in asterisks or parentheses.
    inaudible = "INAUDIBLE"
    repl = lambda matchobj: inaudible if "inaudible" in matchobj.group(0) else ""
    utterance = re.sub(r"\*[^)]*\*", repl, utterance)
    utterance = re.sub(r"\[[^)]*\]", repl, utterance)
    utterance = re.sub(r"\([^)]*\)", repl, utterance)
    utterance = re.sub(r"\binaudible\b", repl, utterance)
    utterance = utterance.replace(r"*", "")

    # process utterance
    doc = nlp(utterance)
    utterances = [' '.join(
        map(lambda token: UNK_TOKEN if token == inaudible else token.lower(),
            map(str, sent))
        ) for sent in doc.sents]

    if len(utterances) > 0:
        # get interpolated timestamps, including end timestamp (which we remove later)
        timestamps = np.linspace(
            start_timestamp, end_timestamp, len(utterances)+1, endpoint=True)
        timestamps = [int(timestamp) for timestamp in timestamps]
        all_timestamps = []
        num_frames = []

        # calculate number of frames to extract per utterance (max: 32 frames at 5fps)
        for i in range(len(timestamps)-1):
            curr_num_frames = max(
                min(int((timestamps[i+1] - timestamps[i]) / 0.2), MAX_FRAMES_PER_UTTERANCE), 1)
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
        transcripts = sorted(
            Path(PREPROCESSED_TRANSCRIPTS_DIRNAME).glob("*.csv"))

        for idx, transcript in enumerate(transcripts):
            # get video filename associated with this transcript
            transcript_df = pd.read_csv(transcript)
            video_filename = Path(RAW_VIDEO_DIRNAME, pd.unique(
                transcript_df["video_filename"]).item())

            # skip if video doesn"t exist
            if not video_filename.exists():
                print(f"{video_filename} missing! Skipping")
                continue

            # otherwise continue extraction process
            print(
                f"Extracting frames: {video_filename.name} ({idx+1}/{len(transcripts)})")

            # read in video and get information
            cap = cv.VideoCapture(str(video_filename))
            video_info = _get_video_info(cap)
            frame_count, frame_width, frame_height, frame_rate, frame_length = video_info

            for transcript_row_idx, row in transcript_df.iterrows():
                # get information for frame extraction
                frame_filename = Path(
                    EXTRACTED_FRAMES_DIRNAME, str(row["frame_filename"]))
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
    resized_minor_length = 256
    new_height = frame_height * \
        resized_minor_length // min(frame_height, frame_width)
    new_width = frame_width * \
        resized_minor_length // min(frame_height, frame_width)

    # function to resize frame and recolor
    try:
        resized_frame = cv.resize(
            frame, (new_width, new_height), interpolation=cv.INTER_CUBIC)
    except Exception as e:
        print(str(e))
        return None

    # crop
    height, width, _ = resized_frame.shape
    startx = width // 2 - (IMAGE_W // 2)
    starty = height // 2 - (IMAGE_H // 2) - 16
    cropped_frame = resized_frame[starty:starty +
                                  IMAGE_H, startx:startx + IMAGE_W]
    assert cropped_frame.shape[0] == IMAGE_H and cropped_frame.shape[1] == IMAGE_W, \
        (cropped_frame.shape, height, width)

    # reverse x/y axes
    cropped_frame = np.array(cropped_frame)
    cropped_frame = cropped_frame[::-1, ::-1, :]
    # cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    return cropped_frame


def _filter_eval_frames():
    """Use CLIP to create a filtered evaluation set"""

    if os.path.exists(FILTERED_LABELED_S_DIRNAME):
        print("Evaluation frames have already been filtered. Skipping this step.")
    else:
        print("Filtering evaluation frames using CLIP")
        
        # get evaluation categories and create folders
        eval_categories = sorted(os.listdir(LABELED_S_DIRNAME))
        eval_categories.remove("carseat")
        eval_categories.remove("couch")
        eval_categories.remove("greenery")
        eval_categories.remove("plushanimal")

        # create directories
        os.makedirs(FILTERED_LABELED_S_DIRNAME, exist_ok=True)
        for eval_category in eval_categories:
            os.makedirs(Path(FILTERED_LABELED_S_DIRNAME) /
                        eval_category, exist_ok=True)

        # load CLIP model
        model, preprocess = clip.load("ViT-B/16", device=device)
        model.eval()

        # get CLIP text embedding for eval categories
        texts = clip.tokenize(
            [f'{category}' for category in eval_categories]).to(device)

        # encode text features and normalize
        text_features = model.encode_text(texts).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        for i, eval_category in enumerate(eval_categories):
            # get frames for each evaluation category
            eval_category_dir = os.path.join(LABELED_S_DIRNAME, eval_category)
            frames = glob.glob(f"{eval_category_dir}/*.jpeg")
            print(
                f"Filtering {len(frames)} from the category: {eval_category}")

            for frame in frames:
                # load and encode image via CLIP
                I = Image.open(frame).convert("RGB")
                image = preprocess(I).unsqueeze(0).to(device)
                image_features = model.encode_image(image).float()

                # normalize features
                image_features /= image_features.norm(dim=-1,
                                                      keepdim=True)

                # calculate top label
                logits_per_text = (100.0 * image_features @
                                   text_features.T).softmax(dim=-1)
                pred = torch.argmax(logits_per_text, dim=-1).item()

                # copy over image frame if prediction is correct
                if pred == eval_categories.index(eval_category):
                    frame_filename = frame.split("/")[-1]
                    print(f"Copying {frame_filename} to filtered set")
                    new_frame = os.path.join(
                        FILTERED_LABELED_S_DIRNAME, eval_category, frame_filename)
                    shutil.copyfile(frame, new_frame)                

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
            eval_category_dirname = os.path.join(
                LABELED_S_DIRNAME, eval_category)
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
                os.makedirs(os.path.join(
                    EVAL_FRAMES_DIRNAME, "dev", eval_category))

            for dev_idx in dev_idxs:
                # get path to original frame
                original_filename = os.path.join(
                    LABELED_S_DIRNAME, eval_category, eval_category_frames[dev_idx])

                # copy frame
                shutil.copyfile(original_filename, os.path.join(
                    EVAL_FRAMES_DIRNAME, "dev", eval_category, eval_category_frames[dev_idx]))

            # copy over test frames into a new directory
            print(f"copying {eval_category} frames for test set")

            # check if directory exists, and if not, create it
            if not os.path.exists(os.path.join(EVAL_FRAMES_DIRNAME, "test", eval_category)):
                os.makedirs(os.path.join(
                    EVAL_FRAMES_DIRNAME, "test", eval_category))

            for test_idx in test_idxs:
                # get path to original frame
                original_filename = os.path.join(
                    LABELED_S_DIRNAME, eval_category, eval_category_frames[test_idx])

                # copy frame
                shutil.copyfile(original_filename, os.path.join(
                    EVAL_FRAMES_DIRNAME, "test", eval_category, eval_category_frames[test_idx]))


def _extract_filtered_eval_frames():
    """Extract evaluation frames from (CLIP filtered) labeled S dataset, splitting evenly for dev and test"""

    if os.path.exists(FILTERED_EVAL_FRAMES_DIRNAME):
        print("Filtered evaluation frames have already been extracted. Skipping this step.")
    else:
        print("Extracting filtered evaluation frames")
        # create directory to store evaluation frames
        if not os.path.exists(FILTERED_EVAL_FRAMES_DIRNAME):
            os.makedirs(FILTERED_EVAL_FRAMES_DIRNAME)
            os.makedirs(FILTERED_EVAL_FRAMES_DIRNAME / "dev")
            os.makedirs(FILTERED_EVAL_FRAMES_DIRNAME / "test")

        # get original set of evaluation categories
        eval_categories = sorted(os.listdir(FILTERED_LABELED_S_DIRNAME))
        for eval_category in eval_categories:
            eval_category_dirname = os.path.join(
                FILTERED_LABELED_S_DIRNAME, eval_category)
            eval_category_frames = sorted(os.listdir(eval_category_dirname))

            # get indices to split original labeled s dataset into dev and test
            split_idxs = np.arange(len(eval_category_frames))
            np.random.shuffle(split_idxs)
            dev_idxs = split_idxs[:int(len(eval_category_frames) * 0.5)]
            test_idxs = split_idxs[int(len(eval_category_frames) * 0.5):]

            # check dataset has been split correct
            assert len(dev_idxs) + len(test_idxs) == len(split_idxs)

            # copy over dev frames into a new directory
            print(f"copying filtered {eval_category} frames for dev set")

            # check if directory exists, and if not, create it
            if not os.path.exists(os.path.join(FILTERED_EVAL_FRAMES_DIRNAME, "dev", eval_category)):
                os.makedirs(os.path.join(
                    FILTERED_EVAL_FRAMES_DIRNAME, "dev", eval_category))

            for dev_idx in dev_idxs:
                # get path to original frame
                original_filename = os.path.join(
                    FILTERED_LABELED_S_DIRNAME, eval_category, eval_category_frames[dev_idx])

                # copy frame
                shutil.copyfile(original_filename, os.path.join(
                    FILTERED_EVAL_FRAMES_DIRNAME, "dev", eval_category, eval_category_frames[dev_idx]))

            # copy over test frames into a new directory
            print(f"copying filtered {eval_category} frames for test set")

            # check if directory exists, and if not, create it
            if not os.path.exists(os.path.join(FILTERED_EVAL_FRAMES_DIRNAME, "test", eval_category)):
                os.makedirs(os.path.join(
                    FILTERED_EVAL_FRAMES_DIRNAME, "test", eval_category))

            for test_idx in test_idxs:
                # get path to original frame
                original_filename = os.path.join(
                    FILTERED_LABELED_S_DIRNAME, eval_category, eval_category_frames[test_idx])

                # copy frame
                shutil.copyfile(original_filename, os.path.join(
                    FILTERED_EVAL_FRAMES_DIRNAME, "test", eval_category, eval_category_frames[test_idx]))


def _create_train_metadata():
    """Creates JSON files with image-utterance information"""

    if os.path.exists(TRAIN_METADATA_FILENAME) and os.path.exists(VAL_METADATA_FILENAME) and os.path.exists(TEST_METADATA_FILENAME):
        print("Training metadata files have already been created. Skipping this step.")
    else:
        print("Creating metadata files for train, validation and test.")

        # get all preprocessed transcripts
        transcripts = sorted(
            Path(PREPROCESSED_TRANSCRIPTS_DIRNAME).glob("*.csv"))

        utterances = []

        for idx, transcript in enumerate(transcripts):
            # read in preprocessed transcript
            transcript_df = pd.read_csv(transcript)

            # group by utterances
            utterance_groups = transcript_df.groupby("utterance_num")
            for utterance, utterance_group in utterance_groups:
                # extract relevant information
                curr_utterance = {}
                curr_utterance["utterance"] = pd.unique(
                    utterance_group["utterance"]).item()
                curr_utterance["transcript_filename"] = pd.unique(
                    utterance_group["transcript_filename"]).item()
                curr_utterance["video_filename"] = pd.unique(
                    utterance_group["video_filename"]).item()
                curr_utterance["utterance_num"] = pd.unique(
                    utterance_group["utterance_num"]).item()
                curr_utterance["num_frames"] = len(utterance_group)
                curr_utterance["timestamps"] = list(
                    utterance_group["timestamp"])

                # extract filenames separately
                # initialize as empty list
                curr_utterance["frame_filenames"] = []
                curr_utterance_filenames = sorted(
                    list(utterance_group["frame_filename"]))

                # skip over any nan utterances
                if not isinstance(curr_utterance["utterance"], str):
                    continue

                # check frame filenames and append all frames that exist
                for frame_filename in curr_utterance_filenames:
                    if (EXTRACTED_FRAMES_DIRNAME / frame_filename).exists():
                        curr_utterance["frame_filenames"].append(
                            frame_filename)
                    else:
                        print(
                            f"{frame_filename} does not exist, removing it from this list")

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


def _create_train_shuffled_metadata():
    """Creates a JSON containing a shuffled version of the training data with image-utterance pairs randomly paired"""

    if os.path.exists(TRAIN_SHUFFLED_METADATA_FILENAME):
        print(
            "Shuffled training metadata file has already been created. Skipping this step.")
    else:
        print("Creating metadata for shuffled train.")

        # get train metadata
        with open(TRAIN_METADATA_FILENAME) as f:
            train_metadata = json.load(f)
            train_metadata = train_metadata["data"]

        # get list of utterances and shuffle
        utterances = [trial["utterance"] for trial in train_metadata]
        random.shuffle(utterances)

        # re-assign shuffled utterances
        for i, trial in enumerate(train_metadata):
            trial["utterance"] = utterances[i]

        train_shuffled_dict = {"data": train_metadata}

        # save shuffled metadata file
        with open(TRAIN_SHUFFLED_METADATA_FILENAME, "w") as f:
            json.dump(train_shuffled_dict, f)


def _create_eval_metadata():
    """Creates files for evaluating multimodal SAYCam model"""

    if os.path.exists(EVAL_DEV_METADATA_FILENAME) and os.path.exists(EVAL_TEST_METADATA_FILENAME):
        print("Evaluation metadata files have already been created. Skipping this step.")
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
                target_category_dirname = os.path.join(
                    eval_dev_dirname, target_category)
                target_img_filename = os.path.join(target_category_dirname,
                                                   np.random.choice(os.listdir(target_category_dirname)))

                foil_categories = eval_categories.copy()
                foil_categories.remove(target_category)
                foil_categories = np.random.choice(
                    foil_categories, size=n_foils, replace=False)
                foil_img_filenames = []

                for j in range(n_foils):
                    foil_category_dirname = os.path.join(
                        eval_dev_dirname, foil_categories[j])
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
                target_category_dirname = os.path.join(
                    eval_test_dirname, target_category)
                target_img_filename = os.path.join(target_category_dirname,
                                                   np.random.choice(os.listdir(target_category_dirname)))

                foil_categories = eval_categories.copy()
                foil_categories.remove(target_category)
                foil_categories = np.random.choice(
                    foil_categories, size=n_foils, replace=False)
                foil_img_filenames = []

                for j in range(n_foils):
                    foil_category_dirname = os.path.join(
                        eval_test_dirname, foil_categories[j])
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


def _create_filtered_eval_metadata():
    """Creates files for evaluating multimodal SAYCam model using filtered evaluation frames"""

    if os.path.exists(FILTERED_EVAL_DEV_METADATA_FILENAME) and os.path.exists(FILTERED_EVAL_TEST_METADATA_FILENAME):
        print("Evaluation metadata files have already been created. Skipping this step.")
    else:
        print("Creating metadata files for evaluation using filtered evaluation frames.")
        
        n_foils = 3  # number of foil referents
        n_evaluations = 100  # number of evaluations per category
        eval_dev_dataset = []
        eval_test_dataset = []

        # get evaluation categories and remove ones not in vocab
        eval_dev_dirname = FILTERED_EVAL_FRAMES_DIRNAME / "dev"
        eval_test_dirname = FILTERED_EVAL_FRAMES_DIRNAME / "test"
        eval_categories = sorted(os.listdir(eval_dev_dirname))

        # generate dev evaluation trials
        for target_category in eval_categories:
            for i in range(n_evaluations):
                # sample item from target category
                target_category_dirname = os.path.join(
                    eval_dev_dirname, target_category)
                target_img_filename = os.path.join(target_category_dirname,
                                                   np.random.choice(os.listdir(target_category_dirname)))

                foil_categories = eval_categories.copy()
                foil_categories.remove(target_category)
                foil_categories = np.random.choice(
                    foil_categories, size=n_foils, replace=False)
                foil_img_filenames = []

                for j in range(n_foils):
                    foil_category_dirname = os.path.join(
                        eval_dev_dirname, foil_categories[j])
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
                target_category_dirname = os.path.join(
                    eval_test_dirname, target_category)
                target_img_filename = os.path.join(target_category_dirname,
                                                   np.random.choice(os.listdir(target_category_dirname)))

                foil_categories = eval_categories.copy()
                foil_categories.remove(target_category)
                foil_categories = np.random.choice(
                    foil_categories, size=n_foils, replace=False)
                foil_img_filenames = []

                for j in range(n_foils):
                    foil_category_dirname = os.path.join(
                        eval_test_dirname, foil_categories[j])
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
        with open(FILTERED_EVAL_DEV_METADATA_FILENAME, "w") as f:
            json.dump(eval_dev_dict, f)

        with open(FILTERED_EVAL_TEST_METADATA_FILENAME, "w") as f:
            json.dump(eval_test_dict, f)

def _create_manual_filtered_eval_metadata():
    """
    Creates files for evaluating multimodal SAYCam model using manually filtered evaluation frames.
    Note that this eval only contains 15, rather than 22 categories, since we removed scene and
    other overlapping categories, and only contains a test folder.
    """

    if os.path.exists(MANUAL_FILTERED_EVAL_TEST_METADATA_FILENAME):
        print("Manual filtered evaluation metadata files have already been created. Skipping this step.")
    else:
        print("Creating metadata files for evaluation using manually filtered evaluation frames.")
        
        n_foils = 3  # number of foil referents
        n_evaluations = 100  # number of evaluations per category
        eval_test_dataset = []

        # get evaluation categories and remove ones not in vocab
        eval_test_dirname = MANUAL_FILTERED_EVAL_FRAMES_DIRNAME / "test"
        eval_categories = sorted(os.listdir(eval_test_dirname))
        print(eval_categories)

        # generate test evaluation trials
        for target_category in eval_categories:
            for i in range(n_evaluations):
                # sample item from target category
                target_category_dirname = os.path.join(
                    eval_test_dirname, target_category)
                target_img_filename = os.path.join(target_category_dirname,
                                                   np.random.choice(os.listdir(target_category_dirname)))

                foil_categories = eval_categories.copy()
                foil_categories.remove(target_category)
                foil_categories = np.random.choice(
                    foil_categories, size=n_foils, replace=False)
                foil_img_filenames = []

                for j in range(n_foils):
                    foil_category_dirname = os.path.join(
                        eval_test_dirname, foil_categories[j])
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

        # put eval trials into dictionary
        eval_test_dict = {"data": eval_test_dataset}

        # save as JSON file
        with open(MANUAL_FILTERED_EVAL_TEST_METADATA_FILENAME, "w") as f:
            json.dump(eval_test_dict, f)
            

def _generate_eval_trial(idx, stage, target_category, n_foils, eval_categories):
    """Generate a single evaluation trial with one category label and N images"""
    eval_dirname = EVAL_FRAMES_DIRNAME / f"{stage}"  # get directories

    # sample item from target category
    target_category_dirname = os.path.join(eval_dirname, target_category)
    target_img_filename = os.path.join(target_category_dirname,
                                       np.random.choice(os.listdir(target_category_dirname)))

    # randomly sample foil categories
    foil_categories = eval_categories.copy()
    foil_categories.remove(target_category)  # remove target category from list
    foil_categories = np.random.choice(
        foil_categories, size=n_foils, replace=False)
    foil_img_filenames = []

    # randomly sample foil items
    for i in range(n_foils):
        foil_category_dirname = os.path.join(eval_dirname, foil_categories[i])
        foil_img_filename = os.path.join(foil_category_dirname,
                                         np.random.choice(os.listdir(foil_category_dirname)))
        foil_img_filenames.append(foil_img_filename)

    # save trial info as a dict
    eval_trial = {}
    eval_trial["trial_num"] = idx
    eval_trial["target_category"] = target_category
    eval_trial["target_img_filename"] = target_img_filename
    eval_trial["foil_categories"] = list(foil_categories)
    eval_trial["foil_img_filenames"] = foil_img_filenames
    return eval_trial


def _create_extra_eval_metadata():
    """Create extra splits for evaluating Multimodal SAYCam models using 10 or 22 possible images per trial"""
    if os.path.exists(EVAL_DEV_METADATA_FILENAME) and os.path.exists(EVAL_TEST_METADATA_FILENAME):
        print(
            "Extra evaluation metadata files have already been created. Skipping this step.")
    else:
        print("Creating extra metadata files for evaluation.")

        stages = ["dev", "test"]
        n_foils = [9, 21]  # number of foil referents
        conds = itertools.product(stages, n_foils)
        n_evaluations = 100  # number of evaluations per category

        # get evaluation categories and remove ones not in vocab
        eval_dev_dirname = EVAL_FRAMES_DIRNAME / "dev"
        eval_categories = sorted(os.listdir(eval_dev_dirname))
        eval_categories.remove("carseat")
        eval_categories.remove("couch")
        eval_categories.remove("greenery")
        eval_categories.remove("plushanimal")

        for cond in conds:
            # initialize condition
            stage = cond[0]
            n_foil = cond[1]
            eval_dataset = []

            # generate trials
            for target_category in eval_categories:
                for i in range(n_evaluations):
                    eval_trial = _generate_eval_trial(
                        i, stage, target_category, n_foil, eval_categories)
                    eval_dataset.append(eval_trial)

            # put dataset into dictionary
            eval_dict = {"data": eval_dataset}

            # save as JSON
            with open(DATA_DIR / f"eval_{stage}_{n_foil}_foils.json", "w") as f:
                json.dump(eval_dict, f)


def _create_extra_filtered_eval_metadata():
    """Create extra splits for evaluating Multimodal SAYCam models using 10 or 22 possible images per trial"""
    if os.path.exists(FILTERED_EVAL_DEV_METADATA_FILENAME) and os.path.exists(FILTERED_EVAL_TEST_METADATA_FILENAME):
        print(
            "Extra evaluation metadata files have already been created. Skipping this step.")
    else:
        print(
            "Creating extra metadata files for evaluation using filtered evaluation frames.")

        stages = ["dev", "test"]
        n_foils = [9, 21]  # number of foil referents
        conds = itertools.product(stages, n_foils)
        n_evaluations = 100  # number of evaluations per category

        # get evaluation categories and remove ones not in vocab
        eval_dev_dirname = FILTERED_EVAL_FRAMES_DIRNAME / "dev"
        eval_categories = sorted(os.listdir(eval_dev_dirname))

        for cond in conds:
            # initialize condition
            stage = cond[0]
            n_foil = cond[1]
            eval_dataset = []

            # generate trials
            for target_category in eval_categories:
                for i in range(n_evaluations):
                    eval_trial = _generate_eval_trial(
                        i, stage, target_category, n_foil, eval_categories)
                    eval_dataset.append(eval_trial)

            # put dataset into dictionary
            eval_dict = {"data": eval_dataset}

            # save as JSON
            with open(DATA_DIR / f"eval_filtered_{stage}_{n_foil}_foils.json", "w") as f:
                json.dump(eval_dict, f)

def _create_vocab(freq_threshold=3):
    """Create vocabulary object and save to file"""

    if VOCAB_FILENAME.exists():
        print("Vocabulary file already exists. Skipping this step.")
    else:
        print("Creating vocab.json file!")

        counter = Counter()

        # load utterances from training set
        with open(TRAIN_METADATA_FILENAME) as f:
            train_dict = json.load(f)

        # get token frequency
        for example in train_dict["data"]:
            utterance = example["utterance"]
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
        transcripts = sorted(
            Path(PREPROCESSED_TRANSCRIPTS_DIRNAME).glob("*.csv"))[:5]

        for idx, transcript in enumerate(transcripts):
            print(
                f"Creating animated gifs: {transcript} ({idx+1}/{len(transcripts)})")

            # read in preprocessed transcript
            transcript_df = pd.read_csv(transcript)

            # group by utterances
            utterance_groups = transcript_df.groupby("utterance_num")

            # create gif
            for utterance, utterance_group in utterance_groups:
                utterance_num = pd.unique(
                    utterance_group["utterance_num"]).item()
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
