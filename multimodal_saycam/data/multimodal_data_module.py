from pathlib import Path
from typing import Any, Callable, Collection, Dict, Optional, Tuple, Union
import os
import glob
import json
import random
import re
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
import pytorch_lightning as pl

import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from multimodal_saycam.data.base_data_module import BaseDataModule, load_and_print_info
from multimodal_saycam.data.util import *


BATCH_SIZE = 4
NUM_WORKERS = 0
TRAIN_FRAC = 0.95

GSHEETS_CREDENTIALS_FILENAME = BaseDataModule.data_dirname() / "credentials.json"
TRANSCRIPT_LINKS_FILENAME = BaseDataModule.data_dirname() / "SAYCam_transcript_links_new.csv"
TRANSCRIPTS_DIRNAME = BaseDataModule.data_dirname() / "transcripts"
PREPROCESSED_TRANSCRIPTS_DIRNAME = BaseDataModule.data_dirname() / "preprocessed_transcripts_5fps"
RAW_VIDEO_DIRNAME = "/misc/vlgscratch4/LakeGroup/shared_data/S_videos_annotations/S_videos/"
LABELED_S_DIR = ""
EXTRACTED_FRAMES_DIRNAME = BaseDataModule.data_dirname() / "train_5fps"
ANIMATED_FRAMES_DIRNAME = BaseDataModule.data_dirname() / "train_animated_5fps"
TRAIN_METADATA_FILENAME = BaseDataModule.data_dirname() / "train_5fps.json"
VOCAB_FILENAME = BaseDataModule.data_dirname() / "vocab_5fps.json"

MAX_FRAMES_PER_UTTERANCE = 32
MAX_LEN_UTTERANCE = 16

AUGMENT_FRAMES = False
USE_MULTIPLE_FRAMES = False

class MultiModalSAYCamDataset(Dataset):
    """
    Dataset that returns paired image-utterances from baby S of the SAYCam Dataset
    """
    
    def __init__(self, data: Dict, vocab: Dict, use_multiple_frames, transform: Callable = None):
        super().__init__()
        self.data = data
        self.vocab = vocab
        self.use_multiple_frames = use_multiple_frames
        self.transform = transform

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data)
    
    def __getitem__(self, i: int) -> Tuple[Any, Any, Any]:
        """
        Returns an image-utterance pair
        """

        # get utterance and convert to indices
        utterance = self.data[i]['utterance']
        # print(utterance)
        utterance_words = utterance.split(' ')
        utterance_length = len(utterance_words)
        utterance_idxs = np.zeros(MAX_LEN_UTTERANCE)  # initialize padded array
        for i, word in enumerate(utterance_words):
            if i >= MAX_LEN_UTTERANCE:
                break
            
            try:
                utterance_idxs[i] = self.vocab[word]
            except KeyError:
                utterance_idxs[i] = self.vocab['<unk>']

        # convert to torch tensor
        utterance_idxs = torch.LongTensor(utterance_idxs)

        # get image
        img_filenames = self.data[i]['frame_filenames']
        
        if self.use_multiple_frames:
            # sample a random image associated with this utterance
            img_filename = Path(EXTRACTED_FRAMES_DIRNAME, random.choice(img_filenames))
        else:
            # otherwise, sample the first frame
            img_filename = Path(EXTRACTED_FRAMES_DIRNAME, img_filenames[0])

        img = Image.open(img_filename).convert('RGB')

        # apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, utterance_idxs, utterance_length

    
class MultiModalSAYCamDataModule(BaseDataModule):
    """
    The MultiModal SAYCam Dataset is a dataset created from baby S of the SAYCam Dataset consisting of
    image frames and the associated child-directed utterances.
    """

    def __init__(self, args=None) -> None:
        super().__init__(args)

        self.args = vars(args) if args is not None else {}
        self.use_multiple_frames = self.args.get("use_multiple_frames", USE_MULTIPLE_FRAMES)
        self.augment_frames = self.args.get("augment_frames", AUGMENT_FRAMES)

        if self.augment_frames:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([transforms.ColorJitter(0.9, 0.9, 0.9, 0.5)], p=0.9),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),            
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def prepare_data(self, *args, **kwargs) -> None:
        _download_transcripts()
        _rename_transcripts()
        _preprocess_transcripts()
        _extract_frames()
        _create_train_metadata()
        # _create_animations()
        _create_vocab()
    
    def setup(self, *args, **kwargs) -> None:
        # read data
        with open(TRAIN_METADATA_FILENAME) as f:
            data = json.load(f)
            data = data['images']
            random.shuffle(data)  # TODO: remove later!! just for double checking something right now

        # read vocab
        with open(VOCAB_FILENAME) as f:
            vocab = json.load(f)

        # create dataset
        trainval_dataset = MultiModalSAYCamDataset(data, vocab, use_multiple_frames, transform=self.transform)
        self.train_dataset, self.val_dataset = split_dataset(trainval_dataset, fraction=TRAIN_FRAC, seed=0)
        

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
        urls = df['GoogleSheets Link'].unique()
            
        for i, url in enumerate(urls):
            print(f'Downloading SAYCam transcript {i+1}/{len(urls)}: {url}')
            s = sheets.get(url)
            title = s.title.split('_')
            title = '_'.join(title[:3])

            # read all sheets (skipping the first one since it is blank)
            for j in range(1, len(s.sheets)):
                try:
                    # try and parse this sheet as a data frame
                    df = s.sheets[j].to_frame()  # convert worksheet to data frame
                    filename = f'{TRANSCRIPTS_DIRNAME}/{title}_{s.sheets[j].title}.csv'  # get filename of dataframe
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
        allowed_speakers = ['M', 'Mom', 'mom', 'm', 'mother', 'Mother', 'papa', 'the mom']

        # preprocess each transcript
        for transcript_idx, transcript_filename in enumerate(transcripts):
            # empty list to store processed transcript information
            preprocessed_transcript = []
            preprocessed_transcript_filename = PREPROCESSED_TRANSCRIPTS_DIRNAME / transcript_filename.name
     
            # read transcript CSV
            print(f'Preprocessing transcript: {transcript_filename.name} ({transcript_idx+1}/{len(transcripts)})')
            transcript = pd.read_csv(transcript_filename)
     
            # skip empty transcripts
            if len(transcript) <= 1:
                continue
            
            # create new column of timestamps converted to seconds
            new_timestamps = convert_timestamps_to_seconds(transcript['Time'])
            transcript['Time (Seconds)'] = new_timestamps
     
            # reset utterance count
            utterance_num = 1
     
            # extract unique video filename from transcript
            video_filename = pd.unique(transcript['Video Name'])
     
            # drop any missing filenames, or any filenames with 'part' in them
            video_filename = [x for x in video_filename if not pd.isnull(x)]
            video_filename = [x for x in video_filename if 'part' not in x]
     
            # skip if video filename is not unique
            if len(video_filename) != 1:
                continue
     
            # extract video filename and replace suffix
            video_filename = video_filename[0]
            video_filename = Path(video_filename).with_suffix('.mp4')
     
            # check video and transcript filenames match
            assert video_filename.stem == transcript_filename.stem
     
            for transcript_row_idx, row in transcript.iterrows():
                # get information from current utterance
                utterance = str(row['Utterance'])  # convert to string
                speaker = str(row['Speaker'])
                start_timestamp = row['Time (Seconds)']
     
                # get end timestamp
                # hack: if last timestamp, just set end timestamp to be start time
                # this means we don't have to read the video file for this to work
                if transcript_row_idx < len(transcript) - 1:
                    end_timestamp = transcript['Time (Seconds)'][transcript_row_idx+1]
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
                        frame_filename = f'{video_filename.stem}_{utterance_num:03}_{frame_num:02}.jpg'
                        preprocessed_transcript.append([transcript_filename.name,
                            video_filename.name, curr_utterance, curr_timestamp,
                            utterance_num, frame_num, frame_filename])
     
                    utterance_num += 1
     
            # save preprocessed transcript as CSV
            if len(preprocessed_transcript) > 0:
                preprocessed_transcript_columns = ['transcript_filename', 'video_filename',
                    'utterance', 'timestamp', 'utterance_num', 'frame_num', 'frame_filename']
                preprocessed_transcript_df = pd.DataFrame(preprocessed_transcript,
                                                          columns=preprocessed_transcript_columns)
                preprocessed_transcript_df.to_csv(preprocessed_transcript_filename, index=False)


def _preprocess_utterance(utterance, start_timestamp, end_timestamp):
    """Preprocesses a single utterance, splitting it into multiple clean utterances with separate timestamps"""

    # check start timestamp is before end timestamp
    assert start_timestamp <= end_timestamp

    # remove special characters, anything in asterisks or parentheses etc.
    utterance = re.sub(r'\*[^)]*\*', '', utterance)
    utterance = re.sub(r'\[[^)]*\]', '', utterance)
    utterance = re.sub(r'\([^)]*\)', '', utterance)
    utterance = re.sub(r' +', ' ', utterance)
    utterance = utterance.replace('--', ' ')
    utterance = utterance.replace('-', '')
    utterance = utterance.replace('"', '')
    utterance = utterance.replace('*', '')
    utterance = utterance.replace('_', '')
    utterance = utterance.replace(',', '')
    utterance = utterance.replace('…', '')
    utterance = utterance.lower().strip()

    # split utterance based on certain delimeters, strip and remove empty utterances
    utterances = msplit(utterance, ('.', '?', '!'))
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
            
def _extract_frames():
    """Extract aligned frames from SAYCam videos"""

    if os.path.exists(EXTRACTED_FRAMES_DIRNAME):
        print("Frames have already been extracted. Skipping this step.")
    else:
        print("Extracting frames")

        # create directory to store extracted frames
        if not os.path.exists(EXTRACTED_FRAMES_DIRNAME):
            os.makedirs(EXTRACTED_FRAMES_DIRNAME)

        # get all preprocessed transcripts
        transcripts = sorted(Path(PREPROCESSED_TRANSCRIPTS_DIRNAME).glob("*.csv"))

        for idx, transcript in enumerate(transcripts):
            # get video filename associated with this transcript
            transcript_df = pd.read_csv(transcript)
            video_filename = Path(RAW_VIDEO_DIRNAME, pd.unique(transcript_df['video_filename']).item())

            # skip if video doesn't exist
            if not video_filename.exists():
                print(f'{video_filename} missing! Skipping')
                continue

            # otherwise continue extraction process
            print(f'Extracting frames: {video_filename.name} ({idx+1}/{len(transcripts)})')

            # read in video and get information
            cap = cv.VideoCapture(str(video_filename))
            video_info = _get_video_info(cap)
            frame_count, frame_width, frame_height, frame_rate, frame_length = video_info

            for transcript_row_idx, row in transcript_df.iterrows():
                # get information for frame extraction
                frame_filename = Path(EXTRACTED_FRAMES_DIRNAME, str(row['frame_filename']))
                timestamp = float(row['timestamp'])  # keep as float
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

    
def _create_train_metadata():
    """Creates a JSON file with image-utterance information"""
    
    if os.path.exists(TRAIN_METADATA_FILENAME):
        print("Training metadata file already been created . Skipping this step.")
    else:
        print("Creating training metadata file")

        # get all preprocessed transcripts
        transcripts = sorted(Path(PREPROCESSED_TRANSCRIPTS_DIRNAME).glob("*.csv"))

        utterances = []

        for idx, transcript in enumerate(transcripts):            
            # read in preprocessed transcript
            transcript_df = pd.read_csv(transcript)
            
            # group by utterances
            utterance_groups = transcript_df.groupby('utterance_num')
            for utterance, utterance_group in utterance_groups:
                # extract relevant information
                curr_utterance = {}
                curr_utterance['utterance'] = pd.unique(utterance_group['utterance']).item()
                curr_utterance['transcript_filename'] = pd.unique(utterance_group['transcript_filename']).item()
                curr_utterance['video_filename'] = pd.unique(utterance_group['video_filename']).item()
                curr_utterance['utterance_num'] = pd.unique(utterance_group['utterance_num']).item()
                curr_utterance['num_frames'] = len(utterance_group)
                curr_utterance['frame_filenames'] = list(utterance_group['frame_filename'])
                curr_utterance['timestamps'] = list(utterance_group['timestamp'])

                # skip over any nan utterances
                if not isinstance(curr_utterance['utterance'], str):
                    continue

                # check frame filenames and remove any frames that were not extracted
                for frame_filename in curr_utterance['frame_filenames']:
                    if not (EXTRACTED_FRAMES_DIRNAME / frame_filename).exists():
                        print(f'{frame_filename} does not exist, removing it from this list')
                        curr_utterance['frame_filenames'].remove(frame_filename)

                # skip utterance completely if no frames were extracted
                if len(curr_utterance['frame_filenames']) == 0:
                    print('No corresponding frames found, skipping this utterance')
                    continue
                
                # append details of remaining utterances to metadata list
                utterances.append(curr_utterance)

        # put utterances into a dictionary
        train_dict = {'images': utterances}

        # save as JSON file
        with open(TRAIN_METADATA_FILENAME, 'w') as f:
            json.dump(train_dict, f)

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
            print(f'Creating animated gifs: {transcript} ({idx+1}/{len(transcripts)})')
            
            # read in preprocessed transcript
            transcript_df = pd.read_csv(transcript)
            
            # group by utterances
            utterance_groups = transcript_df.groupby('utterance_num')
     
            # create gif
            for utterance, utterance_group in utterance_groups:
                utterance_num = pd.unique(utterance_group['utterance_num']).item()
                gif_filename = f"{pd.unique(utterance_group['transcript_filename']).item()[:-4]}_{utterance_num:03}.gif"
                gif_filepath = Path(ANIMATED_FRAMES_DIRNAME, gif_filename)
                frame_filenames = utterance_group['frame_filename']
     
                frames = []
                for frame_filename in frame_filenames:
                    frame_filepath = EXTRACTED_FRAMES_DIRNAME / frame_filename
     
                    try:
                        img = imageio.imread(frame_filepath)
                    except FileNotFoundError:
                        continue
                        
                    frames.append(img)
     
                if len(frames) > 0:
                    print(f'Saving {gif_filepath}, with {len(frames)} frames')
                    imageio.mimsave(gif_filepath, frames, fps=10)

            
def _create_vocab():
    """Create vocabulary object and save to file"""

    if VOCAB_FILENAME.exists():
        print("Vocabulary file already exists. Skipping this step.")
    else:
        print("Creating vocab.json file!")

        # create vocab dictionary
        vocab_dict = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 4}
        num_words = 4

        # load utterances
        with open(TRAIN_METADATA_FILENAME) as f:
            train_dict = json.load(f)

        # fill vocab with all words from utterances
        train = train_dict['images']
        for i in range(len(train)):
            curr_utterance = str(train[i]['utterance'])
            words = curr_utterance.split(' ')
            for word in words:
                if word not in vocab_dict:
                    vocab_dict[word] = num_words
                    num_words += 1

        # save as JSON file
        with open(VOCAB_FILENAME, 'w') as f:
            json.dump(vocab_dict, f)
        

if __name__ == "__main__":
    load_and_print_info(MultiModalSAYCamDataModule)
