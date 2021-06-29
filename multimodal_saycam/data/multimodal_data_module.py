from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union
import os
import glob
import json
import re
import time
import argparse

import numpy as np
import pandas as pd
from gsheets import Sheets
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from multimodal_saycam.data.base_data_module import BaseDataModule, load_and_print_info
from multimodal_saycam.data.util import msplit, convert_timestamps_to_seconds


BATCH_SIZE = 128
NUM_WORKERS = 0

GSHEETS_CREDENTIALS_FILENAME = BaseDataModule.data_dirname() / "desktop-credentials.json"
TRANSCRIPT_LINKS_FILENAME = BaseDataModule.data_dirname() / "SAYCam_transcript_links_new.csv"
TRANSCRIPTS_DIRNAME = BaseDataModule.data_dirname() / "transcripts"
RAW_VIDEO_DIRNAME = "/misc/vlgscratch4/LakeGroup/shared_data/S_videos_annotations/videos/S"
# RAW_TRANSCRIPT_DIRNAME = "/misc/vlgscratch4/LakeGroup/shared_data/S_videos_annotations/annotations/S/"
LABELED_S_DIR = ""
PROCESSED_TRANSCRIPT_DIRNAME = ""
PROCESSED_TRANSCRIPT_FILENAME = ""
EXTRACTED_FRAMES_DIRNAME = ""
VOCAB_FILENAME = ""

MAX_FRAMES_PER_UTTERANCE = 20
    
class MultiModalDataModule(BaseDataModule):
    """
    The MultiModal SAYCam Dataset is a dataset created from baby S of the SAYCam Dataset consisting of
    image frames and the associated child-directed utterances.
    """

    def __init__(self, args=None) -> None:
        super().__init__(args)
        
        # set other variables for our dataset here

    def prepare_data(self, *args, **kwargs) -> None:
        _download_transcripts()
        _rename_transcripts()
        _process_transcripts()
        _extract_frames()
        _process_dataset()
        _create_vocab()
    
    def setup(self) -> None:
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
        

def _download_transcripts():
    """Download SAYCam transcripts."""
    
    # check if transcripts have already been downloaded
    if os.path.exists(TRANSCRIPTS_DIRNAME) and not os.path.isfile(TRANSCRIPTS_DIRNAME):
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
            
def _process_transcripts():
    """Process transcripts by cleaning the text and extracting frame timings."""
    print("Processing transcripts")

    # get all transcripts and allowed speakers
    transcripts = sorted(Path(TRANSCRIPTS_DIRNAME).glob("*.csv"))[10:20]
    allowed_speakers = ['M', 'Mom', 'mom', 'm', 'mother', 'Mother', 'papa', 'the mom']

    for transcript_idx, transcript_filename in enumerate(transcripts):
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
        utterance_count = 1  # TODO: why is this 1 and not 0?

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

        # check if video exists
        # if not video_filename.exists():
        #     continue
        # else:
        #     video_info = _get_video_info(video_filename)
        #     frame_length = video_info[4]

        frame_length = 30 * 60  # TODO: current hack to get end of video

        for transcript_row_idx, row in transcript.iterrows():
            # get information from current utterance
            utterance = str(row['Utterance'])  # convert to string
            speaker = str(row['Speaker'])
            start_timestamp = row['Time (Seconds)']

            # get end timestamp (either start of next timestamp, or end of video)
            if transcript_row_idx < len(transcript) - 1:
                end_timestamp = transcript['Time (Seconds)'][transcript_row_idx+1]
            else:
                end_timestamp = frame_length

            # skip processing utterance if start or end timestamps are null, or speaker is not in
            # list of allowed speakers
            if pd.isnull(start_timestamp) or pd.isnull(end_timestamp) or speaker not in allowed_speakers:
                continue

            # utterances, timestamps, num_frames = _preprocess_utterance(utterance, start_timestamp, end_timestamp)
            _preprocess_utterance(utterance, start_timestamp, end_timestamp)

def _preprocess_utterance(utterance, start_timestamp, end_timestamp):
    """Preprocesses a single utterance, splitting it into multiple clean utterances with separate timestamps"""

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
        num_frames = []

        # calculate number of frames to extract per utterance (max: 20 frames)
        for i in range(len(timestamps)-1):
            curr_num_frames = min(timestamps[i+1] - timestamps[i], MAX_FRAMES_PER_UTTERANCE)
            num_frames.append(curr_num_frames)

        timestamps = timestamps[:-1]  # remove end timestamp

        # created nested timestamps??
    else:
        pass
    
            
def _extract_frames():
    """Extract aligned frames from SAYCam videos"""
    print("Extracting frames!")


def _get_video_info(video_filename):
    """Returns video information"""
    cap = cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # leave this as a float
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
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
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

    
def _process_dataset():
    print("Processing dataset!")

def _create_vocab():
    """Create vocabulary object and save to file"""
    print("Creating vocabulary!")

if __name__ == "__main__":
    load_and_print_info(MultiModalDataModule)
