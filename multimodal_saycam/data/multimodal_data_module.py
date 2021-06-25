from pathlib import Path
from typing import Collection, Dict, Optional, Tuple, Union
import os
import json
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
    """Manually rename a few of the downloaded transcripts that don't match the naming scheme"""

    print("Renaming transcripts")
    
    if os.path.exists(TRANSCRIPTS_DIRNAME / "S_20141029_2412_part 2.csv"):
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
        os.rename(TRANSCRIPTS_DIRNAME / "S_20141122_2505_part 1.csv",
                  TRANSCRIPTS_DIRNAME / "S_20141122_2505_01.csv")
        os.rename(TRANSCRIPTS_DIRNAME / "S_20141122_2505_part 2.csv",
                  TRANSCRIPTS_DIRNAME / "S_20141122_2505_02.csv")
            
def _process_transcripts():
    print("Processing transcripts")

def _extract_frames():
    print("Extracting frames!")

def _process_dataset():
    print("Processing dataset!")

def _create_vocab():
    print("Creating vocabulary!")

if __name__ == "__main__":
    load_and_print_info(MultiModalDataModule)
