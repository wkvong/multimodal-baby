import os
import re
import glob
import pickle
import shutil
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils import *

def convert_timestamps_to_seconds(timestamps):
    # function to convert a variety of timestamps in the SAYcam annotations to seconds
    # extracts the first timestamp and converts this to the number of seconds
    
    new_timestamps = []
    for timestamp in timestamps:
        if isinstance(timestamp, str):
            timestamp_one = msplit(timestamp, '-')[0]
         
            if timestamp_one is not '':
                splits = msplit(timestamp_one, (':', '.', ',', ';'))
         
                if splits[0] == '':
                    splits[0] = '0'
         
                if len(splits) == 1:
                    splits.append('0')
                else:
                    # sometimes only the tens of seconds are reported as single digits
                    # this converts this correctly
                    if splits[1] is '1':
                        splits[1] = '10'
                    elif splits[1] is '2':
                        splits[1] = '20'
                    elif splits[1] is '3':
                        splits[1] = '30'
                    elif splits[1] is '4':
                        splits[1] = '40'
                    elif splits[1] is '5':
                        splits[1] = '50'
         
                timestamp_one_secs = int(splits[0]) * 60 + int(splits[1])
                if timestamp_one_secs > 2000:
                    print(f'timestamp out of range: {timestamp_one_secs}')
                    timestamp_one_secs = np.nan
         
                new_timestamps.append(timestamp_one_secs)
            else:
                new_timestamps.append(None)  # handles non-empty string that is not a timestamp
        else:
            new_timestamps.append(None)  # handles non-strings like nans

    return new_timestamps

def extract_frame(frame, frame_height, frame_width):
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


def generate_train_frames():
    # generate train frames by extracting utterances and their corresponding frames from each video
    
    video_dir = '/misc/vlgscratch4/LakeGroup/shared_data/S_videos_annotations/videos/S'
    transcripts_dir = sorted(glob.glob('/misc/vlgscratch4/LakeGroup/shared_data/S_videos_annotations/annotations/S/*.csv'))
    parent_speakers = ['M', 'Mom', 'mom', 'm', 'mother', 'Mother', 'papa', 'the mom']  # allowed speakers

    # create list to store values
    transcript_info = []
    
    # go through each transcript separately
    for transcript_num, transcript_fn in enumerate(transcripts_dir):
        # read transcript csv
        print(f'extracting frames from: {transcript_fn.split("/")[-1]} ({transcript_num}/{len(transcripts_dir)})')
        transcript = pd.read_csv(transcript_fn)
        
        # only process non-empty transcripts
        if len(transcript) > 1:
            # get timestamps in seconds and create new column
            new_timestamps = convert_timestamps_to_seconds(transcript['Time'])
            transcript['Time (Seconds)'] = new_timestamps

            # reset utterance count
            utterance_num = 1
            
            # read in video
            video_name = pd.unique(transcript['Video Name'])  # check only a single video file per transcript
            if len(video_name) == 1:
                video_filename = pd.unique(transcript['Video Name'])[0][:-4] + '.mp4'  # switch from avi to mp4
                video_filename = os.path.join(video_dir, video_filename)

                # check video name matches transcript, otherwise skip
                if transcript_fn.split('/')[-1][:-4] != pd.unique(transcript['Video Name'])[0][:-4]:
                    continue
                
                # check if video exists
                if os.path.isfile(video_filename):
                    # start video processing
                    cap = cv2.VideoCapture(video_filename)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # leave this as a float
                    frame_length = frame_count // frame_rate

                    print(f'frame_count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')
                    print(f'frame_rate: {cap.get(cv2.CAP_PROP_FPS)}')
                    print(f'frame length: {frame_count // cap.get(cv2.CAP_PROP_FPS)}')

                    # loop over utterances, clean up and extract relevant information to save frames
                    for idx in range(len(transcript)):
                        # get current utterance, current timestamp and next timestamp
                        utterance = transcript['Utterance'][idx]
                        timestamp = transcript['Time (Seconds)'][idx]
                        speaker = transcript['Speaker'][idx]

                        # check if utterance is nan and goes to the next utterance
                        if pd.isnull(utterance):
                            continue

                        # get timestamp of next utterance for interpolating multiple utterances properly
                        if idx >= len(transcript) - 1:
                            next_timestamp = frame_length
                        else:
                            next_timestamp = transcript['Time (Seconds)'][idx + 1]

                        # clean up utterance by removing asterisks/brackets/parentheses
                        utterance = re.sub(r'\*[^)]*\*', '', utterance)
                        utterance = re.sub(r'\[[^)]*\]', '', utterance)
                        utterance = re.sub(r'\([^)]*\)', '', utterance)
                        utterance = utterance.lower().strip()

                        # split utterances by punctuation, and clean up further (remove commas, strip whitespace)
                        utterances = msplit(utterance, ('.', '?', '!'))
                        utterances = [utterance.replace(',', '').strip() for utterance in utterances if
                                      len(utterance) > 0]

                        # check that utterance is not empty and filter by parents
                        if len(utterances) > 0 and speaker in parent_speakers:
                            # check that neither timestamp is not a null value
                            if not pd.isnull(timestamp) and not pd.isnull(next_timestamp):
                                # calculate interpolated timestamps for each sub-utterance
                                timesteps = np.linspace(timestamp, next_timestamp, len(utterances), endpoint=False)
                                timesteps = [int(timestep) for timestep in timesteps]
     
                                for timestep, utterance in zip(timesteps, utterances):
                                    # get all info and append to transcript info
                                    transcript_fn_base = transcript_fn.split('/')[-1][:-4]
                                    video_filename = f'{transcript_fn_base}.mp4'
                                    frame_filename = f'{transcript_fn_base}_{utterance_num}.jpg'
                                    original_timestamp = transcript['Time'][idx]
                                    
                                    # extract frame based on timestep
                                    cap.set(1, int(timestep * frame_rate))  # set frame to extract from
                                    ret, frame = cap.read()  # read frame
                                    frame = extract_frame(frame, frame_height, frame_width)  # extract frame in the manner we want

                                    if frame is not None:
                                        # save frame
                                        cv2.imwrite(os.path.join('data/train', frame_filename), frame)

                                        # save info
                                        transcript_info.append([f'{transcript_fn_base}.csv', video_filename,
                                                                frame_filename, idx, timestep, original_timestamp,
                                                                utterance])
     
                                        # increment utterance counter
                                        utterance_num += 1
                else:
                    print(f'{video_filename} does not exist!')
            else:
                print('multiple video files found in transcript, skipping')
                print(pd.unique(transcript['Video Name']))

    # save transcript info
    transcript_columns = ['annotations_filename', 'video_filename', 'frame_filename', 'original_index', 'time', 'original_time', 'utterance']
    transcript_df = pd.DataFrame(transcript_info, columns=transcript_columns)
    transcript_df.to_csv('data/frame_utterance_pairs.csv', index=False)

def generate_eval_frames():
    # generate evaluation frames by partitioning the existing eval set into val and test splits
    # set random seed
    np.random.seed(0)

    # directories
    original_eval_dir = '/misc/vlgscratch4/LakeGroup/shared_data/S_labeled_data/S_labeled_data_1fps_4'
    new_val_dir = '/home/wv9/code/WaiKeen/multimodal-baby/data/evaluation/val/'
    new_test_dir = '/home/wv9/code/WaiKeen/multimodal-baby/data/evaluation/test/'
    
    eval_categories = os.listdir(original_eval_dir)
    for eval_category in eval_categories:
        eval_category_dir = os.path.join(original_eval_dir, eval_category)
        eval_category_frames = sorted(os.listdir(eval_category_dir))

        # get indices to split original eval dataset into val and test
        split_idxs = np.arange(len(eval_category_frames))
        np.random.shuffle(split_idxs)
        val_idxs = split_idxs[:int(len(eval_category_frames) * 0.2)]
        test_idxs = split_idxs[int(len(eval_category_frames) * 0.2):]

        # check dataset has been split correctly
        assert len(val_idxs) + len(test_idxs) == len(split_idxs)

        # copy over validation frames to new directory
        print(f'copying {eval_category} frames for validation set')
        for val_idx in tqdm(val_idxs):
            # get path to original frame
            original_filename = os.path.join(original_eval_dir, eval_category, eval_category_frames[val_idx])

            # check if directory exists, and if not, create it
            if not os.path.exists(os.path.join(new_val_dir, eval_category)):
                os.makedirs(os.path.join(new_val_dir, eval_category))

            # copy frame
            shutil.copyfile(original_filename, os.path.join(new_val_dir, eval_category, eval_category_frames[val_idx]))

        # copy over test frames to new directory
        print(f'copying {eval_category} frames for test set')
        for test_idx in tqdm(test_idxs):
            # get path to original frame
            original_filename = os.path.join(original_eval_dir, eval_category, eval_category_frames[test_idx])

            # check if directory exists, and if not, create it
            if not os.path.exists(os.path.join(new_test_dir, eval_category)):
                os.makedirs(os.path.join(new_test_dir, eval_category))

            # copy frame
            shutil.copyfile(original_filename, os.path.join(new_test_dir, eval_category, eval_category_frames[test_idx]))

            
def pad_collate_fn(batch):
    # extract elements from batch
    images, utterances, utterance_lengths = zip(*batch)

    # convert to torch tensor and add padding
    images = torch.stack(images, dim=0)
    utterances = [torch.LongTensor(utterance) for utterance in utterances]
    utterances = torch.LongTensor(pad_sequence(utterances, batch_first=True))
    utterance_lengths = torch.LongTensor(utterance_lengths)

    return images, utterances, utterance_lengths


def plot_image_utterance_pairs():
    # function to plot image-utterance pairs for visualization

    # get transcript information
    dataset = SAYcamTrainDataset()
    transcripts = dataset.transcripts
    base_dir = '/home/wv9/code/WaiKeen/multimodal-baby/data/'
    image_dir = os.path.join(base_dir, 'train')

    print(len(transcripts))
    print(transcripts.columns)

    # group by video
    # videos = transcripts.groupby('video_filename')
    video_name = transcripts['video_filename'].iloc[0]
    single_transcript = transcripts[transcripts['video_filename'] == video_name]
    print(video_name)
    print(single_transcript['utterance'])

    n_cols = 5
    n_rows = len(single_transcript) // n_cols + 1

    axes = []
    fig = plt.figure(figsize=(2 * n_cols, 2 * n_rows))

    for i in range(len(single_transcript)):
        image_filename = single_transcript['frame_filename'].iloc[i]
        image = np.array(Image.open(os.path.join(image_dir, image_filename)).convert('RGB'))
        
        utterance = single_transcript['utterance'].iloc[i]
        axes.append(fig.add_subplot(n_rows, n_cols, i+1))
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

    plt.savefig('test.png')

    # for name, video in videos:
    #     print(name)
    #     print(video)
    

class SAYcamTrainDataset(Dataset):
    # train dataset class
    def __init__(self):
        self.transcripts = pd.read_csv('data/frame_utterance_pairs.csv')
        print(len(self.transcripts))
        self._preprocess_transcripts()
        print(len(self.transcripts))
        
        self.base_dir = '/home/wv9/code/WaiKeen/multimodal-baby/data/'
        self.train_dir = os.path.join(self.base_dir, 'train')

        # image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # build vocab
        self._build_vocab()

    def __getitem__(self, i):
        # get caption and preprocess
        utterance = self.transcripts['utterance'].iloc[i]
        # if utterance == '' or utterance == '\"':
        #     print(f'found empty string! utterance: {utterance}, idx: {i}')
            
        utterance_words = utterance.split(' ')
        utterance_length = len(utterance_words)
        utterance_idxs = []
        for word in utterance_words:
            try:
                utterance_idxs.append(self.word2index[word])
            except KeyError:
                utterance_idxs.append(self.word2index['<unk>'])

        # get image and transform
        img_filename = os.path.join(self.train_dir, self.transcripts['frame_filename'].iloc[i])
        img = np.array(Image.open(img_filename).convert('RGB'))
        img = self.transform(img)
                
        return img, utterance_idxs, utterance_length

    def __len__(self):
        return len(self.transcripts)

    def _preprocess_transcripts(self):
        # replace any double quote chars and asterisks with empty strings
        self.transcripts['utterance'] = self.transcripts['utterance'].replace('"', '')
        self.transcripts['utterance'] = self.transcripts['utterance'].replace('*', '')

        # get indices of non-empty or non-null strings and keep these
        self.utterance_idxs = self.transcripts[(self.transcripts['utterance'].notnull()) & (self.transcripts['utterance'] != '')].index
        self.transcripts = self.transcripts.iloc[self.utterance_idxs]
        
    
    def _build_vocab(self):
        # builds vocabulary from utterances extracted from SAYcam
        if os.path.exists('data/vocab.pickle'):
            with open('data/vocab.pickle', 'rb') as f:
                vocab = pickle.load(f)
                self.word2index = vocab['word2index']
                self.num_words = vocab['num_words']
                self.max_len = vocab['max_len']
        else:
            self.word2index = {'<pad>': 0, '<unk>': 1}
            self.num_words = 2
            self.max_len = 0

            utterances = self.transcripts['utterance']
     
            print('building vocab')
            for utterance in tqdm(utterances):
                if isinstance(utterance, str):
                    utterance_words = utterance.split(' ')
          
                    # check if utterance length is larger than current max
                    if len(utterance_words) > self.max_len:
                        self.max_len = len(utterance_words)
       
                    # add words to vocab
                    for word in utterance_words:
                        if word not in self.word2index:
                            self.word2index[word] = self.num_words
                            self.num_words += 1
     
            # save vocab
            vocab = {'word2index': self.word2index, 'num_words': self.num_words, 'max_len': self.max_len}
            with open('data/vocab.pickle', 'wb') as f:
                pickle.dump(vocab, f)

class SAYcamValDataset(Dataset):
    # val dataset class
    pass

class SAYcamTestDataset(Dataset):
    # test dataset class
    pass

if __name__ == "__main__":
    # train_dataset = SAYcamTrainDataset()
    # for idx, (image, utterance, utterance_length) in enumerate(train_dataset):
    #     pass

    plot_image_utterance_pairs()
