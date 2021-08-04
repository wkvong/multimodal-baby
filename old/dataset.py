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
    # function to convert a variety of timestamps in the SAYCam annotations to seconds
    # extracts the first timestamp and converts this to the number of seconds
    
    new_timestamps = []
    for timestamp in timestamps:
        if isinstance(timestamp, str):
            timestamp_one = msplit(timestamp, '-')[0]
         
            if timestamp_one != '':
                splits = msplit(timestamp_one, (':', '.', ',', ';'))
         
                if splits[0] == '':
                    splits[0] = '0'
         
                if len(splits) == 1:
                    splits.append('0')
                else:
                    # sometimes only the tens of seconds are reported as single digits
                    # this converts this correctly
                    if splits[1] == '1':
                        splits[1] = '10'
                    elif splits[1] == '2':
                        splits[1] = '20'
                    elif splits[1] == '3':
                        splits[1] = '30'
                    elif splits[1] == '4':
                        splits[1] = '40'
                    elif splits[1] == '5':
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
    original_eval_dir = '/misc/vlgscratch4/LakeGroup/shared_data/S_clean_labeled_data_1fps_5'
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


def generate_val_dataset():
    # generate a fixed validation set from the extracted validation frames
    n_foils = 3
    n_evaluations_per_category = 100
    val_dataset = []

    # get val categories and remove categories not in vocab
    base_dir = 'data/evaluation'
    val_dir = os.path.join(base_dir, 'val')
    categories = sorted(os.listdir(val_dir))
    categories.remove('plushanimal')
    categories.remove('greenery')
    categories.remove('carseat')
    categories.remove('couch')
    n_categories = len(categories)

    for target_category in categories:
        for i in range(n_evaluations_per_category):
            # sample item from target category
            target_category_dir = os.path.join(val_dir, target_category)
            target_img_filename = os.path.join(target_category_dir,
                                               np.random.choice(os.listdir(target_category_dir)))

            all_foil_categories = categories.copy()
            all_foil_categories.remove(target_category)
            foil_categories = np.random.choice(all_foil_categories, size=n_foils, replace=False)
            foil_img_filenames = []

            for j in range(n_foils):
                foil_category_dir = os.path.join(val_dir, foil_categories[j])
                foil_img_filename = os.path.join(foil_category_dir,
                                             np.random.choice(os.listdir(foil_category_dir)))
                foil_img_filenames.append(foil_img_filename)

            trial = [target_category, *foil_categories, target_img_filename, *foil_img_filenames]
            val_dataset.append(trial)

    val_columns = ['target_category', 'foil_category_one',
                   'foil_category_two', 'foil_category_three',
                   'target_img_filename', 'foil_one_img_filename',
                   'foil_two_img_filename', 'foil_three_img_filename']
    val_dataset = pd.DataFrame(val_dataset, columns=val_columns)
    val_dataset.to_csv('data/validation.csv', index=False)


def generate_test_dataset():
    # generate a fixed test set from the extracted test frames
    n_foils = 3
    n_evaluations_per_category = 100
    test_dataset = []

    # get test categories and remove categories not in vocab
    base_dir = 'data/evaluation'
    test_dir = os.path.join(base_dir, 'test')
    categories = sorted(os.listdir(test_dir))
    categories.remove('plushanimal')
    categories.remove('greenery')
    categories.remove('carseat')
    categories.remove('couch')
    n_categories = len(categories)

    for target_category in categories:
        for i in range(n_evaluations_per_category):
            # sample item from target category
            target_category_dir = os.path.join(test_dir, target_category)
            target_img_filename = os.path.join(target_category_dir,
                                               np.random.choice(os.listdir(target_category_dir)))

            all_foil_categories = categories.copy()
            all_foil_categories.remove(target_category)
            foil_categories = np.random.choice(all_foil_categories, size=n_foils, replace=False)
            foil_img_filenames = []

            for j in range(n_foils):
                foil_category_dir = os.path.join(test_dir, foil_categories[j])
                foil_img_filename = os.path.join(foil_category_dir,
                                             np.random.choice(os.listdir(foil_category_dir)))
                foil_img_filenames.append(foil_img_filename)

            trial = [target_category, *foil_categories, target_img_filename, *foil_img_filenames]
            test_dataset.append(trial)

    test_columns = ['target_category', 'foil_category_one',
                   'foil_category_two', 'foil_category_three',
                   'target_img_filename', 'foil_one_img_filename',
                   'foil_two_img_filename', 'foil_three_img_filename']
    test_dataset = pd.DataFrame(test_dataset, columns=test_columns)
    test_dataset.to_csv('data/test.csv', index=False)
    
            
def pad_collate_fn(batch):
    # extract elements from batch
    images, utterances, utterance_lengths = zip(*batch)

    # convert to torch tensor and add padding
    images = torch.stack(images, dim=0)
    utterances = [torch.LongTensor(utterance) for utterance in utterances]
    utterances = torch.LongTensor(pad_sequence(utterances, batch_first=True))
    utterance_lengths = torch.LongTensor(utterance_lengths)

    return images, utterances, utterance_lengths


def preprocess_transcripts():
    # clean up extracted transcript with a few more things i didn't do in the original extraction step
    # removes additional unnecessary punctuation and also ignores empty strings
    
    transcripts = pd.read_csv('data/frame_utterance_pairs.csv')
    print(f'number of utterances: {len(transcripts)}')
    transcripts['utterance'] = transcripts['utterance'].replace('"', '', regex=True)
    transcripts['utterance'] = transcripts['utterance'].replace('*', '')
    transcripts['utterance'] = transcripts['utterance'].replace('--', ' ')

    # get indices of non-empty or non-null strings and keep these
    utterance_idxs = transcripts[(transcripts['utterance'].notnull()) & (transcripts['utterance'] != '')].index
    transcripts = transcripts.iloc[utterance_idxs]
    print(f'number of cleaned up utterances: {len(transcripts)}')
    transcripts.to_csv('data/frame_utterance_pairs_clean.csv', index=False)
    

class WordDictionary(object):
    def __init__(self):
        self.path = 'data/frame_utterance_pairs_clean.csv'
        self.transcripts = pd.read_csv(self.path)
        self.utterances = self.transcripts['utterance']

        # create vocab
        self.word2index = {}
        self.index2word = []

        # add tokens
        self.add_word('<pad>')
        self.add_word('<unk>')
        self.add_word('<sos>')
        self.add_word('<eos>')

        # build rest of vocab
        self.build_vocab()

    def build_vocab(self):
        # add words to the dictionary
        for utterance in self.utterances:
            words = utterance.split(' ')
            for word in words:
                self.add_word(word)
        
    def add_word(self, word):
        if word not in self.word2index:
            self.index2word.append(word)
            self.word2index[word] = len(self.index2word) - 1
        return self.word2index[word]

    def __len__(self):
        return len(self.index2word)

    
class SAYCamTrainDataset(Dataset):
    # train dataset class
    def __init__(self):
        self.transcripts = pd.read_csv('data/frame_utterance_pairs_clean.csv')
        self.base_dir = '/home/wv9/code/WaiKeen/multimodal-baby/data/'
        self.train_dir = os.path.join(self.base_dir, 'train')

        # image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # build vocab
        self.vocab = WordDictionary()

    def __getitem__(self, i):
        # get caption and preprocess
        utterance = self.transcripts['utterance'].iloc[i]
            
        utterance_words = utterance.split(' ')
        utterance_length = len(utterance_words)
        utterance_idxs = []
        for word in utterance_words:
            try:
                utterance_idxs.append(self.vocab.word2index[word])
            except KeyError:
                utterance_idxs.append(self.vocab.word2index['<unk>'])

        # get image and transform
        img_filename = os.path.join(self.train_dir, self.transcripts['frame_filename'].iloc[i])
        img = np.array(Image.open(img_filename).convert('RGB'))
        img = self.transform(img)
                
        return img, utterance_idxs, utterance_length

    def __len__(self):
        return len(self.transcripts)
    
    # def _build_vocab(self):
    #     # builds vocabulary from utterances extracted from SAYCam
    #     if os.path.exists('data/vocab.pickle'):
    #         with open('data/vocab.pickle', 'rb') as f:
    #             vocab = pickle.load(f)
    #             self.word2index = vocab['word2index']
    #             self.num_words = vocab['num_words']
    #             self.max_len = vocab['max_len']
    #     else:
    #         self.word2index = {'<pad>': 0, '<unk>': 1}
    #         self.num_words = 2
    #         self.max_len = 0

    #         utterances = self.transcripts['utterance']
     
    #         print('building vocab')
    #         for utterance in tqdm(utterances):
    #             if isinstance(utterance, str):
    #                 utterance_words = utterance.split(' ')
          
    #                 # check if utterance length is larger than current max
    #                 if len(utterance_words) > self.max_len:
    #                     self.max_len = len(utterance_words)
       
    #                 # add words to vocab
    #                 for word in utterance_words:
    #                     if word not in self.word2index:
    #                         self.word2index[word] = self.num_words
    #                         self.num_words += 1
     
    #         # save vocab
    #         vocab = {'word2index': self.word2index, 'num_words': self.num_words, 'max_len': self.max_len}
    #         with open('data/vocab.pickle', 'wb') as f:
    #             pickle.dump(vocab, f)

class SAYCamEvalDataset(Dataset):
    # val dataset class
    def __init__(self, eval_type='val'):
        self.eval_type = eval_type

        # determine whether to eval using the generated validation or test set
        if self.eval_type == 'val':
            self.eval_filename = 'data/validation.csv'
            self.eval_dataset = pd.read_csv(self.eval_filename)
        elif self.eval_type == 'test':
            self.eval_filename = 'data/test.csv'
            self.eval_dataset = pd.read_csv(self.eval_filename)

        # read in vocab from training
        self.vocab = WordDictionary()
        # with open('data/vocab.pickle', 'rb') as f:
        #     self.vocab = pickle.load(f)
        #     self.word2index = self.vocab['word2index']
        #     self.index2word = {v: k for k, v in self.word2index.items()}

        # image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
            
    def __getitem__(self, i):
        # read in category labels and map to indices from vocab
        target_category = self.eval_dataset['target_category'].iloc[i]
        foil_category_one = self.eval_dataset['foil_category_one'].iloc[i]
        foil_category_two = self.eval_dataset['foil_category_two'].iloc[i]
        foil_category_three = self.eval_dataset['foil_category_three'].iloc[i]

        target_category_idx = self.vocab.word2index[target_category]
        foil_category_one_idx = self.vocab.word2index[foil_category_one]
        foil_category_two_idx = self.vocab.word2index[foil_category_two]
        foil_category_three_idx = self.vocab.word2index[foil_category_three]

        # convert labels as longtensor 
        labels = torch.LongTensor([target_category_idx])

        # read in target and foil imgs and assign to array
        imgs = np.zeros((4, 3, 224, 224))
        target_img_filename = self.eval_dataset['target_img_filename'].iloc[i]
        foil_one_img_filename = self.eval_dataset['foil_one_img_filename'].iloc[i]
        foil_two_img_filename = self.eval_dataset['foil_two_img_filename'].iloc[i]
        foil_three_img_filename = self.eval_dataset['foil_three_img_filename'].iloc[i]

        target_img = np.array(Image.open(target_img_filename).convert('RGB'))
        target_img = self.transform(target_img)
        imgs[0] = target_img

        foil_one_img = np.array(Image.open(foil_one_img_filename).convert('RGB'))
        foil_one_img = self.transform(foil_one_img)
        imgs[1] = foil_one_img
        
        foil_two_img = np.array(Image.open(foil_two_img_filename).convert('RGB'))
        foil_two_img = self.transform(foil_two_img)
        imgs[2] = foil_two_img

        foil_three_img = np.array(Image.open(foil_three_img_filename).convert('RGB'))
        foil_three_img = self.transform(foil_three_img)
        imgs[3] = foil_three_img

        # convert imgs to float tensor
        imgs = torch.FloatTensor(imgs)
        return imgs, labels

    def __len__(self):
        return len(self.eval_dataset)

    
if __name__ == "__main__":
    generate_val_dataset()
    generate_test_dataset()
