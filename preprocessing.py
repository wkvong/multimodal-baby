# preprocessing of SAYcam transcripts

import glob
from tqdm import tqdm
import collections
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *

def combine_transcripts():
    "combines all the transcripts into a single data frame"
    transcript_filenames = sorted(glob.glob('data/transcripts/*.csv'))
    transcripts = [pd.read_csv(transcript) for transcript in transcript_filenames]
    combined_transcript = pd.concat(transcripts, sort=False)
    return combined_transcript


if __name__ == "__main__":
    # 1. combine transcripts
    combined_transcript = combine_transcripts()
    print(combined_transcript.columns)

    # 2. filter out inaudible utterances and words inside of parentheses
    # for asterisks: re.sub(r'\*[^)]*\*', '', utterance)
    # for parentheses: re.sub(r'\([^)]*\)', '', utterance)
    # also want to call lower() and strip() here
    combined_transcript['Utterance'] = combined_transcript['Utterance'].str.replace(r'\*[^)]*\*', '')
    combined_transcript['Utterance'] = combined_transcript['Utterance'].str.replace(r'\([^)]*\)', '')
    combined_transcript['Utterance'] = combined_transcript['Utterance'].str.replace(r'\[[^)]*\]', '')
    combined_transcript['Utterance'] = combined_transcript['Utterance'].str.strip()
    combined_transcript['Utterance'] = combined_transcript['Utterance'].str.lower()

    # 3. split utterances?
    # saving this code for later on!
    # for utterance in combined_transcript['Utterance']:
    #     if isinstance(utterance, str):
    #         # split utterance based on punctuation
    #         split_utterance = msplit(utterance, (".", "?", "!"))

    #         # clean up split utterances, remove commas and whitespace etc.
    #         split_utterance = [utterance.strip().replace(',', '') for utterance in split_utterance if len(utterance) > 0]

    # 4. filter based on parent
    parent_speakers = ['M', 'Mom', 'mom', 'm', 'mother', 'Mother', 'papa', 'the mom']
    combined_transcript = combined_transcript[combined_transcript['Speaker'].isin(parent_speakers)]
    
    # 5. save dataframe
    combined_transcript.to_csv('data/combined_transcript.csv')

    # TODO: all descriptives should be moved to a separate file!
    
    # transcripts = glob.glob('data/transcripts/*.csv')
    # eval_categories = {'ball': 0, 'basket': 0, 'car': 0, 'carseat': 0,
    #                    'cat': 0, 'chair': 0, 'computer': 0, 'couch': 0,
    #                    'crib': 0, 'door': 0, 'floor': 0, 'foot': 0,
    #                    'greenery': 0, 'ground': 0, 'hand': 0, 'kitchen': 0,
    #                    'paper': 0, 'plushanimal': 0, 'puzzle': 0, 'road': 0,
    #                    'room': 0, 'sand': 0, 'stairs': 0, 'table': 0,
    #                    'toy': 0, 'window': 0}
     
    # # get counts for eval categories
    # for transcript_fn in transcripts:
    #     print(transcript_fn)
    #     df = pd.read_csv(transcript_fn)
    #     utterances = df['Utterance']
    #     speakers = df['Speaker']
    #     for utterance, speaker in zip(utterances, speakers):
    #         if isinstance(utterance, str):
    #             for category in eval_categories.keys():
    #                 if category in utterance.lower():
    #                     eval_categories[category] += 1
     
    # print(eval_categories)
     
    # get mapping information and convert it to a dictionary
    # mappings = np.loadtxt('data/mapping.txt', dtype='str', delimiter=',')
    # mapping_dict = {}
    # for mapping in mappings:
    #     mapping_dict[mapping[0].strip()] = mapping[1].strip()
        
    # speaker_list = []
    # stop_words = []
    # word_dict = {}
    # utterance_lengths = []
    # max_utterance_length = 0
    # n_utterances = 0
    # cooccurrence_dict = {}
     
    # for transcript_fn in transcripts:
    #     df = pd.read_csv(transcript_fn)
    #     utterances = df['Utterance']
    #     speakers = df['Speaker']
    #     objects = df['Object Being Looked At']
    #     for utterance, speaker, object_list in zip(utterances, speakers, objects):
    #         if isinstance(utterance, str):
    #             if 'inaudible' in utterance:
    #                 print(utterance)
            
    #         # get objects in the current scene
    #         if isinstance(object_list, str):
    #             # extract all objects as a list
    #             object_list = [x.strip().lower() for x in msplit(object_list, (',', ':', ';')) if len(x) > 1]
     
    #             # check if any objects map to known categories from emin's mapping list
    #             # if they do not exist in the object list, append them
    #             for obj in object_list:
    #                 if obj in mapping_dict:
    #                     if mapping_dict[obj] not in object_list:
    #                         object_list.append(mapping_dict[obj])
     
    #             # now, check if any of the objects are mentioned in the utterance    
    #             for obj in object_list:
    #                 if isinstance(utterance, str) and obj in utterance.lower():
    #                     if obj in cooccurrence_dict.keys():
    #                         cooccurrence_dict[obj] += 1
    #                     else:
    #                         cooccurrence_dict[obj] = 1
                        
    #                     # print(f'object list: {object_list}')
    #                     # print(f'object: {obj}')
    #                     # print(f'utterance: {utterance}')
    #                     # print('\n')
                    
    #         speaker_list.append(speaker)
    #         if speaker not in ['s', 'S', 'S ', 'sam', 'Sam', 'Sam ']:
    #             if isinstance(utterance, str):
    #                 n_utterances += 1
                    
    #                 words = utterance.lower().split(' ')
    #                 utterance_lengths.append(len(words))
     
    #                 if len(words) > max_utterance_length:
    #                     # print(len(words))
    #                     # print(words)
    #                     max_utterance_length = len(words)
                    
    #                 for word in words:
    #                     # remove any punctuation
    #                     word = word.replace(',', '').replace('.', '').replace('?', '').replace('*', '').replace('!', '').replace('-', '').replace(';', '').replace('\n', '').replace('\\', '').replace('(', '').replace(')', '').replace('’', '\'').replace('\"', '')
         
    #                     if word in word_dict.keys():
    #                         word_dict[word] += 1
    #                     else:
    #                         word_dict[word] = 1
     
    # sorted_dict = {k: v for k, v in sorted(cooccurrence_dict.items(), key=lambda item: item[1], reverse=True)}
     
    # save speaker counts to csv
    # speaker_counter = collections.Counter(speaker_list)
    # speaker_df = pd.DataFrame.from_dict(speaker_counter, orient='index').reset_index()
    # speaker_df = speaker_df.sort_values(0, ascending=False)
    # speaker_df.to_csv('data/speakers.csv', index=False)
     
    # counter = collections.Counter(utterance_lengths)
    # print(counter.keys())
    # print(counter.values())
    # utterance_length = [x[0] for x in sorted(counter.items())]
    # utterance_count = [x[1] for x in sorted(counter.items())]
     
    # plt.bar(utterance_length, utterance_count)
    # plt.xlabel('utterance length')
    # plt.ylabel('frequency')
    # plt.show()
     
    # eng_stopwords = stopwords.words('english')
    # # updated_dict = {k: v for k, v in word_dict.items() if k not in eng_stopwords}
    # sorted_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1])}
    # sorted_words = list(sorted_dict.keys())[::-1][:100]
    # sorted_counts = list(sorted_dict.values())[::-1][:100]
     
    # plt.bar(sorted_words, sorted_counts)
    # plt.xticks(rotation=90)
    # plt.ylabel('frequency')
    # plt.show()
     
    # utterance_lengths = [x for x in utterance_lengths if x <= 50]
     
    # plt.hist(utterance_lengths, bins=50)
    # plt.ylabel('frequency')
    # plt.show()
