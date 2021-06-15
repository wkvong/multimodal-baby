import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import wordnet as wn

# create dataframe looking at dates of transcribed vs. untranscribed videos
transcripts = pd.read_csv('data/frame_utterance_pairs_clean.csv')
transcribed_files = transcripts['annotations_filename'].unique()
# all_files = sorted(os.listdir('/misc/vlgscratch4/LakeGroup/shared_data/S_videos_annotations/annotations/S'))

# transcribed = []
# date = []
# for filename in all_files:
#     date = filename.split('_')[1]

#     if filename in transcribed_files:
#         transcribed.append(True)
#     else:
#         transcribed.append(False)

# transcribed_dates = pd.DataFrame({'filename': all_files,
#                                   'date': date,
#                                   'transcribed': transcribed})

# get vocabulary size and mean utterance length
utterances = transcripts['utterance']
utterance_words = []
utterance_lengths = []
for utterance in utterances:
    words = utterance.split(' ')
    for word in words:
        utterance_words.append(word)
    
    utterance_length = len(words)
    utterance_lengths.append(utterance_length)

# get number of words in the vocabulary    
print('number of words in vocabulary:', len(set(utterance_words)))

# get mean utterance length
print('mean utterance length:', np.mean(np.array(utterance_lengths)))

# get utterance frequency
utterance_counter = Counter(utterance_lengths)
print(utterance_counter.most_common(len(utterance_counter)))

# plot of mean utterance lengths
utterance_lengths = [x for x in utterance_lengths if x <= 50]
print('mean utterance length:', np.mean(np.array(utterance_lengths)))

# plt.hist(utterance_lengths, bins=50)
# plt.ylabel('Frequency')
# plt.title('Distribution of utterance lengths')
# plt.savefig('figures/utterance_lengths.png')

# plot of word frequency
word_counter = Counter(utterance_words)
most_common_words_counter = word_counter.most_common(100)
most_common_words = [word[0] for word in most_common_words_counter]
most_common_frequencies = [word[1] for word in most_common_words_counter]

# plt.figure(figsize=(20, 10))
# plt.barh(most_common_words[::-1], most_common_frequencies[::-1])
# plt.xscale('log')
# plt.xlim(10, 10000)
# plt.xlabel('Log Frequency')
# plt.title('Frequency of Common Words')
# plt.tight_layout()
# plt.show()
# plt.savefig('figures/frequent_words.png')

# plots of noun frequency
# for key, counts, in list(word_counter.items()):
#     tmp = wn.synsets(key)
#     if not tmp:
#         del word_counter[key]
#     elif tmp[0].pos() != 'n':
#         del word_counter[key]

# most_common_nouns_counter = word_counter.most_common(100)
# most_common_nouns = [word[0] for word in most_common_nouns_counter]
# most_common_frequencies = [word[1] for word in most_common_nouns_counter]

# plt.figure(figsize=(20, 10))
# plt.barh(most_common_nouns[::-1], most_common_frequencies[::-1])
# plt.xscale('log')
# plt.xlim(10, 10000)
# plt.xlabel('Log Frequency')
# plt.title('Frequency of Common Nouns')
# plt.tight_layout()
# plt.show()
        
# plot of most common nouns
# categories = ["toy", "ball", "door", "car", "carseat", "cat", "table",
#               "hand", "paper", "room", "chair", "basket", "kitchen",
#               "floor", "road", "ground", "crib", "computer", "stairs",
#               "couch", "foot", "window", "sand", "puzzle"]

# for key, counts, in list(word_counter.items()):
#     if key not in categories:
#         del word_counter[key]

# most_common_categories_counter = word_counter.most_common(24)
# most_common_categories = [word[0] for word in most_common_categories_counter]
# most_common_frequencies = [word[1] for word in most_common_categories_counter]

# for i in range(len(most_common_categories)):
#     print(f'{most_common_categories[i]}: {most_common_frequencies[i]}')

# plt.figure(figsize=(20, 10))
# plt.barh(most_common_categories[::-1], most_common_frequencies[::-1])
# # plt.xscale('log')
# plt.xlabel('Frequency')
# plt.title('Frequency of Common Nouns')
# plt.tight_layout()
# plt.show()

print(word_counter["kitty"])
