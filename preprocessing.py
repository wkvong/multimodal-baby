# preprocessing of SAYcam transcripts

import glob
import collections
import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt

transcripts = glob.glob('data/transcripts/*.csv')
eval_categories = {'ball': 0, 'basket': 0, 'car': 0, 'carseat': 0,
                   'cat': 0, 'chair': 0, 'computer': 0, 'couch': 0,
                   'crib': 0, 'door': 0, 'floor': 0, 'foot': 0,
                   'greenery': 0, 'ground': 0, 'hand': 0, 'kitchen': 0,
                   'paper': 0, 'plushanimal': 0, 'puzzle': 0, 'road': 0,
                   'room': 0, 'sand': 0, 'stairs': 0, 'table': 0,
                   'toy': 0, 'window': 0}

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

stop_words = []
word_dict = {}
utterance_lengths = []
max_utterance_length = 0
n_utterances = 0

for transcript_fn in transcripts:
    print(transcript_fn)
    df = pd.read_csv(transcript_fn)
    utterances = df['Utterance']
    speakers = df['Speaker']
    for utterance, speaker in zip(utterances, speakers):
        if speaker not in ['s', 'S', 'S ', 'sam', 'Sam', 'Sam ']:
            if isinstance(utterance, str):
                n_utterances += 1
                
                words = utterance.lower().split(' ')
                utterance_lengths.append(len(words))

                if len(words) > max_utterance_length:
                    # print(len(words))
                    # print(words)
                    max_utterance_length = len(words)
                
                for word in words:
                    # remove any punctuation
                    word = word.replace(',', '').replace('.', '').replace('?', '').replace('*', '').replace('!', '').replace('-', '').replace(';', '').replace('\n', '').replace('\\', '').replace('(', '').replace(')', '').replace('’', '\'').replace('\"', '')
     
                    if word in word_dict.keys():
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1

print(n_utterances)

counter = collections.Counter(utterance_lengths)
print(counter.keys())
print(counter.values())
utterance_length = [x[0] for x in sorted(counter.items())]
utterance_count = [x[1] for x in sorted(counter.items())]

plt.bar(utterance_length, utterance_count)
plt.xlabel('utterance length')
plt.ylabel('frequency')
plt.show()

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
