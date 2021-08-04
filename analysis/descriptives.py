import json
import pandas as pd
import matplotlib.pyplot as plt

# read in train metadata
with open('../data/train_5fps.json') as f:
    data = json.load(f)

# convert to data frame
df = pd.DataFrame(data['images'])

# get some descriptives
print(f'number of utterances: {len(df)}')
print(f'number of frames: {df["num_frames"].sum()}')

# read in vocab
with open('../data/vocab_5fps.json') as f:
    vocab = json.load(f)

# calculate vocab size
print(f'size of vocabulary: {len(vocab)}')

# calculate average utterance length
df = df.assign(utterance_len = lambda df: df['utterance'].map(lambda utterance: len(utterance.split(' '))))
print(f'average utterance length: {df["utterance_len"].mean()}')

# plots
# histogram of frames per utterance
plt.hist(df['num_frames'], bins=32)
plt.show()

# histogram of utterance lengths
utterance_len = [i for i in df['utterance_len'] if i <= 16]
plt.hist(utterance_len, bins=16)
plt.show()
