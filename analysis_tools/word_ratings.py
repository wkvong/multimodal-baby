from pathlib import Path
import pandas as pd


# modified from https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
from nltk.corpus import wordnet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('JJ'):
        return wordnet.ADJ
    elif treebank_tag.startswith('VB'):
        return wordnet.VERB
    elif treebank_tag.startswith('NN'):
        return wordnet.NOUN
    elif treebank_tag.startswith('RB'):
        return wordnet.ADV
    else:
        return ''

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize(word, pos):
    wordnet_pos = get_wordnet_pos(pos)
    if not wordnet_pos:
        wordnet_pos = wordnet.NOUN  # default
    return lemmatizer.lemmatize(word, pos=wordnet_pos)


class WordRatings:
    """Maintaining ratings of words."""
    def __init__(self, excel_path, word_field='Word', **kwargs):
        """Read data from excel_path."""
        self.data = pd.read_excel(excel_path, **kwargs)
        self.word_field = word_field

        self.word2index = {
            row[self.word_field].lower(): index
            for index, row in self.data.iterrows() if isinstance(row[self.word_field], str)
        }

    def word_to_index(self, word, pos):
        try:
            return self.word2index[word]
        except KeyError:
            word = lemmatize(word, pos)
            word = {
                "kitty": "cat",
                "doggy": "dog",
            }.get(word, word)
            try:
                return self.word2index[word]
            except KeyError:
                return None

    def extend_items(self, items, word_label, idx2word):
        mapped_index = items.index.map(lambda key: self.word_to_index(idx2word[key[0]], key[1]))
        available = mapped_index.notna()
        available_index = items.index[available]
        available_mapped_index = mapped_index[available]
        d = self.data.loc[available_mapped_index]
        d.rename(lambda column: word_label if column == self.word_field else column)
        d.index = available_index
        items[d.columns] = d


data_path = [
    Path("/misc/vlgscratch4/LakeGroup/shared_data"),
    Path("/scratch/ww2135/shared_data"),
][1]
concreteness_data = WordRatings(data_path/"Concreteness ratings Brysbaert2014.xlsx")
norm_data = WordRatings(data_path/"VanArsdall_Blunt_NormData.xlsx", sheet_name=1)

conc_field = 'Conc.M'
