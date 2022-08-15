import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from multimodal.multimodal_data_module import PAD_TOKEN_ID, UNK_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID


class SentencesDataset(Dataset):
    """Dataset for sentences.
    """

    def __init__(self, data, vocab):
        """
        Inputs:
            data: list of spacy-parsed doc for each sentences
            vocab: vocab
        """

        super().__init__()
        self.data = data
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        """Returns a sentence in tuple
        (sentence_idxs, sentence_length, raw_sentence_data)
        """

        raw_sentence_data = self.data[idx]
        sentence_idxs = [
            self.vocab.get(str(token), UNK_TOKEN_ID)
            for token in raw_sentence_data
        ]
        sentence_idxs = [SOS_TOKEN_ID] + sentence_idxs + [EOS_TOKEN_ID]
        sentence_length = len(sentence_idxs)
        sentence_idxs = torch.tensor(sentence_idxs, dtype=torch.long)

        return sentence_idxs, sentence_length, raw_sentence_data


def collate_fn(batch):
    sentence_idxs, sentence_length, raw_sentence_data = zip(*batch)
    sentence_idxs = pad_sequence(
        sentence_idxs, batch_first=True, padding_value=PAD_TOKEN_ID)
    sentence_length = torch.tensor(sentence_length, dtype=torch.long)
    return sentence_idxs, sentence_length, raw_sentence_data
