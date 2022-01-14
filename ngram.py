from collections import Counter, defaultdict
import torch
from math import log

class NGramModel:
    """
    A simple n-gram langauge model.
    """

    def __init__(self, N, vocab_size):
        assert N >= 1, "N-gram model requires N >= 1"
        self._N = N
        self._count = [defaultdict(lambda: [0, Counter()]) for n in range(self._N)]
        self._vocab_size = vocab_size

    @property
    def N(self):
        return self._N

    def update(self, y, y_len):
        """
        Update the counts with sequences.
        Assert all sequences start with SOS token.
        To update with the whole dataset, simply feed all y, y_len from dataloader into this function.
        y: torch.Tensor of shape (batch_size, input_length), batch of sequence token ids.
        y_len: torch.Tensor of shape (batch_size,), lengths of these sequences.
        """
        for seq, seq_len in zip(y, y_len):
            seq = tuple(seq[:seq_len].tolist())

            for n in range(self._N):
                count = self._count[n]
                for i in range(max(1, n), seq_len):
                    cnt_data = count[seq[i - n : i]]
                    cnt_data[0] += 1
                    cnt_data[1][seq[i]] += 1

    def calculate_ce_loss(self, y, y_len, alpha=0.1, tokenwise=True):
        """
        Calculate the cross-entropy losses.
        Same as the update method, assert all sequences start with SOS token.
        y: torch.Tensor of shape (batch_size, input_length), batch of sequence token ids.
        y_len: torch.Tensor of shape (batch_size,), lengths of these sequences.
        alpha: portion of the probability allocated for backoff.
        output: cross-entropy losses
        loss: torch.Tensor of shape (batch_size, input_length - 1).
        """
        loss = torch.zeros_like(y[:, 1:], dtype=torch.float)
        n_tokens = 0

        log_alpha = log(alpha)
        log_1_minus_alpha = log(1 - alpha)

        for batch_i, (seq, seq_len) in enumerate(zip(y, y_len)):
            seq = tuple(seq[:seq_len].tolist())

            for i in range(1, seq_len):
                token_loss = 0.
                for n in range(min(self._N - 1, i), -1, -1):
                    count = self._count[n]
                    if seq[i - n : i] in count:
                        cnt_data = count[seq[i - n : i]]
                        if n == 0:
                            token_loss += log(cnt_data[1].get(seq[i], 0) + 1) - log(cnt_data[0] + self._vocab_size)
                            break
                        elif seq[i] in cnt_data[1]:
                            token_loss += log(cnt_data[1][seq[i]]) - log(cnt_data[0]) + log_1_minus_alpha
                            break
                    token_loss += log_alpha
                else:
                    raise Exception("Even unigram is not applicable.")
                token_loss = - token_loss  # negative log likelihood
                loss[batch_i, i - 1] = token_loss
                n_tokens += 1

        if not tokenwise:
            loss = loss.sum() / n_tokens

        return loss
