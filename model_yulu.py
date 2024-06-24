import torch.nn as nn
from transformers.models.roberta import RobertaForMaskedLM
from transformers import GPT2LMHeadModel
import torch
import torch.nn.functional as F
from typing import Dict


class GPT2Model(nn.Module):
    def __init__(self, config, pad_id):
        super().__init__()
        self.model = GPT2LMHeadModel(config=config)
        self.pad_id = pad_id
    def forward(self, inputs, masked_labels, labels, reduction='sum'):
        loss, logits = self._calculate_loss(self.model, inputs['input_ids'].to('cuda'), inputs['attention_mask'].to('cuda'), reduction=reduction)
        return loss, logits
    def _calculate_loss(self, model, ids, att_masks, reduction='sum'):
        output = model(input_ids=ids, attention_mask=att_masks) # logits shape: (batch_size, max_sent_length, vocab_size) 
        logits = output.logits[:, :-1]
        logits = logits.permute(0, 2, 1)
        targets = ids[:, 1:]
        loss = F.cross_entropy(logits, targets, reduction=reduction, ignore_index=self.pad_id)
        return loss, output.logits
    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)


class RobertaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = RobertaForMaskedLM(config=config)
    def forward(self, input_ids, attention_mask, masked_labels, reduction='sum'):
        loss, logits = self._calculate_loss(self.model, attention_mask, input_ids, masked_labels, reduction=reduction)
        return loss, logits

    def _calculate_loss(self, model,
                mask_matrix: torch.bool,  # mask_matrix is 2D bool array specifying which tokens to predict
                x: Dict[str, torch.tensor],
                y: torch.tensor,
                reduction) -> torch.tensor:
        output = self.model(**{k: v.to('cuda') for k, v in x.items() if torch.is_tensor(v)})

        logits_3d = output['logits']
        logits_2d = logits_3d.view(-1, model.config.vocab_size)
        bool_1d = mask_matrix.view(-1)
        logits_for_masked_words = logits_2d[bool_1d]
        labels = y.cuda()
        loss = F.cross_entropy(logits_for_masked_words,  # [num masks in batch, vocab size]
                        labels, reduction=reduction)  # [num masks in batch]
        return loss, logits_for_masked_words
    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)


# 2. Model Definition
class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, dropout, dim=1):
        if not (self.training and dropout):
            return x
        return x.new_empty(x.shape[:dim] + (1,) + x.shape[dim+1:]).bernoulli_(1 - dropout) / (1 - dropout) * x


class LSTMModel(nn.Module):
    def __init__(self, config, pad_id):
        super(LSTMModel, self).__init__()
        self.pad_id = pad_id
        self.config = config
        self.encoder = nn.Embedding(self.config.vocab_size,self.config.embedding_dim)
        self.lstm = nn.LSTM(self.config.embedding_dim, self.config.hidden_dim, self.config.num_layers, batch_first=True)
        self.decoder = nn.Linear(self.config.hidden_dim, self.config.vocab_size)
        self.lockdrop = LockedDropout()
        self.output_dropout = nn.Dropout(self.config.dropout_rate)
        self.encoder.weight = self.decoder.weight # tie embeddings

    def forward(self, inputs, masked_labels, labels, reduction="sum"):
        embed = self.encoder(inputs['input_ids'].to('cuda'))
        embed = embed.squeeze(-1)
        lstm_out, _ = self.lstm(embed)
        dropout_output = self.lockdrop(lstm_out, self.config.dropout_rate)
        output = self.decoder(dropout_output)
        loss = self._calculate_loss(output, inputs['input_ids'], reduction=reduction)
        return loss, output
    # dimension可能需要调整

    def _calculate_loss(self, output, batch_idx, reduction="sum"):
        batch_idxs = batch_idx.to('cuda')  # Move input data to GPU
        target = batch_idxs[:, 1:].to('cuda')  # Move target data to GPU
        output =output[:, :-1]
        output = output.permute(0, 2, 1)
        loss = F.cross_entropy(
            output, target,
            ignore_index=self.pad_id,  # ignore pad tokens
            reduction=reduction,
        )
        return loss
