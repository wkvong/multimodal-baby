from pathlib import Path
from collections import namedtuple, defaultdict, Counter
import itertools
import functools
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ngram import NGramModel
from multimodal.utils import map_structure
from .token_items_data import token_field, Key
from .sumdata import *
from .utils import get_model_device, identity


def build_ngram_model(N, vocab_size, train_dataloader):
    ngram_model = NGramModel(N, vocab_size)

    for x, y, y_len, raw_y in tqdm(train_dataloader):
        ngram_model.update(y, y_len)

    return ngram_model


def examples_from_batches(batches):
    return itertools.chain.from_iterable((zip(*batch) for batch in batches))


def raw_utterances_from_dataloader(dataloader):
    for x, y, y_len, raw_y in dataloader:
        for raw_y_ in raw_y:
            yield raw_y_[0]


def get_pos_tags(dataloader, dataset_name, split, toolkit='stanza'):
    cache_path = Path('dataset_cache') / dataset_name / f'{split}.pos.cache'
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f'load cached pos tags: {cache_path}')
        return torch.load(cache_path)

    if toolkit == 'stanza':
        import stanza
        nlp = stanza.Pipeline(lang='en', processors='tokenize,pos', tokenize_pretokenized=True)

    elif toolkit == 'spacy':
        import spacy
        nlp = spacy.load("en_core_web_trf", exclude=["parser", "attribute_ruler", "lemmatizer", "ner"])

        def tokenizer(text):
            words = text.split()
            spaces = [True] * len(words)
            if len(spaces) >= 1:
                spaces[-1] = False
            return spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces)

        nlp.tokenizer = tokenizer

    else:
        raise Exception(f"Unknown toolkit {toolkit}")

    pos_tags = []

    def wrap_utterance_pos_tags(utterance_pos_tags):
        return (
            ['.']  # SOS
            + utterance_pos_tags
            + ['.']  # EOS
        )

    if toolkit == 'stanza':
        for x, y, y_len, raw_y in tqdm(dataloader):
            utterance_words_batch = []
            clean_idxes = []
            clean_utterance_words_batch = []

            for raw_y_ in raw_y:
                utterance_words = raw_y_[0].split()
                utterance_words_batch.append(utterance_words)
                # remove empty tagging_words
                if utterance_words:
                    clean_idxes.append(len(clean_utterance_words_batch))
                    clean_utterance_words_batch.append(utterance_words)
                else:
                    clean_idxes.append(-1)

            # pos tagging
            doc = nlp(clean_utterance_words_batch)

            for y_, y_len_, utterance_words, clean_idx in zip(y, y_len, utterance_words_batch, clean_idxes):
                utterance_pos_tags = wrap_utterance_pos_tags(
                    ([token.words[0].xpos for token in doc.sentences[clean_idx].tokens] if clean_idx >= 0 else [])
                )
                pos_tags.append(utterance_pos_tags)

    elif toolkit == 'spacy':
        for doc in nlp.pipe(raw_utterances_from_dataloader(tqdm(dataloader))):
            utterance_pos_tags = wrap_utterance_pos_tags(
                [token.tag_ for token in doc]
            )
            pos_tags.append(utterance_pos_tags)

    torch.save(pos_tags, cache_path)

    return pos_tags


def get_word_pos_cnt(dataloader, pos_tags):
    from multimodal.multimodal_data_module import SOS_TOKEN, EOS_TOKEN
    raw_utterances = raw_utterances_from_dataloader(dataloader)
    word_pos_cnt = Counter()
    for raw_utterance, utterance_pos_tags in zip(raw_utterances, pos_tags):
        tokens = [SOS_TOKEN] + raw_utterance.split() + [EOS_TOKEN]
        word_pos_cnt.update(zip(tokens, utterance_pos_tags))
    return word_pos_cnt


def get_word_pos_stat_from_word_pos_cnt(word_pos_cnt):
    word_pos_stat = defaultdict(Counter)
    for (word, pos), cnt in word_pos_cnt.items():
        word_pos_stat[word][pos] = cnt
    return word_pos_stat


def get_pos_stats_for_words(words, word_pos_stat, pos_mapping=identity):
    """Get POS for words.
    Inputs:
        words: list or 1-d np.ndarray of words.
        word_pos_stat: obtained from get_word_pos_stat_from_word_pos_cnt.
        pos_mapping: callable to map pos in word_pos_stat.
    Output:
        pos_stats: list of sorted list of (pos, cnt) pairs.
    """

    pos_stats = []
    for i, word in enumerate(words):
        pos_stat = word_pos_stat.get(word, {})
        # map pos
        new_pos_stat = Counter()
        for pos, cnt in pos_stat.items():
            new_pos_stat[pos_mapping(pos)] += cnt
        # sort by frequency
        new_pos_stat = sorted(
            new_pos_stat.items(),
            key=lambda item: (-item[1], item[0])
        )
        pos_stats.append(new_pos_stat)
    return pos_stats


def is_regressional(model):
    """Whether the model is regressional, so the predicted loss, logits, labels are shifted.
    """
    return model is not None and (isinstance(model, NGramModel) or model.language_model.text_encoder.regressional)


def run_model(
    model, y, y_len, x=None,
    image_features=None,
    image_feature_map=None,
    single_example=False,
    return_all=False
):
    if model is None or isinstance(model, NGramModel):
        hidden_dim = 0
        device = y.device
    else:
        hidden_dim = model.language_model.text_encoder.hidden_dim
        device = get_model_device(model)

    batch = ((x, y, y_len) if x is not None else (y, y_len))
    if single_example:
        batch = map_structure(lambda t: t.unsqueeze(0), batch)
    batch = map_structure(lambda t: t.to(device=device), batch)
    if x is not None:
        (x, y, y_len) = batch
    else:
        (y, y_len) = batch

    if model is None:
        loss = torch.zeros_like(y, dtype=torch.float, device=device)
        outputs = torch.zeros(*(y.shape + (hidden_dim,)), dtype=torch.float, device=device)
    elif isinstance(model, NGramModel):
        loss = model.calculate_ce_loss(y, y_len, tokenwise=True)
        outputs = torch.zeros(*(y.shape + (hidden_dim,)), dtype=torch.float, device=device)
    else:
        loss, outputs, logits, attns, labels = model.calculate_ce_loss(
            y, y_len, x=x,
            image_features=image_features,
            image_feature_map=image_feature_map,
            tokenwise=True
        )
    if is_regressional(model):
        # pad loss with preceeding 0
        loss = F.pad(loss, (1, 0))

    ret = outputs, loss
    if return_all:
        ret = ret + (logits, attns, labels)

    if single_example:
        ret = map_structure(lambda t: t.squeeze(0) if t is not None else None, ret)

    return ret


def run_model_on_batches(model, batches, return_all=False):
    """Run model on batches and yield batches with model outputs appended.
    """
    with torch.no_grad():
        for batch in batches:
            if isinstance(batch, tuple):
                x, y, y_len, raw_y = batch
                kwargs = {'x': x}
            elif isinstance(batch, dict):
                y = batch['y']
                y_len = batch['y_len']
                kwargs = batch.copy()
                del kwargs['y']
                del kwargs['y_len']
            else:
                raise Exception(f"Unable to process batch: {batch}")
            batch_size = len(y)
            ret = run_model(model, y, y_len, return_all=return_all, **kwargs)
            ret = tuple(t if t is not None else [None] * batch_size for t in ret)
            yield batch, *ret


def run_model_on_data(*args, **kwargs):
    """Run model on batches and yield examples with model outputs appended.
    """
    return examples_from_batches((
        (*batch, *ret)
        for batch, *ret in run_model_on_batches(*args, **kwargs)
    ))


def get_model_losses_on_batches(model, batches):
    """Run model on batches and return a tensor containing all losses.
    """
    losses = []
    for batch, outputs, loss in run_model_on_batches(model, batches):
        loss = loss.sum(-1)
        losses.append(loss.detach())
    losses = torch.cat(losses, 0)
    return losses


def get_token_items(token_pos_items):
    token_pos_items = list(token_pos_items.items())
    token_pos_items.sort()

    token_items = {}

    i, j = 0, 0
    while i < len(token_pos_items):
        token_id = token_pos_items[i][0].token_id

        while j < len(token_pos_items) and token_pos_items[j][0].token_id == token_id:
            j += 1

        items = token_pos_items[i : j]

        # set the POS tag of a word as the most frequent POS tag; if there're multiple most frequent POS tags, select the smallest POS tag string
        key = max(items, key=lambda item: (item[1].cnt.item(), item[0].pos))[0]
        # aggregate values
        token_items[key] = sum([item[1] for item in items], start=zero_sum_data_like(items[0][1]))

        i = j

    return token_items


def update_items_with_embedding(items, embedding):
    return {key: SumData(cnt=value.cnt, loss=value.loss, vector=value.vector, embedding=embedding[key.token_id])
            for key, value in items.items()}


def build_series(items):
    s = pd.Series(items)
    s.index.set_names(Key._fields, inplace=True)
    s.sort_index(inplace=True)
    return s


def build_series_from_pairs(pairs):
    keys, values = zip(*pairs)
    return pd.Series(data=values, index=pd.MultiIndex.from_tuples(keys, names=Key._fields))


ModelItems = namedtuple('ModelItems', ['losses', 'all_token_items', 'token_pos_items', 'token_items'])


def get_model_items(model, dataloader, pos_tags, ignore_all_token_items=True):
    """Get losses and items of various types.
    model: the model to run
    dataloader: dataloader to generate batches
    pos_tags: list of (list of pos tags representing the pos tags of the utterance)
    ignore_all_token_items: if True, ignore all_token_items and use None as placeholder
    Returns: ModelItems
    """

    if model is None or isinstance(model, NGramModel):
        hidden_dim = 0
        device = y.device
    else:
        hidden_dim = model.language_model.text_encoder.hidden_dim
        device = get_model_device(model)

    def torch_zero_sum_data():
        return SumData(
            cnt = np.array(0),
            loss = np.array(0.),
            vector = torch.zeros(hidden_dim, device=device),
            embedding = None,
        )

    losses = []
    all_token_items = [] if not ignore_all_token_items else None
    token_pos_items = defaultdict(torch_zero_sum_data)

    examples = run_model_on_data(model, dataloader)
    for (x, y, y_len, raw_y, outputs, loss), y_pos_tags in zip(examples, pos_tags):
        losses.append(loss[:y_len].cpu().numpy())
        for idx, pos_tag, loss_, outputs_ in zip(y, y_pos_tags, loss, outputs):
            key = Key(idx.item(), pos_tag)
            sdata = SumData(1, loss_.item(), outputs_.detach(), None)
            if not ignore_all_token_items:
                all_token_items.append((key, sdata))
            token_pos_items[key] += sdata

    token_pos_items = {key: value.to_numpy() for key, value in token_pos_items.items()}
    token_items = get_token_items(token_pos_items)

    if hasattr(model, 'text_encoder'):
        embedding = model.text_encoder.embedding.weight.detach().cpu().numpy()
        token_items = update_items_with_embedding(token_items, embedding)

    if all_token_items is not None:
        all_token_items = build_series_from_pairs(all_token_items)
    token_pos_items, token_items = map(build_series, (token_pos_items, token_items))

    return ModelItems(losses, all_token_items, token_pos_items, token_items)


def get_model_probs(model, dataloader, pos_tags):
    all_probs = []

    examples = run_model_on_data(model, dataloader, return_all=True)
    for (x, y, y_len, raw_y, outputs, loss, logits, attns, labels), y_pos_tags in zip(examples, pos_tags):
        probs = logits.softmax(-1)
        if is_regressional(model):
            # pad logits
            probs = F.pad(probs, (0, 0, 1, 0))

        probs = probs.cpu().numpy()

        for idx, pos_tag, probs_ in zip(y, y_pos_tags, probs):
            key = Key(idx.item(), pos_tag)
            all_probs.append((key, probs_))

    all_probs = build_series_from_pairs(all_probs)

    return all_probs


# aggregate results from multiple models


def stack_items(items_list, names, idx2word):
    if any(items is None for items in items_list):
        return None
    is_list = items_list and isinstance(items_list[0], list)
    if is_list:
        items_list = list(map(pd.Series, items_list))
    data = pd.DataFrame(dict(zip(names, items_list)), columns=names)
    if not is_list:
        data.reset_index(inplace=True)
        data[token_field] = data['token_id'].map(idx2word)
        data['pos'] = data['pos'].astype('category')
        data['cnt'] = data[names[0]].map(lambda value: value.cnt)
        data.set_index(list(Key._fields), drop=False, inplace=True)
    return data


def tokenwise_apply(fn, lst):
    """apply fn to every zipped tokenwise items in lst
    fn: an callable that accepts a list of items in each token position
    lst: list of data in the same form: list of list of token items
    """
    return [list(map(fn, zip(*example_lst))) for example_lst in zip(*lst)]


mean_losses = lambda losses_list: tokenwise_apply(lambda token_losses: np.mean(token_losses).item(), losses_list)


def mean_sum_data(sum_data_list, idx=0):
    sum_data_base = list(sum_data_list)[idx]
    return SumData(
        cnt = sum_data_base.cnt,
        loss = np.mean([sum_data.loss for sum_data in sum_data_list]).item(),
        vector = sum_data_base.vector,
        embedding = sum_data_base.embedding,
    )


def itemwise_apply(fn, items_list):
    data = pd.concat(items_list, axis=1)
    return data.apply(fn, axis=1)


mean_items = lambda items_list, idx=0: itemwise_apply(functools.partial(mean_sum_data, idx=idx), items_list)
mean_probs = lambda probs_list: itemwise_apply(functools.partial(np.mean, axis=0), probs_list)


def mean_model_items(model_items_list, idx=0):
    model_items = ModelItems(*zip(*model_items_list))
    return ModelItems(
        mean_losses(model_items.losses),
        *(None if items_list and items_list[0] is None else mean_items(items_list, idx=idx) for items_list in model_items[1:])
    )
