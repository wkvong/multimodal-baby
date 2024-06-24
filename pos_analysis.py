import itertools
import functools
from collections import namedtuple, Counter
import re
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from multimodal.multimodal_lit import MultiModalLitModel
from multimodal.text_only_data_module import TextOnlyDataModule
from multimodal.multimodal_data_module import PAD_TOKEN_ID, UNK_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID
import model_yulu
from ngram import NGramModel
from multimodal.utils import map_structure
from analysis_tools.processing import examples_from_batches, get_pos_tags, get_word_pos_cnt, get_word_pos_stat_from_word_pos_cnt, get_pos_stats_for_words, get_model_losses_on_batches, to_yulu_batch
from analysis_tools.sumdata import *
from analysis_tools.build_data import build_data
from analysis_tools.pos_tags import *
from analysis_tools.utils import *
from analysis_tools.word_categories import *
from analysis_tools.checkpoints import *


figsize = (13, 12)
sns.set_theme(
    style='white',
    font='monospace',
    rc={
        'figure.figsize': figsize,
        'axes.grid': False,
        'xtick.bottom': False,
        'xtick.top': False,
        'ytick.left': False,
        'ytick.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
    }
)

np.set_printoptions(suppress=True, precision=2, linewidth=120)
pd.options.display.width = 120

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pos_mapping = pos_mappings['syntactic category']

used_poses = ["noun", "verb", "adjective", "adverb", "function word", "cardinal number", "."][:2]

is_robustness_check = False



def pos_tagged_seq_repr(tokens, pos_tags):
    return ' '.join(f'{token}/{pos}' for token, pos in zip(tokens, pos_tags))


class Template(namedtuple('TemplateTuple', ['seq', 'pos', 'idx'])):
    """A template where the token at idx of seq is to be filled.
    """

    def __hash__(self):
        return hash((tuple(self.seq[:self.idx]), tuple(self.seq[self.idx+1:])))

    def __str__(self):
        return ' '.join(
            '_' * len(token) if i == self.idx else token
            for i, (token_id, token) in enumerate(
                (token_id, idx2word[token_id]) for token_id in self.seq)
            if token_id not in [SOS_TOKEN_ID, EOS_TOKEN_ID])


def template_from_str(s, use_unk=False):
    """Construct Template from str s. This is for manual construction of templates.
    Inputs:
        s: str. A sentence that is tokenized (tokens are separated by space) and containing exactly one slot, represented by '_'.
            Example: "here 's a _ and a kitty ."
    Return:
        a Template.
    """
    tokens = s.split()
    token_ids = []
    idx = None
    for i, token in enumerate(tokens):
        if re.fullmatch(r'_+', token):
            if idx is not None:
                raise Exception(f'more than one slot in template: {s}')
            idx = i
            token_id = UNK_TOKEN_ID
        else:
            try:
                token_id = word2idx[token]
            except KeyError:
                if use_unk:
                    token_id = UNK_TOKEN_ID
                else:
                    raise
        token_ids.append(token_id)
    if idx is None:
        raise Exception(f'must have one slot in template: {s}')

    token_ids = [SOS_TOKEN_ID] + token_ids + [EOS_TOKEN_ID]
    idx += 1
    pos = ["."] * len(token_ids)
    return Template(token_ids, pos, idx)


def templates_from_example(token_ids, pos_tags):
    for i, (token_id, pos) in enumerate(zip(token_ids, pos_tags)):
        if pos_mapping[pos] in used_poses:
            yield Template(token_ids, pos_tags, i)


def templates_from_examples(examples, pos_tags, print_pos_tagged_seq=False):
    tot, cnt_filtered = 0, 0

    for example, y_pos_tags in zip(examples, pos_tags):
        x, y, y_len, raw_y = example
        tot += 1
        if print_pos_tagged_seq:
            print(pos_tagged_seq_repr(raw_y[0].split(), y_pos_tags[1:]))
        cnt_present_tokens = sum((pos_mapping[pos] not in ["."] for pos in y_pos_tags))
        if cnt_present_tokens <= 2:
            continue
        y_len = y_len.item()
        y = y[:y_len].tolist()
        if UNK_TOKEN_ID in y:
            continue

        cnt_filtered += 1
        yield example, templates_from_example(y, y_pos_tags)

    print(f'filtered {cnt_filtered} / {tot} = {cnt_filtered / tot :.2%} examples')


def joined_templates_from_examples(*args, **kwargs):
    for example, example_templates in templates_from_examples(*args, **kwargs):
        for template in example_templates:
            yield example, template


def construct_batch_by_filling_template(template, word_ids, **kwargs):
    batch_size = len(word_ids)
    y = torch.tensor(template.seq, device=device)
    y_batch = y.unsqueeze(0).repeat(batch_size, *([1] * y.dim()))
    y_batch[:, template.idx] = torch.tensor(word_ids)
    y_len = torch.tensor(len(template.seq), device=device)
    y_len_batch = y_len.unsqueeze(0).expand(batch_size, *([-1] * y_len.dim()))
    return {
        'y': y_batch,
        'y_len': y_len_batch,
        **{
            key: (value.unsqueeze(0).expand(batch_size, *([-1] * value.dim())) if value is not None else None)
            for key, value in kwargs.items()
        }
    }


def construct_batch_by_masking(template):
    y = torch.tensor(template.seq, device=device)
    y[template.idx] = model_yulu.MASK_TOKEN_ID
    y_len = torch.tensor(len(template.seq), device=device)
    y_batch = y.unsqueeze(0)
    y_len_batch = y_len.unsqueeze(0)
    return y_batch, y_len_batch


def run_maskedLM_on_template(model, template, word_ids):
    y, y_len = construct_batch_by_masking(template)
    batch = to_yulu_batch(y, y_len)
    logits = model(batch, y == model_yulu.MASK_TOKEN_ID)
    logits = logits.squeeze(0)
    return -logits[..., word_ids]


def run_model_on_template(model, example, template, word_ids, batch_size=256):
    """Run model and get the whole sentence losses on template filled with every word_id in word_ids
    Inputs:
        model: the model
        example: the example; need to provide this for the model to have image x
        template: the template
        word_ids: fill the template with every word_id in word_ids
        batch_size: the batch size
    Returns:
        losses: an np.ndarray of length len(word_ids)
    """
    if isinstance(model, model_yulu.RobertaModel):
        return run_maskedLM_on_template(model, template, word_ids)

    if isinstance(model, MultiModalLitModel) and (
            model.language_model.text_encoder.captioning or
            model.language_model.text_encoder.has_attention):
        x, y, y_len, raw_y = example
        x = x.to(device=device)
        x = x.unsqueeze(0)
        image_features, image_feature_map = model.model.encode_image(x)
        image_features = image_features.squeeze(0)
        image_feature_map = image_feature_map.squeeze(0)
    else:
        image_features, image_feature_map = None, None

    batches = (
        construct_batch_by_filling_template(
            template,
            word_ids[i : i + batch_size],
            image_features=image_features,
            image_feature_map=image_feature_map,
        )
        for i in range(0, len(word_ids), batch_size))
    return get_model_losses_on_batches(model, batches)


def find_index(lst, e):
    for i, x in enumerate(lst):
        if x == e:
            return i
    return None


def get_prob_dist(model, example, template, word_ids, word_cats, pos_mapping=identity, batch_size=256, top_k=None):
    """Run model on template and get the probability distribution of categories;
    check whether model gives higher probability to the correct category.
    Inputs:
        model: the model
        example: the example; need to provide this for the model to have image x
        template: the template
        word_ids: fill the template with every word_id in word_ids
        word_cats: np.ndarray of same length as word_ids; the categories of each word in word_ids
        pos_mapping: callable, used to map pos to cat
        batch_size: the batch size used to run the model
        top_k: print top k predictions along with the ground-truth label; None if unwanted
    Returns:
        pd.Series with categories indexing the probability distribution
    """
    with torch.no_grad():
        losses = run_model_on_template(model, example, template, word_ids, batch_size=batch_size)
        probs = F.softmax(-losses, 0)

    gt_word_id = template.seq[template.idx]
    gt_word_pos = template.pos[template.idx]
    gt_word_cat = pos_mapping(gt_word_pos)
    gt_word_idx = find_index(word_ids, gt_word_id)
    if gt_word_idx is None:  # the ground-truth word is not in the vocab of prediction
        print(f'ground-truth word {idx2word[gt_word_id]} not in the vocab of prediction')
        # append the ground-truth word to the vocab
        gt_word_idx = len(word_ids)
        word_ids = np.append(word_ids, gt_word_id)
        word_cats = np.append(word_cats, gt_word_cat)
        losses = torch.cat([losses, torch.tensor([np.inf], dtype=losses.dtype, device=losses.device)])
        probs = torch.cat([probs, torch.tensor([0.], dtype=probs.dtype, device=probs.device)])
    else:
        # set the category of the ground-truth word to the category according to the template
        if word_cats[gt_word_idx] != gt_word_cat:
            word_cats = np.copy(word_cats)
            word_cats[gt_word_idx] = gt_word_cat

    if top_k is not None:
        words = idx2word[word_ids]
        print_losses = False
        if print_losses:
            print_top_values(-losses, words, labels=torch.tensor(gt_word_idx), value_formatter=lambda loss: f'{-loss:6.3f}')
        print_top_values(probs, words, labels=torch.tensor(gt_word_idx), value_formatter=prob_formatter)

    probs_ser = pd.Series(probs.cpu())
    return probs_ser.groupby(word_cats).sum()



# select from list of checkpoints
datasets_yulu_path = Path("/scratch/yq810/babyBerta/newBerta/data/corpora/")
dataset_yulu_name = ["saycam", "sarah", "ellie", "childes", "wiki"][3]
dataset_name = f"{dataset_yulu_name}_yulu"
dataset_path = datasets_yulu_path / dataset_yulu_name
names = {
    "saycam": [
        "LSTM 0", "LSTM 1", "LSTM 2",
        "Captioning LSTM 0", "Captioning LSTM 1", "Captioning LSTM 2",
        "CBOW 0", "CBOW 1", "CBOW 2",
    ],
    "coco": [
        "lm",
        "capt_ft",
        "capt_attn_gt_ft",
        "capt_attn_gt_reg_ft",
        "capt_attn_gt_reg_untie_ft",
        "cbow",
    ],
}[dataset_name] if not dataset_name.endswith("_yulu") else [
    "BabyBERTa 8-layers", "BabyBERTa 2-layers", "GPT2 8-layers", "GPT2 2-layers", "LSTM 1-layer", "LSTM 2-layers"
][-1:]
checkpoint_paths = all_checkpoint_paths[dataset_name]
checkpoint_paths = {name: checkpoint_paths[name] for name in names}

# build data and vocab according to the model
if "yulu" in dataset_name:
    data, args = build_data(args=dict(data_dir=dataset_path), dataset=TextOnlyDataModule, return_args=True)
else:
    data, args = build_data(return_args=True)
    dataset_name = args.dataset
vocab = data.datasets['train'].vocab
vocab_size = len(vocab)
print(f'{vocab_size = }')
word2idx = vocab
idx2word = [None] * vocab_size
for word, idx in word2idx.items():
    idx2word[idx] = word
idx2word = np.array(idx2word)

my_batch_size = 256
dataloader_fns = {
    'train': lambda: data.train_dataloader(batch_size=my_batch_size, shuffle=False, drop_last=False),
    'val': lambda: data.val_dataloader(batch_size=my_batch_size)[0],
    'test': lambda: data.test_dataloader(batch_size=my_batch_size)[0],
}


def LSTM_loader(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    expr_name = checkpoint_path.parent.name
    config = model_yulu.LSTMConfig(
        vocab_size = vocab_size - 1,
        num_layers = int(re.search(r'(?<=_num_layers_)\d+', expr_name).group(0)),
        dropout_rate = float(re.search(r'(?<=_dropout_)(\d|\.)+', expr_name).group(0)),
    )
    return model_yulu.LSTMModel(config=config, path=checkpoint_path, map_location=device)


# load model from checkpoint
models = {}
for name, checkpoint_path in checkpoint_paths.items():
    print(f"load {name} from {checkpoint_path}")
    model_loader = {
        "GPT2": lambda checkpoint_path: model_yulu.GPT2Model(path=checkpoint_path),
        "BabyBERTa": lambda checkpoint_path: model_yulu.RobertaModel(path=checkpoint_path),
        "LSTM": LSTM_loader,
    }.get(name.split()[0], lambda checkpoint_path: MultiModalLitModel.load_from_checkpoint(checkpoint_path, map_location=device))
    model = model_loader(checkpoint_path)
    model.to(device)
    model.eval()
    models[name] = model

# get the pos tags of all words in vocab
train_dataloader_fn = dataloader_fns['train']
train_pos_tags = get_pos_tags(train_dataloader_fn(), dataset_name, 'train')
word_pos_cnt = get_word_pos_cnt(train_dataloader_fn(), train_pos_tags)
word_pos_stat = get_word_pos_stat_from_word_pos_cnt(word_pos_cnt)
pos_stats = get_pos_stats_for_words(idx2word, word_pos_stat, pos_mapping=pos_mapping.get)
idx2pos2 = np.array([pos_stat[0][0] if pos_stat else '.' for pos_stat in pos_stats])

for pos in sorted(set(pos_mapping.values())):
    pos_word_ids = np.nonzero(idx2pos2 == pos)[0]
    print(f'#{pos:15s}: {len(pos_word_ids)}')

idx2used = np.isin(idx2pos2, used_poses)

for token_id in [PAD_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID, UNK_TOKEN_ID]:
    idx2used[token_id] = False

for idx, (word, pos_stat) in enumerate(zip(idx2word, pos_stats)):
    used_pos_stat = [(pos, cnt) for pos, cnt in pos_stat if pos in used_poses]
    if not used_pos_stat:
        continue
    word_cnt = sum(cnt for pos, cnt in used_pos_stat)
    stat_str = ' '.join(f'{pos:15} {cnt:5} {cnt / word_cnt :6.1%}' for pos, cnt in used_pos_stat)
    print(f'{word:10s} {stat_str}')

    # remove too ambiguous words
    if used_pos_stat[0][1] / word_cnt < 0.9:
        idx2used[idx] = False

# untypical words
my_untypical_words = ' '.join(
    possessives + negations + be_verbs + do_verbs + modal_verbs +
    pronoun_contractions + other_contractions + quantifiers + pos_ambiguous_words +
    special_tokens
).split()
my_untypical_word_ids = [word2idx[word] for word in my_untypical_words if word in word2idx]

# whether to excluding untypical words from the filling word vocabulary
not_filling_untypical_words = is_robustness_check
if not_filling_untypical_words:
    for word_id in my_untypical_word_ids:
        idx2used[word_id] = False

# get filling word vocabulary
word_ids = np.nonzero(idx2used)[0]
word_cats = idx2pos2[word_ids]
used_vocab_size = len(word_ids)
word_cat_counter = Counter(word_cats)
print(f'{used_vocab_size = } {word_cat_counter}')



# observation: many ambiguous cases seem to be confusing nouns vs V-ings/passive verbs.


printing_labels = False


splits = ['val']

filter_out_gt_word_ids = [SOS_TOKEN_ID, EOS_TOKEN_ID]
if is_robustness_check:
    filter_out_gt_word_ids += my_untypical_word_ids

seen_templates = set()
for split in ['train']:
    dataloader_fn = dataloader_fns[split]
    examples = examples_from_batches(dataloader_fn())
    pos_tags = get_pos_tags(dataloader_fn(), dataset_name, split)
    seen_templates.update(template for example, template in joined_templates_from_examples(examples, pos_tags))

for split in splits:
    print(f'{split} split:')
    dataloader_fn = dataloader_fns[split]
    examples = examples_from_batches(dataloader_fn())
    pos_tags = get_pos_tags(dataloader_fn(), dataset_name, split)
    golds, preds = [], {name: [] for name in names}
    for example, template in joined_templates_from_examples(examples, pos_tags):
        gt_word_id = template.seq[template.idx]
        if gt_word_id in filter_out_gt_word_ids:
            continue
        if not idx2used[gt_word_id]:
            continue
        print(template)
        if template in seen_templates:
            print('seen template; skipped')
            continue

        gold = pos_mapping[template.pos[template.idx]]
        golds.append(gold)
        for model_name, model in models.items():
            prob_dist = get_prob_dist(model, example, template, word_ids, word_cats, pos_mapping=pos_mapping.get, top_k=5)
            if printing_labels:
                print(prob_dist)
            argmax_idx = prob_dist.argmax()
            pred = prob_dist.index[argmax_idx]
            preds[model_name].append(pred)
            correct = pred == gold
            if printing_labels:
                print(f'{gold = }, {pred = }, {correct = }')

    print(f'label distribution: ')
    golds = np.array(golds)
    for pos in used_poses:
        cnt = (golds == pos).sum()
        print(f'{pos}: {frac_format(cnt, len(golds))}')
    for model_name, model in models.items():
        preds[model_name] = np.array(preds[model_name])
        corrects = preds[model_name] == golds
        print(f'{model_name} accuracy: {frac_format(corrects.sum(), len(corrects))}')
