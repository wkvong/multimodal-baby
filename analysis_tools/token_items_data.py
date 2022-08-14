from collections import namedtuple
import numpy as np
from .pos_tags import pos_mappings
from .sumdata import *
from .word_ratings import *


token_field = 'token'


class Key(namedtuple('Key', ['token_id', 'pos'])):
    """
    token_id: token index
    pos: pos tag
    """

# string representations of row information


def row_prefix_str(row, tag_field='pos', with_cnt=False):
    prefix = f"{row[token_field]:<10} {row[tag_field]:<4}" if tag_field and not pd.isna(row[tag_field]) else f"{row[token_field]:<15}"
    if with_cnt:
        prefix = prefix + f" {row['cnt']:>6}"
    return prefix


def row_str(row, names):
    return row_prefix_str(row, with_cnt=True) + ': ' + ' '.join(f"{row[name].ppl:9.3f}" for name in names)


def row_llf(row, tag_field='pos', with_cnt=True, name=None, baseline_name=None):
    s = row_prefix_str(row, tag_field=tag_field, with_cnt=with_cnt)
    if name is not None:
        ppl = row[name].ppl
        baseline_ppl = 0. if baseline_name is None else row[baseline_name].ppl
        s = s + ' ' + f'{ppl-base_ppl:+9.2f}={ppl:8.2f}'
    return s


# extending items with various information


def extend_point_items(items, name, attr, n=None):
    if n is None:
        n = {'tsne': 2, 'eigen': 5}[attr]
    for idx in range(n):
        items[f'{name} {attr} {idx}'] = items[name].map(lambda value: getattr(value, attr + '_point')[idx])


def extend_items_value_diff(items, name, baseline_name, value_attr):
    items[f'{name} {value_attr} diff'] = items[f'{name} {value_attr}'] - items[f'{baseline_name} {value_attr}']


def extend_items_for_name(items, name, baseline_name=None):
    # add f'{name} loss'
    items[f'{name} loss'] = items[name].map(lambda value: value.mean_loss)
    # add f'{name} prob'
    items[f'{name} prob'] = np.exp(-items[f'{name} loss'])

    if baseline_name is not None:
        # add f'{name} loss diff'
        extend_items_value_diff(items, name, baseline_name, 'loss')
        # add f'{name} prob diff'
        extend_items_value_diff(items, name, baseline_name, 'prob')

    try:
        # add tsne
        extend_point_items(items, name, 'tsne')
    except AttributeError:
        pass
    try:
        # add eigen
        extend_point_items(items, name, 'eigen')
    except AttributeError:
        pass


def extend_items(items, names, idx2word, baseline_name=None):
    """Extend the fields of items
    """

    # add 'pos'
    for pos_field, pos_mapping in pos_mappings:
        items[pos_field] = items['pos'].map(pos_mapping).astype('category')
    # add 'logcnt'
    items['logcnt'] = np.log(items['cnt'])

    # use the first name as the baseline by default
    if baseline_name is None:
        baseline_name = names[0]

    extend_items_for_name(items, baseline_name)  # get results from the baseline_name first
    for name in names:
        if name != baseline_name:
            extend_items_for_name(items, name, baseline_name)

    # add all fields from imported data
    try:
        concreteness_data.extend_items(items, 'conc Word', idx2word)
    except:
        print('failed to extend items by concreteness data')
    try:
        norm_data.extend_items(items, 'norm Word', idx2word)
    except:
        print('failed to extend items by norm data')