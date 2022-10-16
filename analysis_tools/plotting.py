import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .representation_similarity import *
from .token_items_data import token_field, row_llf
from .utils import get_n_rows, get_np_attrs_from_values


output_fig = lambda fname: plt.show()


palette = {
    "''": "black",
    "``": "black",
    ",": "black",
    ".": "black",
    ":": "black",
    ".": "black",
    "-LRB-": "black",
    "HYPH": "black",
    "LS": "black",
    "SYM": "black",
    "CC": "brown",
    "cardinal number": "tab:gray",
    "CD": "tab:gray",
    "DT": "salmon",
    "EX": "rosybrown",
    "function word": "tab:purple",
    "IN": "tab:purple",
    "adjective": "tab:orange",
    "JJ": "tab:orange",
    "JJR": "gold",
    "JJS": "yellow",
    "MD": "rosybrown",
    "noun": "tab:blue",
    "NN": "tab:blue",
    "NNP": "navy",
    "NNS": "cadetblue",
    "PDT": "saddlebrown",
    "POS": "grey",
    "PRP": "tab:olive",
    "PRP$": "olivedrab",
    "adverb": "tab:pink",
    "RB": "tab:pink",
    "RBR": "magenta",
    "RBS": "palevioletred",
    "RP": "darkviolet",
    "TO": "indigo",
    "ADD": "tab:green",
    "FW": "tab:green",
    "GW": "tab:green",
    "NFP": "tab:green",
    "UH": "tab:green",
    "verb": "tab:red",
    "VB": "tab:red",
    "VBD": "salmon",
    "VBG": "orangered",
    "VBN": "chocolate",
    "VBP": "pink",
    "VBZ": "crimson",
    "WDT": "darkcyan",
    "WP": "slateblue",
    "WP$": "darkslateblue",
    "wh-word": "tab:cyan",
    "WRB": "tab:cyan",

    # noun subcategories
    "body_parts": "red",
    "clothing": "blue",
    "food_drink": "orange",
    "vehicles": "gray",
    "toys": "green",
    "animals": "pink",
    "household": "cyan",
    "places": "brown",
    "sounds": "yellow",
    "furniture_rooms": "maroon",
    "outside": "limegreen",
    "people": "fuchsia",
    "games_routines": "slategray",

    # verb subcategories
    "trans. verb": "gold",
    "intrans. verb": "purple",
    "(in)trans. verb": "peru",
    "special verb": "lime",
}


def build_linkage_by_same_value(s):
    """build the linkage by clustering same values in the series s
    """
    idxes = list(range(len(s)))
    key_fn = lambda idx: s.iloc[idx]
    idxes.sort(key=key_fn)

    # get groups by finding contingous segments on sorted idxes
    l, r = 0, 0
    groups = []
    while l < len(idxes):
        while r < len(idxes) and key_fn(idxes[l]) == key_fn(idxes[r]):
            r += 1
        groups.append(idxes[l:r])
        l = r

    # initialization
    Z = [(idx, idx, 0., 1) for idx in range(len(s))]

    def merge(idx0, idx1, distance):
        Z.append((idx0, idx1, distance, Z[idx0][3] + Z[idx1][3]))
        return len(Z) - 1

    def merge_group(group, distance):
        root = group[0]
        for idx in group[1:]:
            root = merge(root, idx, distance)
        return root

    # merge items within each group
    group_idxes = [merge_group(group, 0.) for group in groups]
    # merge groups
    root = merge_group(group_idxes, 1.)

    return np.array(Z[len(s):])



def plot_sim_heatmap(matrix, labels, annot=True, size=0.7, ax=None):
    designated_ax = ax is not None
    ax = sns.heatmap(matrix, vmin=-1, vmax=1, center=0, annot=annot, fmt='.2f', xticklabels=labels, yticklabels=labels, square=True, cbar=False, ax=ax)
    if not designated_ax:
        ax.figure.set_size_inches(size * (matrix.shape[0] + 2.), size * (matrix.shape[1] + 2.))
    return ax


def plot_repres_sim_heatmap(vectors, names, ax=None):
    dissim_matrices = [cosine_dissim_matrix(V) for V in vectors]

    repres_sim_matrix = np.array([[rsa_of_dissim_matrices(A, B) for B in dissim_matrices] for A in dissim_matrices])
    return plot_sim_heatmap(repres_sim_matrix, names, annot=True, size=1., ax=ax)


def plot_model_y_value_heatmap(names, values, y_labels, annot=True, size=0.7, plot_diff=True, plot_ori=False):
    values = np.array(values)
    data = [values[0]]
    yticklabels = [names[0]]
    for i in range(1, len(values)):
        if plot_diff:
            data.append(values[i] - values[0])
            yticklabels.append(f'{names[i]} - {names[0]}')
        if plot_ori:
            data.append(values[i])
            yticklabels.append(f'{names[i]}')
    ax = sns.heatmap(data, center=0, annot=annot, fmt='.2f', xticklabels=y_labels, yticklabels=yticklabels, square=False, cbar=False)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    ax.figure.set_size_inches(size * (len(data[0]) + 1.), size * 0.5 * (len(data) + .5))


def plot_vector_sim_heatmap(items, names, diff=False, vector_attr='mean_vector', one_figure=False, size=0.7, figname='similarity heatmap', **kwargs):
    if diff:
        if len(items) % 2 != 0:
            print('Error: number of items should be even.')
            return

    tokens = items[token_field]
    if diff:
        labels = [f'{tokens.iloc[i]}-{tokens.iloc[i+1]}' for i in range(0, len(tokens), 2)]
    vectors = [get_np_attrs_from_values(items[name], vector_attr) for name in names]

    if one_figure:
        n_cols = 3
        n_rows = get_n_rows((0 if diff else 1) + len(vectors), n_cols)
        s = size * (len(items) + 2.)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(s * n_cols, s * n_rows), squeeze=False)
        all_axes = itertools.chain.from_iterable(axes)

    if not diff:
        ax = plot_repres_sim_heatmap(vectors, names, ax=next(all_axes) if one_figure else None)
        _title = figname + ' RSA'
        ax.set_title(_title)
        if not one_figure:
            output_fig(_title)

    for V, name in zip(vectors, names):
        if diff:
            V = V.reshape((V.shape[0] // 2, 2,) + V.shape[1:])
            V = V[:, 1] - V[:, 0]
        ax = plot_sim_heatmap(cosine_matrix(V), labels if diff else tokens, size=size, ax=next(all_axes) if one_figure else None, **kwargs)
        _title = figname + f' {name}'
        ax.set_title(_title)
        if not one_figure:
            output_fig(_title)

    if one_figure:
        for ax in all_axes:
            ax.axis("off")
        output_fig(figname)


plotting_variable_keys = {'x', 'y', 'hue', 'size', 'style'}

def plot(
    fn,
    items,
    token_kwargs=None,
    axis_option="on",
    xlabel=None, ylabel=None,
    **kwargs
):
    """plot items using fn
    fn: seaborn plot function
    items: pd.DataFrame items; will drop items with missing values in the variables
    token_kwargs: kwargs to plt.text to add token text labels; if None, do not add text labels
    kwargs: all other kwargs to pass to fn
    """
    variable_keys = plotting_variable_keys & kwargs.keys()
    variable_keys = {key for key in variable_keys if kwargs[key] is not None}

    ret = fn(data=items, **kwargs)

    if token_kwargs is not None and 'x' in variable_keys and 'y' in variable_keys:
        from adjustText import adjust_text
        x = kwargs['x']
        y = kwargs['y']
        texts = [
            plt.text(
                row[x], row[y], row[token_field],
                ha='center', va='center', **token_kwargs)
            for _, row in items.iterrows()]
        adjust_text(texts)

    if isinstance(ret, sns.FacetGrid):
        all_ax = itertools.chain.from_iterable(ret.axes)
    else:
        all_ax = [ret]

    for ax in all_ax:
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.axis(axis_option)

    return ret
