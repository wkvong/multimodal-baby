import itertools
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from .representation_similarity import *
from .token_items_data import token_field
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


def plot_sim_heatmap(matrix, labels, annot=True, size=0.7, ax=None):
    designated_ax = ax is not None
    ax = sns.heatmap(matrix, vmin=-1, vmax=1, center=0, annot=annot, fmt='.2f', xticklabels=labels, yticklabels=labels, square=True, cbar=False, ax=ax)
    if not designated_ax:
        ax.figure.set_size_inches(size * (matrix.shape[0] + 2.), size * (matrix.shape[1] + 2.))
    return ax


def plot_rsa_heatmap(
    vectors, names,
    dissim_matrix_fn=cosine_dissim_matrix,
    correlation=scipy.stats.pearsonr,
    get_value_fn=lambda result: result.statistic,  # for pearsonr
    ax=None,
):
    dissim_matrices = [dissim_matrix_fn(V) for V in vectors]
    rsa_results = [
        [rsa_of_dissim_matrices(A, B)
         for B in dissim_matrices]
        for A in dissim_matrices]
    rsa_values = np.array([
        [get_value_fn(result)
         for result in rsa_result_row]
        for rsa_result_row in rsa_results])
    return (plot_sim_heatmap(rsa_values, names, annot=True, size=1., ax=ax),
            rsa_results)


def plot_model_y_value_heatmap(
    names, values, y_labels,
    plot_diff=True, plot_ori=False,
    center=0., annot=True, fmt='.2f', square=False, cbar=False, **kwargs
):
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
    ax = sns.heatmap(
        data,
        xticklabels=y_labels,
        yticklabels=yticklabels,
        center=center, annot=annot, fmt=fmt, square=square, cbar=cbar, **kwargs
    )
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    return ax, data


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
        ax, rsa_results = plot_rsa_heatmap(vectors, names, ax=next(all_axes) if one_figure else None)
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

def plot_wrapper(plot_fn):
    """wrapping seaborn plot_fn with additional functions
    plot_fn: seaborn plot function to wrap
    """
    def wrapped_plot_fn(
        data=None,
        text_label=None,
        text_label_kwargs=None,
        axis_option="on",
        xlabel=None, ylabel=None,
        **kwargs
    ):
        """ Wrapped seaborn plot function
        data: pd.DataFrame
        text_label: column name of text label; if None, do not add text labels
        text_label_kwargs: kwargs to plt.text for text labels
        axis_option: set all axis with axis_option
        xlabel, ylabel: set all axis with xlabel and ylabel
        kwargs: all other kwargs to pass to fn
        """
        variable_keys = (
            plotting_variable_keys &
            {key for key, value in kwargs.items() if value is not None}
        )

        ret = plot_fn(data=data, **kwargs)

        if (text_label is not None
            and 'x' in variable_keys and 'y' in variable_keys):
            from adjustText import adjust_text
            x = kwargs['x']
            y = kwargs['y']
            texts = [
                plt.text(
                    row[x], row[y], row[text_label],
                    ha='center', va='center', **text_label_kwargs)
                for _, row in data.iterrows()]
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

    return wrapped_plot_fn
