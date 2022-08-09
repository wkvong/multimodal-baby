import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .representation_similarity import *
from .token_items_data import token_field, row_llf
from .pos_tags import pos_palette
from .word_categories import subcat_palette
from .utils import get_n_rows, get_np_attrs_from_values


def get_palette(field):
    if 'pos' in field:
        palette = pos_palette
    elif field == 'subcat':
        palette = subcat_palette
    else:
        palette = None
    return palette


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



def plot_sim_heatmap(matrix, labels, annot=True, size=0.7, title=None, ax=None):
    designated_ax = ax is not None
    ax = sns.heatmap(matrix, vmin=-1, vmax=1, center=0, annot=annot, fmt='.2f', xticklabels=labels, yticklabels=labels, square=True, cbar=False, ax=ax)
    if not designated_ax:
        ax.figure.set_size_inches(size * (matrix.shape[0] + 2.), size * (matrix.shape[1] + 2.))
    if title is not None:
        ax.set_title(title)


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
    ax.figure.set_size_inches(size * (len(data[0]) + 1.), size * 0.5 * (len(data) + .5))
    plt.show()


def plot_vector_sim_heatmap(items, names, diff=False, vector_attr='mean_vector', one_figure=False, size=0.7, **kwargs):
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
        plot_repres_sim_heatmap(vectors, names, ax=next(all_axes) if one_figure else None)
        if not one_figure:
            plt.show()

    for V, name in zip(vectors, names):
        if diff:
            V = V.reshape((V.shape[0] // 2, 2,) + V.shape[1:])
            V = V[:, 1] - V[:, 0]
        plot_sim_heatmap(cosine_matrix(V), labels if diff else tokens, size=size, title=name, ax=next(all_axes) if one_figure else None, **kwargs)
        if not one_figure:
            plt.show()

    if one_figure:
        for ax in all_axes:
            ax.axis("off")
        plt.show()


def plot_repres_sim_heatmap(vectors, names, title=None, ax=None):
    dissim_matrices = [cosine_dissim_matrix(V) for V in vectors]

    repres_sim_matrix = np.array([[rsa_of_dissim_matrices(A, B) for B in dissim_matrices] for A in dissim_matrices])
    plot_sim_heatmap(repres_sim_matrix, names, annot=True, size=1., title=title, ax=ax)



def plot_dendrogram(items, names, vector_attr='mean_vector', heatmap=False, annot=False, size=0.7, heatmap_linkage=None, tag_field='pos0', ll_tag_field='pos', ll_with_cnt=True, ll_with_ppl=True, title=None):
    """linkage clustering and dendrogram plotting
    items: pd.DataFrame
    names: the names of the models to plot
    vector_attr: use value.vector_attr; default: 'mean_vector'; can be 'embedding'
    heatmap: bool, plot something like plot_sim_heatmap
    heatmap_linkage: the row_linkage and col_linkage used in heatmap; can be any of:
        None: use the linkage result from the vectors of current model
        "first": use the linkage result of the first model
        "tag": build the linkage by clustering items by tag_field
        result from linkage function
    tag_field: the field of tags to obtain the palette of the sidebar of heatmaps
    ll_tag_field: the field of tags to append to leaf labels; set to None or empty string if unwanted
    ll_with_cnt: whether to append cnt to leaf labels
    ll_with_ppl: whether to append ppl to leaf labels
    title: title of the plots
    """
    from scipy.cluster.hierarchy import dendrogram, linkage

    n_items = len(items)
    vectors = [get_np_attrs_from_values(items[name], vector_attr) for name in names]

    plot_repres_sim_heatmap(vectors, names, title=title)
    plt.show()

    # build color map
    colors = items[tag_field].astype('O').map(get_palette(tag_field)).tolist()

    if heatmap:
        if heatmap_linkage == "tag":  # build Z_heatmap based on tag_field
            Z_heatmap = build_linkage_by_same_value(items[tag_field])
        elif not (heatmap_linkage is None or isinstance(heatmap_linkage, str)):  # use heatmap_linkage
            Z_heatmap = heatmap_linkage

    for n, (name, V) in enumerate(zip(names, vectors)):
        print(f'{name}:')
        Z = linkage(V, method='average', metric='cosine')  # of shape (number of merges = n_items - 1, 4)

        def llf(index):
            if index < n_items:
                return row_llf(
                    items.iloc[index],
                    tag_field=ll_tag_field,
                    with_cnt=ll_with_cnt,
                    name=name if ll_with_ppl else None,
                    baseline_name=None if n == 0 else names[0],
                )
            else:
                merge_index = index - n_items
                return f'{merge_index} {int(Z[merge_index, 3])} {Z[merge_index, 2]:.3f}'

        p = 10000

        plt.figure(figsize=(25 / 2, 0.3 * min(p, n_items))) # 0.1
        dendrogram(
            Z,
            truncate_mode='lastp',
            p=p,
            orientation='left',
            leaf_rotation=0.,
            leaf_font_size=16.,
            leaf_label_func=llf,
        )

        if title is not None:
            plt.title(f"{name} {title}")

        if heatmap:
            if heatmap_linkage is None:
                Z_heatmap = Z
            elif heatmap_linkage == "first":
                if n == 0:
                    Z_heatmap = Z

            prefix_labels = [row_llf(row, tag_field=tag_field, with_cnt=False) for _, row in items.iterrows()]
            llf_labels = list(map(llf, range(n_items)))

            matrix = cosine_matrix(V)

            off_diag = ~np.eye(matrix.shape[0], matrix.shape[1], dtype=bool)
            v = np.max(np.abs(matrix[off_diag]))
            vmin = -v
            vmax = +v

            g = sns.clustermap(
                matrix,
                row_linkage=Z_heatmap,
                col_linkage=Z_heatmap,
                figsize=(22, 20),
                cbar_pos=None,
                # kwargs for heatmap
                vmin=vmin, vmax=vmax, center=0,
                annot=annot, fmt='.2f',
                xticklabels=prefix_labels,
                yticklabels=prefix_labels,
                row_colors=colors,
                col_colors=colors,
                square=True,
                #cbar=False,
                dendrogram_ratio=0., # remove all dendrograms
                colors_ratio=0.02,
            )
            g.ax_col_dendrogram.remove()

            if title is not None:
                plt.title(f"{name} {title}")

        plt.show()



plotting_variable_keys = {'x', 'y', 'hue', 'size', 'style'}
default_figsize = (13, 12)

def plot(fn, items, n_items=None, xrange=None, yrange=None, token_kwargs=None, palette=None, title=None, suptitle=None, hlines=None, vlines=None, figsize=default_figsize, **kwargs):
    """plot items using fn
    fn: seaborn plot function
    items: pd.DataFrame items; will drop items with missing values in the variables
    n_items: plot at most n_items items
    xrange, yrange: plot points only within this range
    token_kwargs: kwargs to plt.text to add token text labels; if None, do not add text labels
    palette: palette for hue; if None, for pos it will use pos_palette; for other categories, use 'tab20'
    kwargs: all other kwargs to pass to fn
    """

    if hlines is None:
        hlines = []
    if vlines is None:
        vlines = []

    variable_keys = plotting_variable_keys & kwargs.keys()
    variable_keys = {key for key in variable_keys if kwargs[key] is not None}

    if palette is None and 'hue' in variable_keys:
        hue = kwargs['hue']
        palette = get_palette(hue)
        if palette is None and items.dtypes[hue] == "category":
            palette = 'tab20'
    if fn not in [sns.regplot]:
        kwargs['palette'] = palette

    data = items.dropna(subset=[kwargs[key] for key in variable_keys])
    if xrange is not None:
        x = kwargs['x']
        data = data[data[x].map(lambda x: xrange[0] <= x <= xrange[1])]
    if yrange is not None:
        y = kwargs['y']
        data = data[data[y].map(lambda y: yrange[0] <= y <= yrange[1])]
    if n_items is not None:
        data = data.iloc[:n_items]
    print(f'plotting {len(data)}/{len(items)} = {len(data) / len(items) if len(items) else 0.:.2%} items...')
    ret = fn(data=data, **kwargs)

    if token_kwargs is not None and 'x' in variable_keys and 'y' in variable_keys:
        x = kwargs['x']
        y = kwargs['y']
        for _, row in data.iterrows():
            plt.text(row[x], row[y], row[token_field], **token_kwargs)

    if isinstance(ret, sns.FacetGrid):
        ret.figure.set_size_inches(*figsize)
        all_ax = itertools.chain.from_iterable(ret.axes)
    else:
        all_ax = [ret]

    for ax in all_ax:
        for hline in hlines:
            ax.axhline(hline)
        for vline in vlines:
            ax.axvline(vline)

    if title is not None:
        if title == "vs":
            title = f"{kwargs['x']} vs {kwargs['y']}"
        plt.title(title)
    if suptitle is not None:
        plt.suptitle(suptitle)

    plt.show()
    return ret
