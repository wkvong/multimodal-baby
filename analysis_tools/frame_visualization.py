import re
from textwrap import wrap
import matplotlib.pyplot as plt


def frame_subplots(
    nrows=1, ncols=1,
    fig_width=8.,
    gridspec_kw={'wspace': 0.025, 'hspace': 0.25},
    aspect=1.,
    squeeze=False,
    **kwargs
):
    frame_width = fig_width / (ncols + (ncols - 1) * gridspec_kw['wspace'])
    frame_height = frame_width / aspect
    fig_height = frame_height * (nrows + (nrows - 1) * gridspec_kw['hspace'])
    fig, ax = plt.subplots(
        nrows, ncols,
        figsize=(fig_width, fig_height),
        squeeze=squeeze,
        gridspec_kw=gridspec_kw,
        **kwargs
    )
    return fig, ax, frame_width


def untokenize(utterance):
    return re.sub(r" (?=([\.\?\!,']|((n('|)t|na)\b)))", r"", utterance)


def get_wrap_width(fontsize, frame_width, c=96.):
    return int(c * frame_width / fontsize)


def add_caption(
    ax,
    caption,
    fontsize=5.,
    wrap_width=None,
    frame_width=None,
    **kwargs
):
    if wrap_width is None:
        wrap_width = get_wrap_width(fontsize, frame_width)
    return ax.text(
        0.5, -0.05,
        "\n".join(wrap(caption[:100], wrap_width)),
        ha='center', va='top',
        transform=ax.transAxes,
        fontsize=fontsize,
    )
