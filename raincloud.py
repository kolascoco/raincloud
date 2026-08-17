"""
raincloud.py -- half-violin + boxen + swarm ("raincloud") plots with seaborn.

Reproduces the R/ggplot2 look of `ggdist::stat_halfeye()` + `geom_boxplot()` +
`geom_jitter()` (the layout used in panel (a) of the reference figure), but
built purely from seaborn primitives:

    violinplot  -> clipped to its upper half   (the "cloud")
    boxenplot   -> letter-value box            (the summary)
    swarmplot   -> non-overlapping points      (the "rain")
    pointplot   -> optional mean marker

Public API
----------
raincloud(...)      draw one raincloud axes
raincloud_grid(...) facet several value columns side by side (figure of panel a)

Requires seaborn >= 0.13 (uses the new categorical engine).
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.transforms import Affine2D

__all__ = ["raincloud", "raincloud_grid", "save_figure",
           "PALETTE", "PALETTE_MUTED", "REFERENCE_PALETTE"]

# Default trio: steel blue / amber / deep green. Chosen so the three separate
# by lightness as well as hue (L* ~ 47 / 66 / 39) -- they stay distinct in
# greyscale, and under deuteranopia, where the amber and the green drift
# towards a common hue, the lightness gap still holds them apart.
PALETTE = ["#3E6FA8", "#E2913A", "#2C6B4F"]
PALETTE_MUTED = ["#7C9CC4", "#EDBA84", "#7BA793"]   # same hues, for fills

# Palette sampled from the reference figure (orange / teal / dusty red).
REFERENCE_PALETTE = ["#F0A73A", "#3DA89B", "#C4576B"]


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _new_artists(ax, before):
    """Artists added to `ax` since the `before` snapshot."""
    b_coll, b_lines, b_patches = before
    return (
        [a for a in ax.collections if a not in b_coll],
        [a for a in ax.lines if a not in b_lines],
        [a for a in ax.patches if a not in b_patches],
    )


def _snapshot(ax):
    return (list(ax.collections), list(ax.lines), list(ax.patches))


def _shift(artists, delta, cat_axis, ax, *, offsets_only=False):
    """Translate artists by `delta` data units along the categorical axis.

    `offsets_only=True` is required for swarmplot: seaborn recomputes the
    beeswarm layout (and rewrites the offsets) on *every* draw, so the shift
    has to live in the offset transform instead of in the offset data.
    """
    dx, dy = (delta, 0.0) if cat_axis == "x" else (0.0, delta)
    trans = Affine2D().translate(dx, dy)
    for art in artists:
        if offsets_only:
            art.set_offset_transform(trans + ax.transData)
        else:
            art.set_transform(trans + ax.transData)


def _split_point_alpha(coll, face_alpha, edge_alpha, edgecolor=None):
    """Give a scatter collection a translucent fill over an opaque stroke.

    `Collection.set_alpha` applies to fill *and* stroke, so the two opacities
    have to be baked into the RGBA arrays instead.
    """
    face = np.array(coll.get_facecolors(), dtype=float)
    if not face.size:
        return
    edge = face.copy()
    if edgecolor is not None:
        edge[:, :3] = to_rgb(edgecolor)
    face[:, 3] = face_alpha
    edge[:, 3] = edge_alpha
    coll.set_alpha(None)          # drop any collection-wide alpha first
    coll.set_facecolors(face)
    coll.set_edgecolors(edge)


def _clip_to_half(collections, cat_axis, sign):
    """Fold each (symmetric) violin body onto the half that faces `sign`."""
    idx = 0 if cat_axis == "x" else 1
    fold = np.maximum if sign > 0 else np.minimum
    for coll in collections:
        for path in coll.get_paths():
            v = path.vertices
            center = 0.5 * (v[:, idx].min() + v[:, idx].max())
            v[:, idx] = fold(v[:, idx], center)
        coll.stale = True


def _palette_colors(palette, levels):
    """Colours for `levels`, accepting a name / list / dict / None palette."""
    if isinstance(palette, dict):
        return [palette[lvl] for lvl in levels]
    if palette is None:
        return sns.color_palette(n_colors=len(levels))
    return sns.color_palette(palette, len(levels))


class _beeswarm_width:
    """Force seaborn's beeswarm to pack/clamp inside a custom band width.

    ``sns.swarmplot`` hard-codes the swarm width to 0.8 category slots and
    exposes no parameter for it, so the band is narrowed by swapping in a
    ``Beeswarm`` subclass for the duration of the call. Points that no longer
    fit are pushed into the gutters -- which is exactly the dense flat band of
    the reference figure.
    """

    def __init__(self, width):
        self.width = width
        self._orig = sns.categorical.Beeswarm

    def __enter__(self):
        forced, orig = self.width, self._orig

        class _Narrow(orig):
            def __init__(self, orient="x", width=0.8, warn_thresh=0.05):
                super().__init__(orient=orient, width=forced, warn_thresh=warn_thresh)

        sns.categorical.Beeswarm = _Narrow
        return self

    def __exit__(self, *exc):
        sns.categorical.Beeswarm = self._orig
        return False


def _resolve_orient(orient, x, y, data):
    """Return ('h'|'v') -- 'h' = value on x, categories on y (reference look)."""
    if orient in ("h", "y"):
        return "h"
    if orient in ("v", "x"):
        return "v"
    # infer: the categorical-looking variable defines the orientation
    for var, other in ((y, x), (x, y)):
        if var is None:
            continue
        s = data[var] if isinstance(var, str) else pd.Series(var)
        if not pd.api.types.is_numeric_dtype(s):
            return "h" if var is y else "v"
    return "h"


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def raincloud(
    data=None,
    *,
    x=None,
    y=None,
    hue=None,
    order=None,
    hue_order=None,
    palette=None,
    color=None,
    orient="h",
    dodge=True,
    ax=None,
    # --- geometry (data units of one category slot, which is 1.0 wide) ------
    violin_width=0.70,
    box_width=0.13,
    point_width=0.32,
    # the violin sits low enough that its base cuts a third of the way down
    # the box:  violin_offset = box_offset + box_width / 6
    violin_offset=0.0117,
    box_offset=-0.01,
    point_offset=-0.25,
    # --- violin ------------------------------------------------------------
    bw_adjust=0.9,
    cut=0,
    density_norm="width",
    violin_alpha=0.85,
    # --- box ---------------------------------------------------------------
    box_kind="boxen",
    k_depth="tukey",
    whis=1.5,
    box_alpha=1.0,
    showfliers=False,
    # --- points ------------------------------------------------------------
    point_kind="auto",
    point_size=2.8,
    point_alpha=0.6,
    point_edge_alpha=1.0,
    point_edge_width=0.45,
    point_edgecolor=None,
    swarm_limit=600,
    jitter=None,
    # --- extras ------------------------------------------------------------
    show_mean=True,
    mean_kws=None,
    linewidth=0.5,
    linecolor="0.15",
    legend=True,
    legend_title=None,
    violin_kws=None,
    box_kws=None,
    point_kws=None,
):
    """Draw a raincloud: half violin + letter-value box + swarm of raw points.

    Parameters
    ----------
    data : DataFrame (long form)
    x, y : column names. With ``orient="h"`` (default) ``x`` is the value and
        ``y`` the grouping variable -- the layout of the reference figure.
    hue : optional second grouping variable; dodged inside each category.
    palette / color : palette for ``hue`` (name, list or dict), or a single
        colour when there is no ``hue``.
    orient : {"h", "v", None}. ``None`` infers from the dtypes.
    violin_width / box_width / point_width : widths of the three layers,
        in units of one category slot (which is 1.0 wide, so keep the sum of
        widths + offsets inside +-0.5).
    violin_offset / box_offset / point_offset : signed positions of each layer
        relative to the category centre. Positive = towards the violin side.
        The defaults sit the violin's base a third of the way down the box, so
        the two interlock; ``violin_offset = box_offset + box_width / 6`` keeps
        that relation if you resize the box.
    point_kind : {"auto", "swarm", "strip", "none"}. ``"auto"`` falls back to a
        jittered strip once a group has more than ``swarm_limit`` points,
        because beeswarm layout is O(n^2).
    box_kind : {"boxen", "box"}. ``"boxen"`` is a letter-value box, which shows
        more of the tail; ``"box"`` is a plain Tukey boxplot. ``k_depth``
        applies to the former, ``whis`` to the latter.
    point_alpha / point_edge_alpha : the dots carry two opacities -- a
        translucent fill over a fully opaque stroke, so that overlapping dots
        still read as individual observations. ``point_edgecolor=None`` keeps
        the stroke in each dot's own colour.
    show_mean : draw a black dot at the group mean (as in the reference).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.gca()

    orient = _resolve_orient(orient, x, y, data)
    cat_axis = "y" if orient == "h" else "x"
    cat = y if orient == "h" else x

    hue_levels = None
    if hue is not None:
        hue_levels = list(hue_order) if hue_order is not None \
            else list(pd.unique(data[hue].dropna()))

    # Hue handling. seaborn dodges each layer by *that layer's* `width`, so a
    # 0.13-wide box and a 0.70-wide violin would end up in different places.
    # When dodging we therefore draw one hue level at a time and place the
    # sub-slots ourselves; everything then shares one geometry.
    if hue_levels is not None and dodge and len(hue_levels) > 1:
        n_hue = len(hue_levels)
        colors = _palette_colors(palette, hue_levels)
        units = [
            (data[data[hue] == lvl],                      # subset
             (j + 0.5) / n_hue - 0.5,                     # sub-slot centre
             {"color": colors[j]})                        # colour spec
            for j, lvl in enumerate(hue_levels)
        ]
    elif hue_levels is not None:
        n_hue = 1
        units = [(data, 0.0,
                  {"hue": hue, "hue_order": hue_order, "palette": palette})]
    else:
        # no hue: pin one colour, otherwise every seaborn call would advance
        # the axes colour cycle and the three layers would disagree
        n_hue = 1
        if color is None:
            color = sns.color_palette(palette, 1)[0] if palette is not None \
                else sns.color_palette()[0]
        units = [(data, 0.0, {"color": color})]
    scale = 1.0 / n_hue

    kind = point_kind
    if kind == "auto":
        keys = [k for k in (cat, hue) if k is not None]
        biggest = data.groupby(keys, observed=True).size().max() if keys else len(data)
        kind = "swarm" if biggest <= swarm_limit else "strip"

    sign = None
    for sub, sub_center, color_kws in units:
        if sub.empty:
            continue
        base = dict(data=sub, x=x, y=y, order=order, orient=orient, ax=ax,
                    legend=False, **color_kws)

        # ------------------------------------------------------------ violin
        snap = _snapshot(ax)
        sns.violinplot(
            **base,
            width=violin_width * scale,
            inner=None,
            cut=cut,
            bw_adjust=bw_adjust,
            density_norm=density_norm,
            linewidth=linewidth,
            linecolor=linecolor,
            saturation=1,
            **(violin_kws or {}),
        )
        v_coll, v_lines, v_patch = _new_artists(ax, snap)

        if sign is None:
            # only now is the category axis final: seaborn inverts the y axis
            # for horizontal categoricals, so "up on screen" is -y there
            inverted = ax.yaxis_inverted() if cat_axis == "y" else ax.xaxis_inverted()
            sign = -1.0 if inverted else 1.0

        # sub-slot centres follow the category axis itself (first hue level
        # nearest the label), so no `sign` flip here -- only the layer offsets
        # need to know which way is "up" on screen
        d0 = sub_center
        _clip_to_half(v_coll, cat_axis, sign)
        for c in v_coll:
            c.set_alpha(violin_alpha)
        # the box overlaps the foot of the violin, so the stacking order is
        # pinned rather than left to insertion order
        for a in v_coll + v_lines + v_patch:
            a.set_zorder(2.0)
        _shift(v_coll + v_lines + v_patch,
               d0 + sign * violin_offset * scale, cat_axis, ax)

        # --------------------------------------------------------------- box
        snap = _snapshot(ax)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if box_kind in ("box", "boxplot"):
                sns.boxplot(
                    **base,
                    width=box_width * scale,
                    whis=whis,
                    showfliers=showfliers,
                    linewidth=linewidth,
                    linecolor=linecolor,
                    saturation=1,
                    **(box_kws or {}),
                )
            else:
                sns.boxenplot(
                    **base,
                    width=box_width * scale,
                    k_depth=k_depth,
                    showfliers=showfliers,
                    linewidth=linewidth,
                    linecolor=linecolor,
                    saturation=1,
                    **(box_kws or {}),
                )
        b_coll, b_lines, b_patch = _new_artists(ax, snap)
        # boxen fills are collections, boxplot fills are patches -- and alpha
        # must not touch the outlines, which are Line2D either way
        for c in b_coll + b_patch:
            c.set_alpha(box_alpha)
        for a in b_coll + b_lines + b_patch:
            a.set_zorder(2.4)
        _shift(b_coll + b_lines + b_patch,
               d0 + sign * box_offset * scale, cat_axis, ax)

        # ------------------------------------------------------------ points
        if kind != "none":
            snap = _snapshot(ax)
            # no `alpha=` here: a collection-level alpha would hit the stroke
            # too, and the fill/stroke opacities are set separately below
            pkws = dict(**base, size=point_size, linewidth=point_edge_width,
                        edgecolor="face", **(point_kws or {}))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # gutter / palette chatter
                if kind == "swarm":
                    with _beeswarm_width(point_width * scale):
                        sns.swarmplot(**pkws, warn_thresh=1.0)
                else:
                    # seaborn's `jitter` is a half-width in category units
                    jit = (point_width * scale) / 2 if jitter is None else jitter
                    sns.stripplot(**pkws, jitter=jit)
            p_coll, p_lines, p_patch = _new_artists(ax, snap)
            for c in p_coll:
                _split_point_alpha(c, point_alpha, point_edge_alpha,
                                   point_edgecolor)
            dp = d0 + sign * point_offset * scale
            _shift(p_coll, dp, cat_axis, ax, offsets_only=True)
            _shift(p_lines + p_patch, dp, cat_axis, ax)

        # ---------------------------------------------------------- mean dot
        if show_mean:
            mkws = dict(color="black", markersize=3.5, zorder=10)
            mkws.update(mean_kws or {})
            snap = _snapshot(ax)
            pp = dict(base)
            for k in ("hue", "hue_order", "palette", "color"):
                pp.pop(k, None)
            sns.pointplot(
                **pp,
                estimator="mean",
                errorbar=None,
                linestyle="none",
                marker="o",
                **mkws,
            )
            m_coll, m_lines, m_patch = _new_artists(ax, snap)
            _shift(m_coll + m_lines + m_patch,
                   d0 + sign * box_offset * scale, cat_axis, ax)

    if sign is None:  # nothing was drawn
        return ax

    # ------------------------------------------------------------ cosmetics
    # the shifted artists are invisible to autoscaling, so pin the cat axis
    n_cat = len(order) if order is not None else data[cat].nunique()
    lo, hi = -0.55, n_cat - 0.45
    setter = ax.set_ylim if cat_axis == "y" else ax.set_xlim
    setter((hi, lo) if inverted else (lo, hi))

    if legend and hue_levels is not None:
        colors = _palette_colors(palette, hue_levels)
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=linecolor,
                          linewidth=linewidth, alpha=violin_alpha)
            for c in colors
        ]
        ax.legend(handles, hue_levels, title=legend_title or hue, frameon=False)

    return ax


# --------------------------------------------------------------------------- #
# faceted version -- the full panel (a) of the reference figure
# --------------------------------------------------------------------------- #
def raincloud_grid(
    data,
    value_vars,
    group,
    *,
    orient="h",
    titles=None,
    palette=PALETTE,
    order=None,
    figsize=None,
    share_cat=True,
    value_labels=None,
    legend_loc=None,
    **kwargs,
):
    """One raincloud panel per column in `value_vars`, laid out in a row.

    `group` colours *and* positions the categories (each panel shows every
    group; the shared legend names them).

    orient : {"h", "v"}. ``"h"`` puts the values on x and stacks the groups as
        rows -- the reference-figure layout. ``"v"`` puts the values on y and
        lines the groups up as columns, with the clouds leaning right.
    value_labels : per-panel axis label for the value axis (units, usually).
    """
    n = len(value_vars)
    horizontal = orient in ("h", "y")
    titles = titles or list(value_vars)
    figsize = figsize or ((4.2 * n, 3.4) if horizontal else (3.3 * n, 4.4))
    fig, axes = plt.subplots(
        1, n, figsize=figsize,
        sharey=share_cat and horizontal,
        sharex=share_cat and not horizontal,
    )
    axes = np.atleast_1d(axes)

    for i, (axi, col) in enumerate(zip(axes, value_vars)):
        cat_kw = dict(x=col, y=group) if horizontal else dict(x=group, y=col)
        raincloud(
            data, **cat_kw, hue=group, order=order, hue_order=order,
            palette=palette, orient="h" if horizontal else "v", dodge=False,
            ax=axi, legend=False, **kwargs,
        )
        axi.set_title(titles[i], fontsize=10, fontweight="bold")
        label = value_labels[i] if value_labels else ""
        if horizontal:
            axi.set_xlabel(label)
            axi.set_ylabel("")
            if i and share_cat:
                axi.tick_params(labelleft=False)
        else:
            axi.set_ylabel(label)
            axi.set_xlabel("")

    levels = list(order) if order is not None else list(pd.unique(data[group].dropna()))
    lw = kwargs.get("linewidth", 0.5)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="0.15", linewidth=lw)
        for c in _palette_colors(palette, levels)
    ]
    # vertical panels already name the groups on the x axis, so the legend goes
    # to the side there instead of eating figure height under the labels
    loc = legend_loc or ("lower center" if horizontal else "center right")
    if loc == "lower center":
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        anchor, ncol = (0.5, 0.0), len(levels)
    else:
        fig.tight_layout(rect=(0, 0, 0.88, 1))
        anchor, ncol = (1.0, 0.5), 1
    fig.legend(handles, levels, title=group, ncol=ncol, frameon=False,
               loc=loc, bbox_to_anchor=anchor)
    return fig, axes


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def save_figure(fig, path, *, dpi=300, editable_text=True, transparent=False,
                bbox_inches="tight", **kwargs):
    """Save `fig`, keeping the text as text rather than outlines.

    matplotlib's SVG backend converts glyphs to paths by default, which makes
    labels uneditable in Illustrator / Inkscape / Affinity. ``svg.fonttype
    ="none"`` emits real ``<text>`` elements instead (the opening machine then
    needs the font -- stick to a common family such as Arial or Helvetica if
    the file has to travel). For PDF/EPS the equivalent is font type 42, which
    embeds a TrueType font and keeps the text selectable.

    >>> save_figure(fig, "figure1.svg")
    >>> save_figure(fig, "figure1.pdf")   # also editable
    """
    rc = {"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42} \
        if editable_text else {}
    with plt.rc_context(rc):
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches,
                    transparent=transparent, **kwargs)
    return path


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #
def _demo_data(seed=7, n=180):
    rng = np.random.default_rng(seed)
    rows = []
    spec = {  # group: (angle shape/scale, distance, duration)
        "Control": ((1.7, 9.0), (2.2, 0.42), (3.0, 260)),
        "Sham":    ((2.2, 11.0), (2.6, 0.40), (3.4, 270)),
        "Active":  ((2.6, 11.5), (3.0, 0.36), (3.2, 300)),
    }
    for g, (a, d, t) in spec.items():
        rows.append(pd.DataFrame({
            "Group": g,
            "Angle Deviation": rng.gamma(*a, n),
            "Distance traveled": rng.gamma(*d, n),
            "Duration": rng.gamma(*t, n) + 200,
        }))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"axes.edgecolor": "0.25", "axes.linewidth": 0.5,
                         "grid.linewidth": 0.5, "grid.color": "0.9",
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5})

    ORDER = ["Control", "Sham", "Active"]
    VARS = ["Angle Deviation", "Distance traveled", "Duration"]
    UNITS = ["degrees", "shoulder lengths", "ms"]
    df = _demo_data()

    # the docs/ copies are what the README embeds -- kept small and opaque so
    # they stay light in the repo and render predictably on GitHub
    DOCS_KW = dict(dpi=110, facecolor="white")

    fig, axes = raincloud_grid(df, VARS, "Group", order=ORDER,
                               value_labels=UNITS)
    save_figure(fig, "raincloud_demo.png", dpi=200)
    save_figure(fig, "raincloud_demo.svg")      # text stays editable
    save_figure(fig, "docs/example_horizontal.png", **DOCS_KW)

    fig_v, axes_v = raincloud_grid(df, VARS, "Group", orient="v", order=ORDER,
                                   value_labels=UNITS)
    save_figure(fig_v, "raincloud_demo_vertical.png", dpi=200)
    save_figure(fig_v, "docs/example_vertical.png", **DOCS_KW)

    # single axes, with a hue split
    df2 = df.assign(Session=np.resize(["Pre", "Post"], len(df)))
    fig2, ax = plt.subplots(figsize=(7, 4))
    raincloud(df2, x="Angle Deviation", y="Group", hue="Session",
              order=ORDER, hue_order=["Pre", "Post"],
              palette=[PALETTE[0], PALETTE[1]], ax=ax)
    save_figure(fig2, "raincloud_hue_demo.png", dpi=200)
    print("wrote raincloud_demo.png/.svg, raincloud_demo_vertical.png, "
          "raincloud_hue_demo.png and the docs/ copies")
