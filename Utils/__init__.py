import torch
import numpy as np
import pandas as pd
import random
import os
import pickle
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numba
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.colors
import matplotlib.cm
from matplotlib.patches import Patch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed=42):

    np.random.seed(seed)
    random.seed(seed)
    g = torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")

    return g


# Load raw dataset by channel:
def load_channel_raw(filepath):

    # Take columns five onward to include only the measurements.
    # Expand dims to include a channel dim (i.e., to form shape (N, C, L)):
    sigs = np.expand_dims(pd.read_excel(filepath).fillna(0).values[:, 5:], axis=1)

    return sigs


def get_base_metadata(dataset_name):

    trial_names = np.load('./Datasets/%s/objects/trial_names.npy' % dataset_name, allow_pickle=True)
    labels = np.load('./Datasets/%s/objects/labels.npy' % dataset_name, allow_pickle=True)
    ids = np.load('./Datasets/%s/objects/ids.npy' % dataset_name, allow_pickle=True)
    counts_samples = np.load('./Datasets/%s/objects/counts_samples.npy' % dataset_name, allow_pickle=True)

    return trial_names, labels, ids, counts_samples


def get_full_metadata(dataset_name):

    trial_names = np.load('./Datasets/%s/objects/trial_names.npy' % dataset_name, allow_pickle=True)
    labels = np.load('./Datasets/%s/objects/labels.npy' % dataset_name, allow_pickle=True)
    ids = np.load('./Datasets/%s/objects/ids.npy' % dataset_name, allow_pickle=True)
    counts_samples = np.load('./Datasets/%s/objects/counts_samples.npy' % dataset_name, allow_pickle=True)
    sexes = np.load('./Datasets/%s/objects/sexes.npy' % dataset_name, allow_pickle=True)
    ages = np.load('./Datasets/%s/objects/ages.npy' % dataset_name, allow_pickle=True)
    heights = np.load('./Datasets/%s/objects/heights.npy' % dataset_name, allow_pickle=True)
    masses = np.load('./Datasets/%s/objects/masses.npy' % dataset_name, allow_pickle=True)
    footwear = np.load('./Datasets/%s/objects/footwear.npy' % dataset_name, allow_pickle=True)
    speeds = np.load('./Datasets/%s/objects/speeds.npy' % dataset_name, allow_pickle=True)

    return trial_names, labels, ids, counts_samples, sexes, ages, heights, masses, footwear, speeds


def balance_ids_in_ds(trial_names, labels, ids, counts_samples, n_ids_new):

    indices_sort_counts_samples = np.argsort(counts_samples)[::-1]
    ids_sorted_balanced = ids[indices_sort_counts_samples][:n_ids_new]
    indices_sort_id_numbers = np.argsort(ids_sorted_balanced)
    ids_new = ids_sorted_balanced[indices_sort_id_numbers]
    counts_samples_new = counts_samples[indices_sort_counts_samples][:n_ids_new][indices_sort_id_numbers]
    indices_sort_labels = np.concatenate([np.asarray(labels == id_).nonzero()[0] for id_ in ids_new])
    trial_names_new = trial_names[indices_sort_labels]

    return trial_names_new, counts_samples_new, indices_sort_labels


# This function allocates IDs into training, validation, and test sets for five-fold cross-validation following a
# 60:20:20% distribution:
def gen_cv_folds(trial_names_by_id):

    indices = np.arange(trial_names_by_id.shape[0])
    n_folds = 5
    n_ids_te = int(np.round(trial_names_by_id.shape[0] / n_folds))

    trial_names_te = []
    trial_names_va = []
    trial_names_tr = []

    for i in range(n_folds):

        if i not in [n_folds - 2, n_folds - 1]:

            indices_te = np.arange((i + 0) * n_ids_te, (i + 1) * n_ids_te)
            indices_va = np.arange((i + 1) * n_ids_te, (i + 2) * n_ids_te)
            indices_tr = [idx for idx in indices if idx not in np.concatenate((indices_te, indices_va))]

            trial_names_te.append(list(trial_names_by_id[indices_te]))
            trial_names_va.append(list(trial_names_by_id[indices_va]))
            trial_names_tr.append(list(trial_names_by_id[indices_tr]))

        elif i == n_folds - 2:

            indices_te = np.arange((i + 0) * n_ids_te, (i + 1) * n_ids_te)
            indices_va = np.arange((i + 1) * n_ids_te, trial_names_by_id.shape[0])
            indices_tr = [idx for idx in indices if idx not in np.concatenate((indices_te, indices_va))]

            trial_names_te.append(list(trial_names_by_id[indices_te]))
            trial_names_va.append(list(trial_names_by_id[indices_va]))
            trial_names_tr.append(list(trial_names_by_id[indices_tr]))

        else:

            indices_te = np.arange((i + 0) * n_ids_te, trial_names_by_id.shape[0])
            indices_va = np.arange(0 * n_ids_te, 1 * n_ids_te)
            indices_tr = [idx for idx in indices if idx not in np.concatenate((indices_te, indices_va))]

            trial_names_te.append(list(trial_names_by_id[indices_te]))
            trial_names_va.append(list(trial_names_by_id[indices_va]))
            trial_names_tr.append(list(trial_names_by_id[indices_tr]))

    return trial_names_tr, trial_names_va, trial_names_te


def save_list(filepath, obj):
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp)


def load_list(filepath):
    with open(filepath, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


le = preprocessing.LabelEncoder()


def get_unique_unsorted(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


# Below are functions and classes from the UMAP package...
# The following have not been edited:
# ---------------------------------------------------------------------
def _select_font_color(background):
    if background == "black":
        font_color = "white"
    elif background.startswith("#"):
        mean_val = np.mean(
            [int("0x" + c) for c in (background[1:3], background[3:5], background[5:7])]
        )
        if mean_val > 126:
            font_color = "black"
        else:
            font_color = "white"

    else:
        font_color = "black"

    return font_color


def _get_extent(points):
    """Compute bounds on a space with appropriate padding"""
    min_x = np.nanmin(points[:, 0])
    max_x = np.nanmax(points[:, 0])
    min_y = np.nanmin(points[:, 1])
    max_y = np.nanmax(points[:, 1])

    extent = (
        np.round(min_x - 0.05 * (max_x - min_x)),
        np.round(max_x + 0.05 * (max_x - min_x)),
        np.round(min_y - 0.05 * (max_y - min_y)),
        np.round(max_y + 0.05 * (max_y - min_y)),
    )

    return extent


def _get_metric(umap_object):
    if hasattr(umap_object, "metric"):
        return umap_object.metric
    else:
        # Assume euclidean if no attribute per cuML.UMAP
        return "euclidean"


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _red(x):
    return (x & 0xFF0000) >> 16


@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _green(x):
    return (x & 0x00FF00) >> 8


@numba.vectorize(["uint8(uint32)", "uint8(uint32)"])
def _blue(x):
    return x & 0x0000FF


def _embed_datashader_in_an_axis(datashader_image, ax):
    img_rev = datashader_image.data[::-1]
    mpl_img = np.dstack([_blue(img_rev), _green(img_rev), _red(img_rev)])
    ax.imshow(mpl_img)
    return ax
# ----------------------------------------------------------------------------


# The following have minor changes in the places indicated by double comments:
# ------------------------------------------------------------------------------
def _matplotlib_points(
    points,
    figure=None, # # Added this
    ax=None,
    labels=None,
    values=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="Spectral",
    background="white",
    width=800,
    height=800,
    show_legend=True,
    alpha=None,
    attr='Datasets'
):
    """Use matplotlib to plot points"""
    point_size = 100.0 / np.sqrt(points.shape[0])

    legend_elements = None

    if ax is None:
        dpi = plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(width / dpi, height / dpi))
        ax = fig.add_subplot(111)

    ax.set_facecolor(background)

    # Color by labels
    if labels is not None:
        if labels.shape[0] != points.shape[0]:
            raise ValueError(
                "Labels must have a label for "
                "each sample (size mismatch: {} {})".format(
                    labels.shape[0], points.shape[0]
                )
            )
        if color_key is None:
            unique_labels = np.unique(labels)
            num_labels = unique_labels.shape[0]
            color_key = plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
            legend_elements = [
                Patch(facecolor=color_key[i], label=unique_labels[i])
                for i, k in enumerate(unique_labels)
            ]

        if isinstance(color_key, dict):
            colors = pd.Series(labels).map(color_key)

            # # Added this conditional:
            if attr == 'Datasets':

                unique_labels = get_unique_unsorted(labels)

            else:

                unique_labels = np.unique(labels)

            legend_elements = [
                Patch(facecolor=color_key[k], label=k) for k in unique_labels
            ]
        else:
            unique_labels = np.unique(labels)

            if len(color_key) < unique_labels.shape[0]:
                raise ValueError(
                    "Color key must have enough colors for the number of labels"
                )

            new_color_key = {
                k: matplotlib.colors.to_hex(color_key[i])
                for i, k in enumerate(unique_labels)
            }
            legend_elements = [
                Patch(facecolor=color_key[i], label=k)
                for i, k in enumerate(unique_labels)
            ]
            colors = pd.Series(labels).map(new_color_key)

        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=colors, alpha=alpha)

    # Color by values
    elif values is not None:
        if values.shape[0] != points.shape[0]:
            raise ValueError(
                "Values must have a value for "
                "each sample (size mismatch: {} {})".format(
                    values.shape[0], points.shape[0]
                )
            )
        ax.scatter(
            points[:, 0], points[:, 1], s=point_size, c=values, cmap=cmap, alpha=alpha
        )

    # No color (just pick the midpoint of the cmap)
    else:

        color = plt.get_cmap(cmap)(0.5)
        ax.scatter(points[:, 0], points[:, 1], s=point_size, c=color)

    if show_legend and legend_elements is not None:

        # # Think this was edited:
        ax.legend(handles=legend_elements, prop={'size': 5})

    return ax


# # There were changes made to this function but the _matplotlib_points one above get used anyway:
def _datashade_points(
    points,
    figure=None, # # Added this.
    ax=None,
    labels=None,
    values=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="Spectral",
    background="white",
    show_legend=True,
    alpha=255,
):

    """Use datashader to plot points"""
    extent = _get_extent(points)

    # # Added this
    bbox = ax.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width = int(width * figure.dpi)
    height = int(height * figure.dpi)

    canvas = ds.Canvas(
        plot_width=width,
        plot_height=height,
        x_range=(extent[0], extent[1]),
        y_range=(extent[2], extent[3]),
    )
    data = pd.DataFrame(points, columns=("x", "y"))

    legend_elements = None

    # Color by labels
    if labels is not None:
        if labels.shape[0] != points.shape[0]:
            raise ValueError(
                "Labels must have a label for "
                "each sample (size mismatch: {} {})".format(
                    labels.shape[0], points.shape[0]
                )
            )

        data["label"] = pd.Categorical(labels)
        aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("label"))
        if color_key is None and color_key_cmap is None:
            result = tf.shade(aggregation, how="eq_hist", alpha=alpha)
        elif color_key is None:
            unique_labels = np.unique(labels)
            num_labels = unique_labels.shape[0]
            color_key = _to_hex(
                plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
            )
            legend_elements = [
                Patch(facecolor=color_key[i], label=k)
                for i, k in enumerate(unique_labels)
            ]
            result = tf.shade(
                aggregation, color_key=color_key, how="eq_hist", alpha=alpha
            )
        else:
            legend_elements = [
                Patch(facecolor=color_key[k], label=k) for k in color_key.keys()
            ]
            result = tf.shade(
                aggregation, color_key=color_key, how="eq_hist", alpha=alpha
            )

    # Color by values
    elif values is not None:
        if values.shape[0] != points.shape[0]:
            raise ValueError(
                "Values must have a value for "
                "each sample (size mismatch: {} {})".format(
                    values.shape[0], points.shape[0]
                )
            )
        unique_values = np.unique(values)
        if unique_values.shape[0] >= 256:
            min_val, max_val = np.min(values), np.max(values)
            bin_size = (max_val - min_val) / 255.0
            data["val_cat"] = pd.Categorical(
                np.round((values - min_val) / bin_size).astype(np.int16)
            )
            aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("val_cat"))
            color_key = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, 256)))
            result = tf.shade(
                aggregation, color_key=color_key, how="eq_hist", alpha=alpha
            )
        else:
            data["val_cat"] = pd.Categorical(values)
            aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("val_cat"))
            color_key_cols = _to_hex(
                plt.get_cmap(cmap)(np.linspace(0, 1, unique_values.shape[0]))
            )
            color_key = dict(zip(unique_values, color_key_cols))
            result = tf.shade(
                aggregation, color_key=color_key, how="eq_hist", alpha=alpha
            )

    # Color by density (default datashader option)
    else:
        aggregation = canvas.points(data, "x", "y", agg=ds.count())
        result = tf.shade(aggregation, cmap=plt.get_cmap(cmap), alpha=alpha)

    if background is not None:
        result = tf.set_background(result, background)

    if ax is not None:
        _embed_datashader_in_an_axis(result, ax)
        if show_legend and legend_elements is not None:

            # # Think this was edited:
            ax.legend(handles=legend_elements, prop={'size': 6})
        return ax
    else:
        return result


def points(
    umap_object,
    figure=None, # # Added this.
    points=None,
    labels=None,
    values=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="Spectral",
    background="white",
    width=800,
    height=800,
    show_legend=True,
    subset_points=None,
    ax=None,
    alpha=None,
    attr='Datasets' # # Added this.
):
    """Plot an embedding as points. Currently this only works
    for 2D embeddings. While there are many optional parameters
    to further control and tailor the plotting, you need only
    pass in the trained/fit umap model to get results. This plot
    utility will attempt to do the hard work of avoiding
    over-plotting issues, and make it easy to automatically
    colour points by a categorical labelling or numeric values.
    This method is intended to be used within a Jupyter
    notebook with ``%matplotlib inline``.
    Parameters
    ----------
    umap_object: trained UMAP object
        A trained UMAP object that has a 2D embedding.
    points: array, shape (n_samples, dim) (optional, default None)
        An array of points to be plotted. Usually this is None
        and so the original embedding points of the umap_object
        are used. However points can be passed explicitly instead
        which is useful for points manually transformed.
    labels: array, shape (n_samples,) (optional, default None)
        An array of labels (assumed integer or categorical),
        one for each data sample.
        This will be used for coloring the points in
        the plot according to their label. Note that
        this option is mutually exclusive to the ``values``
        option.
    values: array, shape (n_samples,) (optional, default None)
        An array of values (assumed float or continuous),
        one for each sample.
        This will be used for coloring the points in
        the plot according to a colorscale associated
        to the total range of values. Note that this
        option is mutually exclusive to the ``labels``
        option.
    theme: string (optional, default None)
        A color theme to use for plotting. A small set of
        predefined themes are provided which have relatively
        good aesthetics. Available themes are:
           * 'blue'
           * 'red'
           * 'green'
           * 'inferno'
           * 'fire'
           * 'viridis'
           * 'darkblue'
           * 'darkred'
           * 'darkgreen'
    cmap: string (optional, default 'Blues')
        The name of a matplotlib colormap to use for coloring
        or shading points. If no labels or values are passed
        this will be used for shading points according to
        density (largely only of relevance for very large
        datasets). If values are passed this will be used for
        shading according the value. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.
    color_key: dict or array, shape (n_categories) (optional, default None)
        A way to assign colors to categoricals. This can either be
        an explicit dict mapping labels to colors (as strings of form
        '#RRGGBB'), or an array like object providing one color for
        each distinct category being provided in ``labels``. Either
        way this mapping will be used to color points according to
        the label. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.
    color_key_cmap: string (optional, default 'Spectral')
        The name of a matplotlib colormap to use for categorical coloring.
        If an explicit ``color_key`` is not given a color mapping for
        categories can be generated from the label list and selecting
        a matching list of colors from the given colormap. Note
        that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.
    background: string (optional, default 'white)
        The color of the background. Usually this will be either
        'white' or 'black', but any color name will work. Ideally
        one wants to match this appropriately to the colors being
        used for points etc. This is one of the things that themes
        handle for you. Note that if theme
        is passed then this value will be overridden by the
        corresponding option of the theme.
    width: int (optional, default 800)
        The desired width of the plot in pixels.
    height: int (optional, default 800)
        The desired height of the plot in pixels
    show_legend: bool (optional, default True)
        Whether to display a legend of the labels
    subset_points: array, shape (n_samples,) (optional, default None)
        A way to select a subset of points based on an array of boolean
        values.
    ax: matplotlib axis (optional, default None)
        The matplotlib axis to draw the plot to, or if None, which is
        the default, a new axis will be created and returned.
    alpha: float (optional, default: None)
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    Returns
    -------
    result: matplotlib axis
        The result is a matplotlib axis with the relevant plot displayed.
        If you are using a notebooks and have ``%matplotlib inline`` set
        then this will simply display inline.
    """

    # # Think I commented these out:
    # if not hasattr(umap_object, "embedding_"):
    #     raise ValueError(
    #         "UMAP object must perform fit on data before it can be visualized"
    #     )

    # if theme is not None:
    #     cmap = _themes[theme]["cmap"]
    #     color_key_cmap = _themes[theme]["color_key_cmap"]
    #     background = _themes[theme]["background"]

    if labels is not None and values is not None:
        raise ValueError(
            "Conflicting options; only one of labels or values should be set"
        )

    if alpha is not None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0 and 1 inclusive")

    # # Commented this out:
    # if points is None:
    #     points = _get_embedding(umap_object)

    if subset_points is not None:
        if len(subset_points) != points.shape[0]:
            raise ValueError(
                "Size of subset points ({}) does not match number of input points ({})".format(
                    len(subset_points), points.shape[0]
                )
            )
        points = points[subset_points]

        if labels is not None:
            labels = labels[subset_points]
        if values is not None:
            values = values[subset_points]

    if points.shape[1] != 2:
        raise ValueError("Plotting is currently only implemented for 2D embeddings")

    font_color = _select_font_color(background)

    # # Commented this out:
    # if ax is None:
    #     dpi = plt.rcParams["figure.dpi"]
    #     fig = plt.figure(figsize=(width / dpi, height / dpi))
    #     ax = fig.add_subplot(111)

    if points.shape[0] <= width * height // 10: # # This condition gets met.
        ax = _matplotlib_points(
            points,
            figure,
            ax,
            labels,
            values,
            cmap,
            color_key,
            color_key_cmap,
            background,
            width,
            height,
            show_legend,
            alpha,
            attr # # Added this in.
        )
    else:

        # Datashader uses 0-255 as the range for alpha, with 255 as the default
        if alpha is not None:
            alpha = alpha * 255
        else:
            alpha = 255

        ax = _datashade_points(

            points,
            figure,
            ax,
            labels,
            values,
            cmap,
            color_key,
            color_key_cmap,
            background,
            show_legend,
            alpha,
        )

    ax.set(xticks=[], yticks=[])

    # # Commented this out:
    # if _get_metric(umap_object) != "euclidean":
    #     ax.text(
    #         0.99,
    #         0.01,
    #         "UMAP: metric={}, n_neighbors={}, min_dist={}".format(
    #             _get_metric(umap_object), umap_object.n_neighbors, umap_object.min_dist
    #         ),
    #         transform=ax.transAxes,
    #         horizontalalignment="right",
    #         color=font_color,
    #     )
    # else:
    #     ax.text(
    #         0.99,
    #         0.01,
    #         "UMAP: n_neighbors={}, min_dist={}".format(
    #             umap_object.n_neighbors, umap_object.min_dist
    #         ),
    #         transform=ax.transAxes,
    #         horizontalalignment="right",
    #         color=font_color,
    #     )

    return ax
# ----------------------------------------------------------------------------
