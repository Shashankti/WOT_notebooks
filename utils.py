#Utility functions for the plotting and lineage tracing


import os
import time
from logging import raiseExceptions

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import scipy.stats as stats
import statsmodels.sandbox.stats.multicomp
from ete3 import Tree
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy

# from plotnine import *
from sklearn.manifold import SpectralEmbedding
import cospar as cs

def sparse_rowwise_multiply(E, a):
    """
    Multiply each row of the sparse matrix E by a scalar a

    Parameters
    ----------
    E: `np.array` or `sp.spmatrix`
    a: `np.array`
        A scalar vector.

    Returns
    
    -------
    Rescaled sparse matrix
    """

    nrow = E.shape[0]
    if nrow != a.shape[0]:
#        logg.error("Dimension mismatch, multiplication failed")
        return E
    else:
        w = ssp.lil_matrix((nrow, nrow))
        w.setdiag(a)
        return w * E
def analyze_selected_fates(state_info, selected_fates):
    """
    Analyze selected fates.

    We return only valid fate clusters.

    Parameters
    ----------
    selected_fates: `list`
        List of selected fate clusters.
    state_info: `list`
        The state_info vector.

    Returns
    -------
    mega_cluster_list: `list`, shape (n_fate)
        The list of names for the mega cluster. This is relevant when
        `selected_fates` has a nested structure.
    valid_fate_list: `list`, shape (n_fate)
        It is the same as selected_fates, could contain a nested list
        of fate clusters. It screens for valid fates, though.
    fate_array_flat: `list` shape (>n_fate)
        List of all selected fate clusters. It flattens the selected_fates if it contains
        nested structure, and allows only valid clusters.
    sel_index_list: `list`, shape (n_fate)
        List of selected cell indexes for each merged cluster.
    """

    state_info = np.array(state_info)
    valid_state_annot = list(set(state_info))
    if type(selected_fates) is str:
        selected_fates = [selected_fates]
    if selected_fates is None:
        selected_fates = valid_state_annot

    fate_array_flat = []  # a flatten list of cluster names
    valid_fate_list = (
        []
    )  # a list of cluster lists, each cluster list is a macro cluster
    mega_cluster_list = []  # a list of string description for the macro cluster
    sel_index_list = []
    for xx in selected_fates:
        if type(xx) is list:
            valid_fate_list.append(xx)
            des_temp = ""
            temp_idx = np.zeros(len(state_info), dtype=bool)
            for zz in xx:
                if zz in valid_state_annot:
                    fate_array_flat.append(zz)
                    if des_temp != "":
                        des_temp = des_temp + "_"

                    des_temp = des_temp + str(zz)
                    temp_idx = temp_idx | (state_info == zz)
                else:
                    raise ValueError(
                        f"{zz} is not a valid cluster name. Please select from: {valid_state_annot}"
                    )
            mega_cluster_list.append(des_temp)
            sel_index_list.append(temp_idx)
        else:
            if xx in valid_state_annot:
                valid_fate_list.append([xx])

                fate_array_flat.append(xx)
                mega_cluster_list.append(str(xx))
            else:
                raise ValueError(
                    f"{xx} is not a valid cluster name. Please select from: {valid_state_annot}"
                )
                mega_cluster_list.append("")

            temp_idx = state_info == xx
            sel_index_list.append(temp_idx)

    # exclude invalid clusters
    mega_cluster_list = np.array(mega_cluster_list)
    fate_array_flat = np.array(fate_array_flat)
    sel_index_list = np.array(sel_index_list)
    valid_idx = mega_cluster_list != ""

    return (
        mega_cluster_list[valid_idx],
        valid_fate_list,
        fate_array_flat,
        sel_index_list[valid_idx],
    )
def sparse_column_multiply(E, a):
    """
    Multiply each columns of the sparse matrix E by a scalar a

    Parameters
    ----------
    E: `np.array` or `sp.spmatrix`
    a: `np.array`
        A scalar vector.

    Returns
    -------
    Rescaled sparse matrix
    """

    ncol = E.shape[1]
    if ncol != a.shape[0]:
#        logg.error("Dimension mismatch, multiplication failed")
        return E
    else:
        w = ssp.lil_matrix((ncol, ncol))
        w.setdiag(a)
        return ssp.csr_matrix(E) * w

#custom functions for singe cell fate 
def customized_embedding(
    x,
    y,
    vector,
    title=None,
    ax=None,
    order_points=True,
    set_ticks=False,
    col_range=None,
    buffer_pct=0.03,
    point_size=1,
    color_map=None,
    smooth_operator=None,
    set_lim=True,
    vmax=None,
    vmin=None,
    color_bar=False,
    color_bar_label="",
    color_bar_title="",
):
    """
    Plot a vector on an embedding.

    Parameters
    ----------
    x: `np.array`
        x coordinate of the embedding
    y: `np.array`
        y coordinate of the embedding
    vector: `np.array`
        A vector to be plotted.
    color_map: {plt.cm.Reds,plt.cm.Blues,...}, (default: None)
    ax: `axis`, optional (default: None)
        An external ax object can be passed here.
    order_points: `bool`, optional (default: True)
        Order points to plot by the gene expression
    col_range: `tuple`, optional (default: None)
        The default setting is to plot the actual value of the vector.
        If col_range is set within [0,100], it will plot the percentile of the values,
        and the color_bar will show range [0,1]. This re-scaling is useful for
        visualizing gene expression.
    buffer_pct: `float`, optional (default: 0.03)
        Extra space for the plot box frame
    point_size: `int`, optional (default: 1)
        Size of the data point
    smooth_operator: `np.array`, optional (default: None)
        A smooth matrix to be applied to the subsect of gene expression matrix.
    set_lim: `bool`, optional (default: True)
        Set the plot range (x_limit, and y_limit) automatically.
    vmax: `float`, optional (default: np.nan)
        Maximum color range (saturation).
        All values above this will be set as vmax.
    vmin: `float`, optional (default: np.nan)
        The minimum color range, all values below this will be set to be vmin.
    color_bar: `bool`, optional (default, False)
        If True, plot the color bar.
    set_ticks: `bool`, optional (default, False)
        If False, remove figure ticks.

    Returns
    -------
    ax:
        The figure axis
    """

    from matplotlib.colors import Normalize as mpl_Normalize

    if color_map is None:
        color_map = darken_cmap(plt.cm.Reds, 0.9)
    if ax is None:
        fig, ax = plt.subplots()

    coldat = vector.astype(float)

    if smooth_operator is None:
        coldat = coldat.squeeze()
    else:
        coldat = np.dot(smooth_operator, coldat).squeeze()

    if order_points:
        o = np.argsort(coldat)
    else:
        o = np.arange(len(coldat))

    if vmin is None:
        if col_range is None:
            vmin = np.min(coldat)
        else:
            vmin = np.percentile(coldat, col_range[0])

    if vmax is None:
        if col_range is None:
            vmax = np.max(coldat)
        else:
            vmax = np.percentile(coldat, col_range[1])

    if vmax == vmin:
        vmax = coldat.max()

    ax.scatter(
        x[o], y[o], c=coldat[o], s=point_size, cmap=color_map, vmin=vmin, vmax=vmax
    )

    if not set_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    if set_lim == True:
        ax.set_xlim(x.min() - x.ptp() * buffer_pct, x.max() + x.ptp() * buffer_pct)
        ax.set_ylim(y.min() - y.ptp() * buffer_pct, y.max() + y.ptp() * buffer_pct)

    if title is not None:
        ax.set_title(title)

    if color_bar:

        norm = mpl_Normalize(vmin=vmin, vmax=vmax)
        Clb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax)
        Clb.set_label(
            color_bar_label,
            rotation=270,
            labelpad=20,
        )
        Clb.ax.set_title(color_bar_title)
    return ax

def darken_cmap(cmap, scale_factor):
    """
    Generate a gradient color map for plotting.
    """

    cdat = np.zeros((cmap.N, 4))
    for ii in range(cdat.shape[0]):
        curcol = cmap(ii)
        cdat[ii, 0] = curcol[0] * scale_factor
        cdat[ii, 1] = curcol[1] * scale_factor
        cdat[ii, 2] = curcol[2] * scale_factor
        cdat[ii, 3] = 1
    cmap = cmap.from_list(cmap.N, cdat)
    return cmap

#Main function
def single_cell_transition(
    adata,
    df,
    selected_state_id_list,
    source="transition_map",
    map_backward=True,
    initial_point_size=3,
    color_bar=True,
    savefig=False):
    
    fig_width=5.5
    fig_height=5
    point_size=3
    state_info=ot_adata.obs["state_info"]
    x_emb = ot_adata.obsm["X_emb"][:,0]
    #x_emb = ot_adata[ot_adata.obs['time']==0].obsm["X_emb"][:,0]
    y_emb = ot_adata.obsm["X_emb"][:,1]
    #y_emb = ot_adata[ot_adata.obs['time']==0].obsm["X_emb"][:,1]
    if not map_backward:
        cell_id_t1 = np.array(list(df.index))
        cell_id_t1 = cell_id_t1.astype(int)
        cell_id_t2 = np.array(list(df.columns))
        cell_id_t2 = cell_id_t2.astype(int)
        Tmap = source
    else:
        cell_id_t2 = np.array(list(df.index))
        cell_id_t2 = cell_id_t2.astype(int)
        cell_id_t1 = np.array(list(df.columns))
        cell_id_t1 = cell_id_t1.astype(int)
        Tmap = source.T
    row = len(selected_state_id_list)
    col = 1
    selected_state_id_list = np.array(selected_state_id_list)
    full_id_list = np.arange(len(cell_id_t1))
    valid_idx = np.in1d(full_id_list, selected_state_id_list)
    if np.sum(valid_idx) < len(selected_state_id_list):
        selected_state_id_list = full_id_list[valid_idx]
    row = len(selected_state_id_list)
    col = 1
    fig = plt.figure(figsize=(fig_width* col, fig_height * row))
    for j, target_cell_ID in enumerate(selected_state_id_list):
        ax0 = plt.subplot(row, col, col * j +1)
        if target_cell_ID < Tmap.shape[0]:
            prob_vec = np.zeros(len(x_emb))
            prob_vec[cell_id_t2] = Tmap[target_cell_ID, :]
            prob_vec = prob_vec / np.max(prob_vec)
            customized_embedding(
                x_emb,
                y_emb,
                prob_vec,
                point_size=point_size,
                ax=ax0,
                color_bar=True,
                color_bar_label="Probability")        
            ax0.plot(
                x_emb[cell_id_t1][target_cell_ID],
                y_emb[cell_id_t1][target_cell_ID],
                "*b",
                markersize=initial_point_size * point_size)
            if map_backward:
                ax0.set_title(f"ID (t2): {target_cell_ID}")
            else:
                ax0.set_title(f"ID (t1): {target_cell_ID}")
    plt.tight_layout()
    #change the file name here
    if savefig:
        fig.savefig(
            os.path.join(
                 "/fast/AG_Haghverdi/Shashank/fig_cospar/",
                 f"_single_cell_transition_{map_backward}.png",
            )
        )
    

## Helper functions

def analyze_selected_fates(state_info, selected_fates):
    """
    Analyze selected fates.

    We return only valid fate clusters.

    Parameters
    ----------
    selected_fates: `list`
        List of selected fate clusters.
    state_info: `list`
        The state_info vector.

    Returns
    -------
    mega_cluster_list: `list`, shape (n_fate)
        The list of names for the mega cluster. This is relevant when
        `selected_fates` has a nested structure.
    valid_fate_list: `list`, shape (n_fate)
        It is the same as selected_fates, could contain a nested list
        of fate clusters. It screens for valid fates, though.
    fate_array_flat: `list` shape (>n_fate)
        List of all selected fate clusters. It flattens the selected_fates if it contains
        nested structure, and allows only valid clusters.
    sel_index_list: `list`, shape (n_fate)
        List of selected cell indexes for each merged cluster.
    """

    state_info = np.array(state_info)
    valid_state_annot = list(set(state_info))
    if type(selected_fates) is str:
        selected_fates = [selected_fates]
    if selected_fates is None:
        selected_fates = valid_state_annot

    fate_array_flat = []  # a flatten list of cluster names
    valid_fate_list = (
        []
    )  # a list of cluster lists, each cluster list is a macro cluster
    mega_cluster_list = []  # a list of string description for the macro cluster
    sel_index_list = []
    for xx in selected_fates:
        if type(xx) is list:
            valid_fate_list.append(xx)
            des_temp = ""
            temp_idx = np.zeros(len(state_info), dtype=bool)
            for zz in xx:
                if zz in valid_state_annot:
                    fate_array_flat.append(zz)
                    if des_temp != "":
                        des_temp = des_temp + "_"

                    des_temp = des_temp + str(zz)
                    temp_idx = temp_idx | (state_info == zz)
                else:
                    raise ValueError(
                        f"{zz} is not a valid cluster name. Please select from: {valid_state_annot}"
                    )
            mega_cluster_list.append(des_temp)
            sel_index_list.append(temp_idx)
        else:
            if xx in valid_state_annot:
                valid_fate_list.append([xx])

                fate_array_flat.append(xx)
                mega_cluster_list.append(str(xx))
            else:
                raise ValueError(
                    f"{xx} is not a valid cluster name. Please select from: {valid_state_annot}"
                )
                mega_cluster_list.append("")

            temp_idx = state_info == xx
            sel_index_list.append(temp_idx)

    # exclude invalid clusters
    mega_cluster_list = np.array(mega_cluster_list)
    fate_array_flat = np.array(fate_array_flat)
    sel_index_list = np.array(sel_index_list)
    valid_idx = mega_cluster_list != ""

    return (
        mega_cluster_list[valid_idx],
        valid_fate_list,
        fate_array_flat,
        sel_index_list[valid_idx],
    )

###
def parse_output_choices(adata, key_word, where="obs", interrupt=True):
    if where == "obs":
        raw_choices = [xx for xx in adata.obs.keys() if xx.startswith(f"{key_word}")]
    else:
        raw_choices = [xx for xx in adata.uns.keys() if xx.startswith(f"{key_word}")]

    if (interrupt) and (len(raw_choices) == 0):
        raise ValueError(
            f"{key_word} has not been computed yet. Please run the counterpart function at cs.tl.XXX using the appropriate source name."
        )

    available_choices = []
    for xx in raw_choices:
        y = xx.split(f"{key_word}")[1]
        if y.startswith("_"):
            y = y[1:]
        available_choices.append(y)

    return available_choices
#========================================================#
def selecting_cells_by_time_points(time_info, selected_time_points):
    """
    Check validity of selected time points, and return the selected index.

    selected_time_points can be either a string or a list.

    If selected_time_points=[], we select all cell states.
    """

    time_info = np.array(time_info)
    valid_time_points = set(time_info)
    if selected_time_points is not None:
        if type(selected_time_points) is str:
            selected_times = [selected_time_points]
        else:
            selected_times = list(selected_time_points)

        sp_idx = np.zeros(len(time_info), dtype=bool)
        for xx in selected_times:
            if xx not in valid_time_points:
                logg.error(f"{xx} is not a valid time point.")
            sp_id_temp = np.nonzero(time_info == xx)[0]
            sp_idx[sp_id_temp] = True
    else:
        sp_idx = np.ones(len(time_info), dtype=bool)

    return sp_idx

def compute_fate_probability_map(
    adata,
    selected_fates=None,
    used_Tmap="transition_map",
    map_backward=True,
    method="norm-sum",
    fate_count=True,
):
    """
    Compute fate map and the relative bias compared to the expectation.

    `selected_fates` could contain a nested list of clusters. If so, we combine each sub-list
    into a mega-fate cluster and compute the fate map correspondingly.

    The relative bias is obtained by comparing the fate_prob with the
    expected_prob from the relative size of the targeted cluster. It ranges from [0,1],
    with 0.5 being the point that the fate_prob agrees with expected_prob.
    1 is extremely biased.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`, optional (default: all)
        List of targeted clusters, consistent with adata.obs['state_info'].
        If set to be None, use all fate clusters in adata.obs['state_info'].
    used_Tmap: `str`
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map','weinreb_transition_map','naive_transition_map',
        'OT_transition_map','HighVar_transition_map'}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, compute for initial cell states (rows of Tmap, at t1);
        else, compute for later cell states (columns of Tmap, at t2)
    method: `str`, optional (default: 'norm-sum')
        Method to aggregate the transition probability within a cluster. Available options: {'sum','mean','max','norm-sum'},
        which computes the sum, mean, or max of transition probability within a cluster as the final fate probability.
    fate_count: `bool`, optional (default: True)
        Used to determine the method for computing the fate potential of a state.
        If ture, jus to count the number of possible fates; otherwise, use the Shannon entropy.

    Returns
    -------
    Store `fate_array`, `fate_map`, `fate_entropy` in adata.uns['fate_map'].

    fate_map: `np.array`, shape (n_cell, n_fate)
        n_fate is the number of mega cluster, equals len(selected_fates).
    mega_cluster_list: `list`, shape (n_fate)
        The list of names for the mega cluster. This is relevant when
        `selected_fates` has a nested structure.
    relative_bias: `np.array`, shape (n_cell, n_fate)
    expected_prob: `np.array`, shape (n_fate,)
    valid_fate_list: `list`, shape (n_fate)
        It is the same as selected_fates, could contain a nested list
        of fate clusters. It screens for valid fates, though.
    """

#    hf.check_available_map(adata)
    if method not in ["max", "sum", "mean", "norm-sum"]:
        logg.warn(
            "method not in {'max','sum','mean','norm-sum'}; use the 'norm-sum' method"
        )
        method = "norm-sum"

    if map_backward:
        cell_id_t2 = adata.uns["Tmap_cell_id_t2"]
    else:
        cell_id_t2 = adata.uns["Tmap_cell_id_t1"]

    state_annote = adata.obs["state_info"]
    if selected_fates is None:
        selected_fates = list(set(state_annote))
    (
        mega_cluster_list,
        valid_fate_list,
        fate_array_flat,
        sel_index_list,
    ) = analyze_selected_fates(state_annote, selected_fates)

    state_annote_0 = np.array(adata.obs["state_info"])
    if map_backward:
        cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
        cell_id_t2 = adata.uns["Tmap_cell_id_t2"]

    else:
        cell_id_t2 = adata.uns["Tmap_cell_id_t1"]
        cell_id_t1 = adata.uns["Tmap_cell_id_t2"]

    x_emb = adata.obsm["X_emb"][:, 0]
    y_emb = adata.obsm["X_emb"][:, 1]
    data_des = adata.uns["data_des"][-1]

    state_annote_1 = state_annote_0.copy()
    for j1, new_cluster_id in enumerate(mega_cluster_list):
        idx = np.in1d(state_annote_0, valid_fate_list[j1])
        state_annote_1[idx] = new_cluster_id

    state_annote_BW = state_annote_1[cell_id_t2]

    if used_Tmap in adata.uns["available_map"]:
        used_map = adata.uns[used_Tmap]

        fate_map, fate_entropy = compute_state_potential(
            used_map,
            state_annote_BW,
            mega_cluster_list,
            fate_count=fate_count,
            map_backward=map_backward,
            method=method,
        )

    else:
        raise ValueError(f"used_Tmap should be among {adata.uns['available_map']}")

    # Note: we compute relative_bias (normalze against cluster size). This is no longer in active use
    N_macro = len(valid_fate_list)
    relative_bias = np.zeros((fate_map.shape[0], N_macro))
    expected_prob = np.zeros(N_macro)
    for jj in range(N_macro):
        for yy in valid_fate_list[jj]:
            expected_prob[jj] = expected_prob[jj] + np.sum(
                state_annote[cell_id_t2] == yy
            ) / len(cell_id_t2)

        # transformation, this is useful only when the method =='sum'
        temp_idx = fate_map[:, jj] < expected_prob[jj]
        temp_diff = fate_map[:, jj] - expected_prob[jj]
        relative_bias[temp_idx, jj] = temp_diff[temp_idx] / expected_prob[jj]
        relative_bias[~temp_idx, jj] = temp_diff[~temp_idx] / (1 - expected_prob[jj])

        relative_bias[:, jj] = (
            relative_bias[:, jj] + 1
        ) / 2  # rescale to the range [0,1]

    return (
        fate_map,
        mega_cluster_list,
        relative_bias,
        expected_prob,
        valid_fate_list,
        sel_index_list,
        fate_entropy,
    )


def compute_state_potential(
    transition_map,
    state_annote,
    fate_array,
    fate_count=False,
    map_backward=True,
    method="sum",
):
    """
    Compute state probability towards/from given clusters

    Before any calculation, we row-normalize the transition map.
    If map_backward=True, compute the fate map towards given
    clusters. Otherwise, compute the ancestor map, the probabilities
    of a state to originate from given clusters.

    Parameters
    ----------
    transition_map: `sp.spmatrix` (also accept `np.array`)
        Transition map of the shape: (n_t1_cells, n_t2_cells).
    state_annote: `np.array`
        Annotation for each cell state.
    fate_array: `np.array` or `list`
        List of targeted clusters, consistent with state_annote.
    fate_count: `bool`, optional (default: False)
        Relevant for compute the fate_entropy. If true, just count
        the number of possible (Prob>0) fate outcomes for each state;
        otherwise, compute the shannon entropy of fate outcome for each state
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, compute for initial cell states (rows of Tmap, at t1);
        else, for later cell states (columns of Tmap, at t2)
    method: `str`, optional (default: 'sum')
        Method to aggregate the transition probability within a cluster. Available options: {'sum','mean','max','norm-sum'},
        which computes the sum, mean, or max of transition probability within a cluster as the final fate probability.

    Returns
    -------
    fate_map: `np.array`, shape (n_cells, n_fates)
        A matrix of fate potential for each state
    fate_entropy: `np.array`, shape (n_fates,)
        A vector of fate entropy for each state
    """

    if not ssp.issparse(transition_map):
        transition_map = ssp.csr_matrix(transition_map).copy()
    resol = 10 ** (-10)
    '''
    transition_map = sparse_rowwise_multiply(
        transition_map, 1 / (resol + np.sum(transition_map, 1).A.flatten())
    )'''
    fate_N = len(fate_array)
    N1, N2 = transition_map.shape

    # logg.info(f"Use the method={method} to compute differentiation bias")

    if map_backward:
        idx_array = np.zeros((N2, fate_N), dtype=bool)
        for k in range(fate_N):
            idx_array[:, k] = state_annote == fate_array[k]

        fate_map = np.zeros((N1, fate_N))
        fate_entropy = np.zeros(N1)

        for k in range(fate_N):
            if method == "max":
                fate_map[:, k] = np.max(
                    transition_map[:, idx_array[:, k]], 1
                ).A.flatten()
            elif method == "mean":
                fate_map[:, k] = np.mean(
                    transition_map[:, idx_array[:, k]], 1
                ).A.flatten()
            else:  # just perform summation
                fate_map[:, k] = np.sum(
                    transition_map[:, idx_array[:, k]], 1
                ).A.flatten()

        # rescale. After this, the fate map value spreads between [0,1]. Otherwise, they can be tiny.
        if (method != "sum") and (method != "norm-sum"):
            fate_map = fate_map / np.max(fate_map)
        elif method == "norm-sum":
            # perform normalization of the fate map. This works only if there are more than two fates
            if fate_N > 1:
                # logg.info('conditional method: perform column normalization')
                fate_map = sparse_column_multiply(
                    fate_map, 1 / (resol + np.sum(fate_map, 0).flatten())
                ).A
                fate_map = fate_map / np.max(fate_map)

        for j in range(N1):
            ### compute the "fate-entropy" for each state
            if fate_count:
                p0 = fate_map[j, :]
                fate_entropy[j] = np.sum(p0 > 0)
            else:
                p0 = fate_map[j, :]
                p0 = p0 / (resol + np.sum(p0)) + resol
                for k in range(fate_N):
                    fate_entropy[j] = fate_entropy[j] - p0[k] * np.log(p0[k])

    ### forward map
    else:
        idx_array = np.zeros((N1, fate_N), dtype=bool)
        for k in range(fate_N):
            idx_array[:, k] = state_annote == fate_array[k]

        fate_map = np.zeros((N2, fate_N))
        fate_entropy = np.zeros(N2)

        for k in range(fate_N):
            if method == "max":
                fate_map[:, k] = np.max(
                    transition_map[idx_array[:, k], :], 0
                ).A.flatten()
            elif method == "mean":
                fate_map[:, k] = np.mean(
                    transition_map[idx_array[:, k], :], 0
                ).A.flatten()
            else:
                fate_map[:, k] = np.sum(
                    transition_map[idx_array[:, k], :], 0
                ).A.flatten()

        # rescale. After this, the fate map value spreads between [0,1]. Otherwise, they can be tiny.
        if (method != "sum") and (method != "norm-sum"):
            fate_map = fate_map / np.max(fate_map)
        elif method == "norm-sum":
            # perform normalization of the fate map. This works only if there are more than two fates
            if fate_N > 1:
                # logg.info('conditional method: perform column normalization')
                fate_map = sparse_column_multiply(
                    fate_map, 1 / (resol + np.sum(fate_map, 0).flatten())
                ).A

        for j in range(N2):

            ### compute the "fate-entropy" for each state
            if fate_count:
                p0 = fate_map[j, :]
                fate_entropy[j] = np.sum(p0 > 0)
            else:
                p0 = fate_map[j, :]
                p0 = p0 / (resol + np.sum(p0)) + resol
                for k in range(fate_N):
                    fate_entropy[j] = fate_entropy[j] - p0[k] * np.log(p0[k])

    return fate_map, fate_entropy

def cl_fate_map(
    adata,
    selected_fates=None,
    source="transition_map",
    map_backward=True,
    method="norm-sum",
    fate_count=False,
    force_run=False,
):
    """
    Compute transition probability to given fate/ancestor clusters.

    Given a transition map :math:`T_{ij}`, we explore build
    the fate map :math:`P_i(\mathcal{C})` towards a set of states annotated with
    fate :math:`\mathcal{C}` in the following ways.

    Step 1: Map normalization: :math:`T_{ij}\leftarrow T_{ij}/\sum_j T_{ij}`.

    Step 2: If `map_backward=False`, perform matrix transpose :math:`T_{ij} \leftarrow T_{ji}`.

    Step 3: aggregate fate probabiliteis within a given cluster :math:`\mathcal{C}`:

    * method='sum': :math:`P_i(\mathcal{C})=\sum_{j\in \mathcal{C}} T_{ij}`.
      This gives the intuitive meaning of fate probability.

    * method='norm-sum': We normalize the map from 'sum' method within a cluster, i.e.
      :math:`P_i(\mathcal{C})\leftarrow P_i(\mathcal{C})/\sum_j P_j(\mathcal{C})`.
      This gives the probability that a fate cluster :math:`\mathcal{C}` originates
      from an initial state :math:`i`.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    selected_fates: `list`, optional (default: all fates)
        List of cluster ids consistent with adata.obs['state_info'].
        It allows a nested list, where we merge clusters within
        each sub-list into a mega-fate cluster.
    source: `str`, optional (default: 'transition_map')
        The transition map to be used for plotting: {'transition_map',
        'intraclone_transition_map',...}. The actual available
        map depends on adata itself, which can be accessed at adata.uns['available_map']
    map_backward: `bool`, optional (default: True)
        If `map_backward=True`, show fate properties of initial cell states :math:`i`;
        otherwise, show progenitor properties of later cell states :math:`j`.
        This is used for building the fate map :math:`P_i(\mathcal{C})`. See :func:`.fate_map`.
    method: `str`, optional (default: 'norm-sum')
        Method to obtain the fate probability map :math:`P_i(\mathcal{C})` towards a set
        of states annotated with fate :math:`\mathcal{C}`. Available options:
        {'sum', 'norm-sum'}. See :func:`.fate_map`.
    fate_count: `bool`, optional (default: False)
        Used to determine the method for computing the fate potential of a state.
        If ture, just to count the number of possible fates; otherwise, use the Shannon entropy.
    force_run: `bool`, optional (default: False)
        Re-compute the fate map.

    Returns
    -------
    Fate map for each targeted fate cluster is updated at adata.obs[f'fate_map_{source}_{fate_name}'].
    The accompanying parameters are saved at adata.uns[f"fate_map_{source}_{fate}"]
    """


    if source not in adata.uns["available_map"]:
        raise ValueError(f"source should be among {adata.uns['available_map']}")

    else:

        state_annote = adata.obs["state_info"]
        (
            mega_cluster_list,
            __,
            __,
            sel_index_list,
        ) = analyze_selected_fates(state_annote, selected_fates)

        key_word = f"fate_map_{source}"
        available_choices = parse_output_choices(
            adata, key_word, where="obs", interrupt=False
        )

        # check if we need to recompute the map
        re_compute = True
        condi_0 = len(available_choices) > 0
        condi_1 = set(mega_cluster_list) <= set(available_choices)
        if condi_0 & condi_1:
            map_backward_all = set(
                [
                    adata.uns["fate_map_params"][f"{source}_{x}"]["map_backward"]
                    for x in mega_cluster_list
                ]
            )
            method_all = set(
                [
                    adata.uns["fate_map_params"][f"{source}_{x}"]["method"]
                    for x in mega_cluster_list
                ]
            )
            # check that the parameters are uniform and equals to input
            if len(map_backward_all) == 1 and len(method_all) == 1:
                condi_2 = map_backward == list(map_backward_all)[0]
                condi_3 = method == list(method_all)[0]
                if condi_2 and condi_3:
                    re_compute = False
        if not (re_compute or force_run):
            print("Use pre-computed fate map")
        else:
            (
                fate_map,
                mega_cluster_list,
                relative_bias,
                expected_prob,
                valid_fate_list,
                sel_index_list,
                fate_entropy,
            ) = compute_fate_probability_map(
                adata,
                selected_fates=selected_fates,
                used_Tmap=source,
                map_backward=map_backward,
                method=method,
                fate_count=fate_count,
            )

            if map_backward:
                cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
            else:
                cell_id_t1 = adata.uns["Tmap_cell_id_t2"]

            if "fate_map_params" not in adata.uns.keys():
                adata.uns[f"fate_map_params"] = {}

            if len(mega_cluster_list) == 0:
                print("No cells selected. Computation aborted!")
            else:
                for j, fate in enumerate(mega_cluster_list):
                    temp_map = np.zeros(adata.shape[0]) + np.nan
                    temp_map[cell_id_t1] = fate_map[:, j]
                    adata.obs[f"fate_map_{source}_{fate}"] = temp_map
                    adata.uns[f"fate_map_params"][f"{source}_{fate}"] = {
                        "map_backward": map_backward,
                        "method": method,
                    }
                    print(f"Results saved at adata.obs['fate_map_{source}_{fate}']")

                temp_map = np.zeros(adata.shape[0]) + np.nan
                temp_map[cell_id_t1] = fate_entropy
                adata.uns[f"fate_potency_tmp"] = temp_map


##Util function
def fate_map_embedding(
    adata,
    fate_vector,
    cell_id_t1,
    sp_idx,
    mask=None,
    target_list=None,
    color_bar_label="",
    color_bar_title="",
    figure_title="",
    background=True,
    show_histogram=False,
    auto_color_scale=True,
    color_bar=True,
    horizontal=False,
    target_transparency=0.2,
    histogram_scales=None,
    color_map=None,
    order_method=None,
    vmax=None,
    vmin=None,
):
    """
    Note: sp_idx is a bool array, of the length len(cell_id_t1)
    mask: bool array  of length adata.shape[0]
    """

    fig_width = cs.settings.fig_width
    fig_height = cs.settings.fig_height
    point_size = cs.settings.fig_point_size

    x_emb = adata.obsm["X_emb"][:, 0]
    y_emb = adata.obsm["X_emb"][:, 1]
    state_info = np.array(adata.obs["state_info"])

    if mask is not None:
        if len(mask) == adata.shape[0]:
            mask = mask.astype(bool)
            sp_idx = sp_idx & (mask[cell_id_t1])
        else:
            print("mask length does not match adata.shape[0]. Ignored mask.")

    if np.sum(sp_idx) == 0:
        raise ValueError("No cells selected")

    if color_bar:
        fig_width = fig_width + 1

    if show_histogram:
        tot_N = 2
    else:
        tot_N = 1

    if horizontal:
        row = 1
        col = tot_N
    else:
        row = tot_N
        col = 1

    fig = plt.figure(figsize=(fig_width * col, fig_height * row))

    fate_map_temp = fate_vector[cell_id_t1][sp_idx]
    ax0 = plt.subplot(row, col, 1)
    if background:
        customized_embedding(
            x_emb,
            y_emb,
            np.zeros(len(y_emb)),
            point_size=point_size,
            ax=ax0,
            title=figure_title,
        )
    else:
        customized_embedding(
            x_emb[cell_id_t1][sp_idx],
            y_emb[cell_id_t1][sp_idx],
            np.zeros(len(y_emb[cell_id_t1][sp_idx])),
            point_size=point_size,
            ax=ax0,
            title=figure_title,
        )

    if target_list is not None:
        for zz in target_list:
            idx_2 = state_info == zz
            ax0.plot(
                x_emb[idx_2],
                y_emb[idx_2],
                ".",
                color="#17becf",
                markersize=point_size * 1,
                alpha=target_transparency,
            )

    if auto_color_scale:
        vmax = None
        vmin = None
    else:
        if vmax is None:
            vmax = 1
        if vmin is None:
            vmin = 0

    if order_method == "fate_bias":
        new_idx = np.argsort(abs(fate_map_temp - 0.5))
    else:
        # new_idx = np.arange(len(fate_map_temp))
        new_idx = np.argsort(abs(fate_map_temp))
    mat2 = fate_map_temp[new_idx]
    #mat2 = mat2/np.linalg.norm(mat2)
    mat2 = mat2/max(mat2)

    customized_embedding(
        x_emb[cell_id_t1][sp_idx][new_idx],
        y_emb[cell_id_t1][sp_idx][new_idx],
        fate_map_temp[new_idx],
        #mat2,
        point_size=point_size,
        ax=ax0,
        title=figure_title,
        set_lim=False,
        vmax=vmax,
        vmin=vmin,
        color_bar=color_bar,
        color_bar_label=color_bar_label,
        color_bar_title=color_bar_title,
        color_map=color_map,
        order_points=False,
    )

    if show_histogram:
        ax = plt.subplot(row, col, 2)
        ax.hist(fate_map_temp, 50, color="#2ca02c", density=True)
        if histogram_scales is not None:
            ax.set_xlim(histogram_scales)
        ax.set_xlabel(color_bar_label)
        ax.set_ylabel("Density")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"Ave.: {np.mean(fate_map_temp):.2f}")


color_map_reds = cs.pl.darken_cmap(plt.cm.Reds, scale_factor=0.9)
color_map_coolwarm = cs.pl.darken_cmap(plt.cm.coolwarm, scale_factor=1)
color_map_greys = cs.pl.darken_cmap(plt.cm.Greys, scale_factor=1)
def pl_fate_map(
    adata,
    selected_fates=None,
    source="transition_map",
    selected_times=None,
    background=True,
    show_histogram=False,
    plot_target_state=False,
    auto_color_scale=True,
    color_bar=True,
    target_transparency=0.2,
    figure_index="",
    mask=None,
    color_map=color_map_reds,
    savefig=True,
    **kwargs,
):
    """
    Plot transition probability to given fate/ancestor clusters.

    The results should be pre-computed from :func:`cospar.tl.fate_map`.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData` object
        Assume to contain transition maps at adata.uns.
    {selected_fates}
    {source}
    {selected_times}
    {background}
    {show_histogram}
    plot_target_state:
        If True, plot target states.
    {color_bar}
    auto_color_scale:
        True: automatically rescale the color range to match the value range.
    {target_transparency}
    {figure_index}
    {mask}
    {color_map}
    """

    key_word = f"fate_map_{source}"
    available_choices = parse_output_choices(adata, key_word, where="obs")
    time_info = np.array(adata.obs["time_info"])
    state_info = adata.obs["state_info"]
    if selected_fates is None:
        selected_fates = sorted(list(set(state_info)))
    (
        mega_cluster_list,
        valid_fate_list,
        __,
        sel_index_list,
    ) = analyze_selected_fates(state_info, selected_fates)

    for j, fate_key in enumerate(mega_cluster_list):
        if fate_key not in available_choices:
            print(
                f"The fate map for {fate_key} have not been computed yet. Skipped!"
            )
        else:
            fate_vector = np.array(adata.obs[f"fate_map_{source}_{fate_key}"])
            params = adata.uns["fate_map_params"][f"{source}_{fate_key}"]
            map_backward = params["map_backward"]
            method = params["method"]

            if map_backward:
                cell_id_t1 = adata.uns["Tmap_cell_id_t1"]
            else:
                cell_id_t1 = adata.uns["Tmap_cell_id_t2"]
            sp_idx = selecting_cells_by_time_points(
                time_info[cell_id_t1], selected_times
            )

            if method == "norm-sum":
                color_bar_label = "Progenitor prob."
            else:
                color_bar_label = "Fate prob."

            if plot_target_state:
                target_list = valid_fate_list[j]
            else:
                target_list = None

            fate_map_embedding(
                adata,
                fate_vector,
                cell_id_t1,
                sp_idx,
                mask=mask,
                target_list=target_list,
                color_bar_label=color_bar_label,
                color_bar_title="",
                figure_title=fate_key,
                background=background,
                show_histogram=show_histogram,
                auto_color_scale=auto_color_scale,
                color_bar=color_bar,
                target_transparency=target_transparency,
                color_map=color_map,
                **kwargs,
            )

            plt.tight_layout()
            data_des = adata.uns["data_des"][-1]
            if figure_index != "":
                figure_index == f"_{figure_index}"
            plt.savefig(
                os.path.join(
                    f"{cs.settings.figure_path}",
                    f"{data_des}_{key_word}_{fate_key}{figure_index}.{cs.settings.file_format_figs}",
                )
            )




### import libraries

# standard libraries
import numpy as np
import pandas as pd

# math libraries
# import scipy.sparse.issparse
# import scipy.stats.ranksums
# import scipy.interpolate
import scipy
from statsmodels.nonparametric.smoothers_lowess import lowess as  sm_lowess
import math #to round up

# single cell libraries
import scanpy as sc
sc.settings.verbosity = 0 

### functions
def filter_data(
    adata, 
    mito_perc=5, 
    min_genes=700, 
    no_doublet=True, 
    no_negative=True,
):
    '''
    Filters raw count matrix.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    mito_perc : `int` (default: 5)
        Maximum percentage of mitochondrial genes in each individual cell.
    min_genes : `int` (default: 700)
        Minimum number of unique genes expressed in each individual cell. 
    no_doublet : `bool` (default: True)
        Remove doublets (as indicated by "Doublet" in adata.obs["hashtags"])
    no_negative : `bool` (default: True)
        Remove cells with no hashtag (as indicated by "Negative" in adata.obs["hashtags"])
    '''
    
    # annotate the group of mitochondrial genes as 'mt'
    adata.var['mt'] = adata.var_names.str.startswith('mt-')
    
    # filter cells with high % of mitochondrial genes
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs['pct_counts_mt'] < mito_perc, :]

    # filter cells with low no. of unique genes expressed
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # filter out doublets and negatives
    if no_doublet==True:
        adata = adata[adata.obs["hashtags"]!= "Doublet",:]
    if no_negative==True:
        adata = adata[adata.obs["hashtags"]!= "Negative",:]
    
    # remove unnecessary obs and vars in anndata object
    if "hashtags" in adata.obs.columns:
        adata.obs = adata.obs[["hashtags"]]
    else:
        adata.obs = adata.obs[[]]
    adata.var = adata.var[[]]
    
    return(adata)

def pearson_residuals(
    counts, 
    theta=100,
):
    '''
    Computes analytical residuals for NB model with a fixed theta, 
    clipping outlier residuals to sqrt(N) as proposed in 
    Lause et al. 2021 https://doi.org/10.1186/s13059-021-02451-7
    
    Parameters
    ----------
    counts: `matrix` 
        Matrix (dense) with cells in rows and genes in columns
    theta: `int` (default: 100)
        Gene-shared overdispersion parameter
    '''
    
    counts_sum0 = np.sum(counts, axis=0)
    counts_sum1 = np.sum(counts, axis=1)
    counts_sum  = np.sum(counts)

    # get residuals
    mu = counts_sum1 @ counts_sum0 / counts_sum
    z = (counts - mu) / np.sqrt(mu + (np.square(mu)/theta))

    # clip to sqrt(n)
    n = counts.shape[0]
    z[z >  np.sqrt(n)] =  np.sqrt(n)
    z[z < -np.sqrt(n)] = -np.sqrt(n)

    return z

def get_hvgs(
    adata, 
    no_of_hvgs=2000, 
    theta=100,
):
    '''
    Function to select the top x highly variable genes (HVGs) 
    from an anndata object. 
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    no_of_hvgs: `int` (default: 2000)
        Number of hig
    theta: `int` (default: 100)
        Gene-shared overdispersion parameter used in pearson_residuals 
    '''
    
    # get pearson residuals
    if scipy.sparse.issparse(adata.X):
        residuals = pearson_residuals(adata.X.todense(),theta)
    else:
        residuals = pearson_residuals(adata.X,theta)
    
    # get variance of residuals
    residuals_variance = np.var(residuals,axis=0) 
    variances = pd.DataFrame({"variances":pd.Series(np.array(residuals_variance).flatten()), 
                              "genes":pd.Series(np.array(adata.var_names))})
    
    # get top x genes with highest variance
    hvgs = variances.sort_values(by="variances", ascending=False)[0:no_of_hvgs]["genes"].values
    
    return hvgs

def find_degs(
    adata1, 
    adata2, 
    top_x=50, 
    return_values="degs",
):
    '''
    Finds differentially expressed genes using Wilcoxon rank sum test on the 
    distribution of each gene in adata1 vs adata2.
    
    Parameters
    ----------
    adata1
        Annotated data matrix with all cells from selected cluster.
    adata2
        Annotated data matrix with all cells from all but selected cluster.
    top_x: `int` (default: 50)
        Number of DEGs to return
    return_values: `str` (default:"degs")
        String determing whether to return the degs or their p-values.
    '''
    
    ### find the genes that the sets have in common
    genes = adata1.var_names.intersection(adata2.var_names) 

    p_values = []
    
    ### get count matrices
    counts1 = pd.DataFrame(adata1.X.todense(), columns=adata1.var_names)
    counts2 = pd.DataFrame(adata2.X.todense(), columns=adata2.var_names)

    ### calculate p-values per gene
    for gene in genes:
        p_value = scipy.stats.ranksums(counts1[gene], counts2[gene])[1] #rank sum test
        p_values.append(p_value)
    p_values = pd.Series(p_values, index=adata1.var_names) #add gene names
    
    ### select top x DEGs
    p_values = p_values.sort_values()[0:top_x]
    degs = p_values.index
    
    if return_values == "degs":
        return(degs)
    
    elif return_values == "p_values":
        return(p_values)
    
    else:
        print("False entry for returned")
        
def calculate_geneset_scores(
    adata, 
    geneset,
):
    '''
    Scores each cell in the adata dataset for the marker genes of each
    celltype in the set of marker genes.
    
    Parameters
    ----------
    adata
        Annotated data matrix
    geneset
        Dataframe with in each column marker genes assigned to one
        specific cell type. The column names should include the 
        corresponding cell type names.
    '''
    ### create empty dataframe
    df_geneset_scores = pd.DataFrame(index=adata.obs.index) 
    
    ### loop over each cell type in the geneset
    for j in range(geneset.shape[1]):
        geneset_name = geneset.columns[j]
        sc.tl.score_genes(adata, list(geneset.iloc[:, j]), score_name="geneset_score")
        df_geneset_scores[geneset_name] = list(adata.obs["geneset_score"])
        del adata.obs["geneset_score"]
    
    return(df_geneset_scores)        

def integrate_datasets(adata, basis, batch_key="time", n_comps=100):
    '''
    Function to integrate multiple subsets in an Adata object using
    Scanorama (Hie, Bryson & Berger, 2019). The resulting Scanorama
    space is reduced in dimensions using PCA.
    
    Parameters
    ----------
    adata
        Annotated data matrix
    basis: `matrix` 
        Matrix (dense) with cells in rows and genes or reduced 
        dimensions in columns
    batch_key: `str` (default: "time")
        The name of the column in adata.obs that differentiates among 
        experiments/batches.Cells from the same batch must be 
        contiguously stored in adata
    n_comps: `int` (default: 100)
        Number of dimensions to return for the integrated space
    '''
    
    # add basis to adata
    adata.obsm["X_basis"] = basis
    
    # scanorama batch correction
    sc.external.pp.scanorama_integrate(adata, batch_key, basis="X_basis")
    
    # reduce dimensions 
    adata.obsm["X_scanorama_reduced"] = sc.tl.pca(adata.obsm["X_scanorama"], n_comps=n_comps)
    
    # create dataframe of scanorama space
    scanorama_df = pd.DataFrame(adata.obsm["X_scanorama_reduced"], index=adata.obs_names)
    
    return(scanorama_df)
        
def label_transfer(adata, batch_key="time", basis='X_scanorama_reduced', label_key="clusters",
                   reference="control", query="3h", no_neighbours=10):
    '''
    Function to transfer labels from a reference to a query. 
    Query and reference should both be included in one
    Anndata object. 
    
    Parameters
    ----------
    adata
        Annotated data matrix
    batch_key: `str` (default: "time")
        The name of the column in adata.obs that differentiates 
        reference from query
    basis: `str` (default: "X_scanorama_reduced")
        The name of the matrix in adata.obsm that is used to 
        calculate distance between cells
    label_key: `str` (default: "clusters")
        The name of the column in adata.obs which contains 
        the labels that have to be transferred
    reference: `str` (default: "control")
        The name that seperates the reference from the query
        in the adata.obs column indicated using batch_key
    query
        The name that seperates the query from the 
        reference in the adata.obs column indicated using 
        batch_key
    no_neighbours: `int` (default: 10)
        Number of neighbours to use for data integration
    '''

    distances = scipy.spatial.distance.cdist(adata[adata.obs[batch_key]==reference].obsm[basis],
                                             adata[adata.obs[batch_key]==query].obsm[basis], 
                                             metric='euclidean')
    df_distances = pd.DataFrame(distances,
                                index=adata[adata.obs[batch_key]==reference].obs[label_key], 
                                columns=adata[adata.obs[batch_key]==query].obs_names)
    neighbours = df_distances.apply(lambda x: pd.Series(x.nsmallest(no_neighbours).index))
    transferred_labels = neighbours.value_counts().idxmax()
    transferred_labels = pd.Series(transferred_labels, dtype="category")
    transferred_labels.index = adata[adata.obs[batch_key]==query].obs_names  
    
    return transferred_labels
    
def get_delta_expression(adata, genes, time_key="time", 
                         label_key="clusters", cluster=None):
    '''
    Computes the expression change for each timestep for one or 
    multiple genes. The expression change is either calculated 
    for the whole dataset or just for one cluster. 
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    genes: `array` 
        Array with all genes for which we want to get the 
        expression change in each timestep.
    time_key: `str` (default: "time")
        The name of the column in adata.obs that differentiates 
        between timepoints.
    label_key: `str` (default: "clusters")
        The name of the column in adata.obs that differentiates 
        between clusters.
    cluster: `str` (default: None)
        Which cluster to calculate the expression change for. 
        If not entered, the function will calculate expression
        change for the whole dataset.
    '''
    
    # get timepoints
    timepoints = adata.obs[time_key].cat.categories.values
    
    # get expression for all response genes in each cell
    if cluster is None:
        expression = adata[:,genes].X.toarray()
        expression = pd.DataFrame(expression,
                                  columns=genes,
                                  index=adata.obs[time_key])
    else:
        expression = adata[adata.obs[label_key]==cluster][:,genes].X.toarray()
        expression = pd.DataFrame(expression,
                                  columns=genes,
                                  index=adata[adata.obs[label_key]==cluster].obs[time_key])

    # calculate mean expression in each timepoint (for all response genes)
    mean_expr = expression.groupby(level=0).mean()

    # get delta expression in each timestep (for all response genes)
    delta_expr = mean_expr.diff()[1:len(timepoints)]

    return delta_expr

def get_expression(adata, genes, time_key="time",
                   label_key="clusters", cluster=None):
    '''
    Gets the expression in each timepoint for one or 
    multiple genes. The expression can be found
    for the whole dataset or just for one cluster. 
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    genes: `array` 
        Array with all genes for which we want to get the 
        expression change in each timestep.
    time_key: `str` (default: "time")
        The name of the column in adata.obs that differentiates 
        between timepoints.
    label_key: `str` (default: "clusters")
        The name of the column in adata.obs that differentiates 
        between clusters.
    cluster: `str` (default: None)
        Which cluster to calculate the expression change for. 
        If not entered, the function will find expression
        for the whole dataset.
    '''
    
    # get expression for all response genes in each cell
    if cluster is None:
        expression = adata[:,genes].X.toarray()
        expression = pd.DataFrame(expression,
                                  columns=genes,
                                  index=adata.obs[time_key])
    else:
        expression = adata[adata.obs[label_key]==cluster][:,genes].X.toarray()
        expression = pd.DataFrame(expression,
                                  columns=genes,
                                  index=adata[adata.obs[label_key]==cluster].obs[time_key])

    # calculate mean expression in each timepoint (for all response genes)
    mean_expr = expression.groupby(level=0).mean()

    return mean_expr
    
    

def bin_smooth(x, y, xgrid, sample_size=0.5, window_size=50):
    
    # take samples
    samples = np.random.choice(len(x), int(len(x)*sample_size), replace=True)
    x_sample = x[samples]
    y_sample = y[samples]
    
    if window_size >= len(samples):
            print("Unable to smooth: sample size is not bigger than window size.")
    
    # sort samples
    x_s_sorted = np.sort(x_sample)
    y_s_sorted = y_sample[np.argsort(x_sample)]

    window_halfsize = window_size/2
    
    y_smooth = []
            
    for idx, yi in enumerate(y_s_sorted):
        if window_halfsize < idx < (len(y_s_sorted) - window_halfsize):
            y_smooth.append(np.mean(y_s_sorted[int(idx-window_halfsize-1):int(idx+window_halfsize-1)]))
        else:
            y_smooth.append(None)
    
    # apply found funcction to x values on xgrid
    ygrid = scipy.interpolate.interp1d(x_s_sorted, y_smooth, fill_value='extrapolate')(xgrid) 

    return(ygrid)

def loess_smooth(x, y, xgrid, sample_size=0.5):
    
    # take samples
    samples = np.random.choice(len(x), int(len(x)*sample_size), replace=True)
    #samples = np.random.choice(len(x), 200, replace=True)
    x_sample = x[samples]
    y_sample = y[samples]
    
    y_smooth = sm_lowess(y_sample, x_sample, frac=0.4, it=5, return_sorted = False)
    
    # apply found function to x values on xgrid
    ygrid = scipy.interpolate.interp1d(x_sample, y_smooth, fill_value='extrapolate')(xgrid) 

    return(ygrid)

def bootstrap_smoothing(x, y, method="bin", sampling_rounds=20, sample_size=0.5, window_size=50):
    """
    This function takes x and y data (such as gene expression or gene score in 
    time) and smooths the datapoints. Method for smoothing is 'bin' or 'loess'.
    For bin smoothing a window size can be specified. 
    """
    
    xgrid = np.linspace(x.min(),x.max(),num=50) #set x-es that are compared in the end
    K = sampling_rounds #set number of samples for bootstrap
    
    if method == "bin":
        ygrids = np.stack([bin_smooth(x, y, xgrid, sample_size=sample_size, window_size=window_size) for k in range(K)]).T
        
    elif method == "loess":
        ygrids = np.stack([loess_smooth(x, y, xgrid, sample_size=sample_size) for k in range(K)]).T
        
    mean = np.nanmean(ygrids, axis=1)
    stderr = np.nanstd(ygrids, axis=1, ddof=0)
    
    return(xgrid, ygrids, mean, stderr)
