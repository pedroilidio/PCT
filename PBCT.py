#!/bin/env python
"""Provides a PBCT classifier implementation and related tools.

Main version, aggregating all the knowledge from other versions
experimentation.

Classes
-------
    PBCT, PBCTClassifier
"""
# TODO: Cost-complexity prunning.
# TODO: Lookahead.
# TODO: There are duplicate instances which receives different labels
#       in database. Preprocessing it (i.e. averaging) may enhance performance.
#       Changes in Node.predict must follow. You can also force to not happen.

import numba
from itertools import product
import pandas as pd
import numpy as np
import graphviz
from tqdm.auto import tqdm, trange
from matplotlib import colors
import colorsys
from matplotlib.cm import get_cmap

DEF_PATH_MODEL = 'model_data.json'
DEF_MAX_DEPTH = 30
DEF_MIN_SAMPLES_LEAF = 20
DEF_PATH_RENDERING = 'model'


def ax_means(Y):
    dims = tuple(range(Y.ndim))
    all_but_i = (dims[:i] + dims[i+1:] for i in dims)
    return [Y.mean(axis=ax) for ax in all_but_i]


@numba.njit(fastmath=True)
def _find_best_split(X, Y_means, Y_sq_means, Yvar, min_samples_leaf):
    """Find the best cuttof value of a row or column attribute.

    For a single attribute column, find the value wich most reduces the
    overall variance of the dataset when used as a threshold to split
    the interaction matrix (Y) in two parts. The overall variance is
    the size-weighted average of the subsets' variances.

    Parameters
    ----------
        X : array_like
            All the values of a row or column attribute in the dataset.
        Y_means : array_like
            Label (y) means of the interaction matrix (Y), collapsing all
            Y values to the axis of X.
        Y_sq_means : array_like
            The same as Y_means, but taking the square of Y's values
            before averaging them.
        Yvar : float
            The precomputed variance of Y.

    Returns
    -------
        None
            if no split was found to reduce Y's variance.

        highest_quality : float
            Quality of the best variance reducer split found. A quality
            value of 0.7 means the weighted mean variance of the two Y
            submatrices is 30% of Y's variance before splitting.
        cutoff : float
            The attribute's threshold value which best cuts Y.
        ids1, ids2 : array_like
            The indices of X's values below and above the cutoff.
    """
    total_length = X.size
    highest_quality = 0
    # The index right before the cutoff value in the sorted X.
    cut_id = -1

    # Sort Ymeans by X.
    sorted_ids = X.argsort()
    Y_means = Y_means[sorted_ids]
    Y_sq_means = Y_sq_means[sorted_ids]

    # Split Y in Y1 and Y2.
    i = min_samples_leaf
    Y1_means_sum = Y_means[:i].sum()
    Y2_means_sum = Y_means[i:].sum()
    Y1_sq_means_sum = Y_sq_means[:i].sum()
    Y2_sq_means_sum = Y_sq_means[i:].sum()
    Y1size, Y2size = i, total_length-i

    # To save processing, calculate variance as <x^2> - <x>^2, instead
    # of the more common form <(x - <x>)^2>. <x> refers to the mean va-
    # lue of x. This allows us to have <x> and <x^2> values precomputed
    # along the Y axis we are examining.

    # var(x) = <(x - <x>)^2> =
    #        = <x^2 - 2x<x> + <x>^2> =
    #        = <x^2> - 2<x><x> + <x>^2 =
    #        = <x^2> - <x>^2

    for j in range(i, Y2size+1):  # Y2size happens to be what I needed.
        Y1var = Y1_sq_means_sum - Y1_means_sum ** 2 / Y1size
        Y2var = Y2_sq_means_sum - Y2_means_sum ** 2 / Y2size
        quality = 1 - (Y1var + Y2var) / total_length / Yvar

        ### Clearer but verboser equivalent:
        # Y1mean = Y1_means_sum / Y1size
        # Y2mean = Y2_means_sum / Y2size
        # Y1sq_mean = Y1_sq_means_sum / Y1size
        # Y2sq_mean = Y2_sq_means_sum / Y2size
        # Y1var = Y1sq_mean - Y1mean ** 2
        # Y2var = Y2sq_mean - Y2mean ** 2
        # split_var = (Y1size*Y1var + Y2size*Y2var) /total_length
        # quality = (Yvar - split_var) / Yvar

        if quality > highest_quality:
            highest_quality = quality
            cut_id = j

        current_mean = Y_means[j]
        current_sq_mean = Y_sq_means[j]

        Y1_means_sum += current_mean
        Y2_means_sum -= current_mean
        Y1_sq_means_sum += current_sq_mean
        Y2_sq_means_sum -= current_sq_mean

        Y1size += 1
        Y2size -= 1

    if cut_id == -1:
        return None  # FIXME: Is it OK to return different types?
    else:
        ids1, ids2 = sorted_ids[:cut_id], sorted_ids[cut_id:]
        cutoff = (X[ids1[-1]] + X[ids2[0]]) / 2
        return highest_quality, cutoff, ids1, ids2


# TODO: Really repeat the whole function def.?
@numba.njit(fastmath=True)
def _find_best_split_binY(X, Y_means, Yvar, min_samples_leaf):
    """Equals to _find_best_split, but assumes Y content is binary.

    See _find_best_split().
    """
    total_length = X.size
    highest_quality = 0
    # The index right before the cutoff value in the sorted X.
    cut_id = -1

    # Sort Ymeans by X.
    sorted_ids = X.argsort()
    Y_means = Y_means[sorted_ids]

    # Split Y in Y1 and Y2.
    i = min_samples_leaf
    Y1_means_sum = Y_means[:i].sum()
    Y2_means_sum = Y_means[i:].sum()
    Y1size, Y2size = i, total_length-i

    for j in range(i, Y2size+1):  # Y2size happens to be what I needed.
        Y1var = Y1_means_sum - Y1_means_sum ** 2 / Y1size
        Y2var = Y2_means_sum - Y2_means_sum ** 2 / Y2size
        quality = 1 - (Y1var + Y2var) / total_length / Yvar

        if quality > highest_quality:
            highest_quality = quality
            cut_id = j

        current_mean = Y_means[j]
        Y1_means_sum += current_mean
        Y2_means_sum -= current_mean
        Y1size += 1
        Y2size -= 1

    if cut_id == -1:
        return None  # FIXME: Is it OK to return different types?
    else:
        ids1, ids2 = sorted_ids[:cut_id], sorted_ids[cut_id:]
        cutoff = (X[ids1[-1]] + X[ids2[0]]) / 2
        return highest_quality, cutoff, ids1, ids2


class PBCT:
    """self.tree is nested dicts, itertools.product over np broadcast, JIT compiled _find_best_split"""

    def __init__(self, savepath=None, verbose=False, max_depth=DEF_MAX_DEPTH,
                 min_samples_leaf=DEF_MIN_SAMPLES_LEAF, split_min_quality=0,
                 leaf_target_quality=1, max_variance=0):
        """Instantiate new PBCT."""
        self.savepath = savepath
        self.verbose = verbose
        self.max_depth = max_depth
        # TODO: make it one value per axis.
        self.min_samples_leaf = min_samples_leaf
        self.split_min_quality = split_min_quality
        self.leaf_target_quality = leaf_target_quality
        self.max_variance = max_variance

    def _find_best_split(self, X, Y_means, Y_sq_means, Yvar):
        return _find_best_split(X, Y_means, Y_sq_means,
                                Yvar, self.min_samples_leaf)

    def _cut_Y(self, XX, Y):
        """Find the best row or column attribute to cut Y.

        Find the attribute of either Y's rows or columns which best splits
        Y in two parts, i. e. generates the lowest size-weighted average of
        the parts' variances after splitting.

        Returns
        -------
            Highest quality : float
                Quality of the best variance reducer split found. A quality
                value of 0.7 means the weighted mean variance of the two Y
                submatrices is 30% of Y's variance before splitting.
            Best cutoff : float
                The attribute's threshold value which best cuts Y.
            ids1, ids2 : array_like
                The row [column] indices of Y for that generates each split.
            axis : int
                The axis on which to perform Y splitting.
            Attribute's index : int
                The index of the attribute column used to split Y, 4 mean-
                ing it is the 5th attribute column of the selected axis.

            or

            None
                if no split was found to reduce the variance.
        """
        # FIXME: Inherit it for  each node. Speed improvement < 6% :(
        Ysq = Y ** 2
        dims = tuple(range(Y.ndim))
        all_but_i = [dims[:i] + dims[i+1:] for i in dims]
        Y_ax_means = [Y.mean(axis=ax) for ax in all_but_i]
        Ysq_ax_means = [Ysq.mean(axis=ax) for ax in all_but_i]

        # FIXME: Inherit means. Speed improvement < 6% :(
        Ymean, Ysq_mean = Y_ax_means[0].mean(), Ysq_ax_means[0].mean()
        Yvar = Ysq_mean - Ymean ** 2
        if Yvar == 0:  # If goal achieved.
            return Ymean, Y_ax_means, None

        # best_split = (highest_quality, best_cutoff,
        #               ids1, ids2, split_axis, attribute_index)
        best_split = 0,

        for axis, X in enumerate(XX):
            X_cols = enumerate(X.T)
            if self.verbose:
                X_cols = tqdm(X_cols, desc=f'ax={axis}',
                              total=X.shape[1])

            for attr_id, attr_col in X_cols:
                split = self._find_best_split(attr_col, Y_ax_means[axis],
                                             Ysq_ax_means[axis], Yvar)
                if split and (split[0] > best_split[0]):
                    best_split = split + (axis, attr_id)
        
        if best_split == (0,):
            return Ymean, Y_ax_means, None
        return Ymean, Y_ax_means, best_split

    def _make_node(self, XX, Y, **kwargs):
        # TODO: If quality == 1 found, automatically tag the children no-
        # des as leaves, as its Yvar will be 0 for sure. (here is where
        # you would also use q >= min_q.
        if kwargs['depth'] == self.max_depth:
            # NOTE: You already calculate these in self._cut_Y
            Ymean = Y.mean()
            Y_ax_means = ax_means(Y)
            is_leaf = True
        else:
            Ymean, Y_ax_means, best_split = self._cut_Y(XX, Y)
            # Node is leaf if no split was found to reduce Y variance or
            # if no split satisfied minimum required quality.
            # NOTE: Use pos as tree dict keys?
            is_leaf = best_split is None \
                      or best_split[0] < self.split_min_quality
                      # FIXME: Does the above make sense?
        if is_leaf:
            node = dict(
              # is_leaf=is_leaf,
                is_leaf=True,
                XX=XX,
              # Y=Y,
                mean=Ymean,
                axmeans=Y_ax_means,
            )
        else:
            q, cutoff, ids1, ids2, axis, attr_id = best_split

            XX1, XX2 = XX.copy(), XX.copy()
            XX1[axis] = XX[axis][ids1]
            XX2[axis] = XX[axis][ids2]

            Y_slice1 = [slice(None)] * Y.ndim
            Y_slice2 = [slice(None)] * Y.ndim
            Y_slice1[axis] = ids1
            Y_slice2[axis] = ids2
            Y1, Y2 = Y[tuple(Y_slice1)], Y[tuple(Y_slice2)]

            node = dict(
                is_leaf=False,
                cutoff=cutoff,
                coord=(axis, attr_id),
                XXXX=(XX1, XX2),
                YY=(Y1, Y2),
                quality=q,
            )

        kwargs.update(node)
        return kwargs

    def _build_tree(self, XX, Y):
        """Build decision tree as nested dicts."""
        tree = self._make_node(XX, Y, pos=0, depth=0)
        if tree['is_leaf']: return tree
        node_queue = [tree]

        while node_queue:
            parent_node = node_queue.pop()
            # NOTE: is pos simply loop count in binary for BFS?
            # pos == 0b01101 means left right right left right
            pos = parent_node['pos'] << 1
            dep = parent_node['depth'] + 1
            XX1, XX2 = parent_node['XXXX']
            Y1, Y2 = parent_node['YY']
            del parent_node['XXXX'], parent_node['YY']

            child1 = self._make_node(XX1, Y1, depth=dep, pos=pos)
            print('\x1b[1K\r' + bin(pos)[2:], end='')
            child2 = self._make_node(XX2, Y2, depth=dep, pos=pos+1)
            print('\x1b[1K\r' + bin(pos+1)[2:], end='')
            parent_node['children'] = (child1, child2)

            if not child1['is_leaf']:
                node_queue.append(child1)
            if not child2['is_leaf']:
                node_queue.append(child2)

        print()
        return tree

    def _polish_input(self, XX, Y, X_names):
        if isinstance(Y, pd.DataFrame):  # FIXME: Testing only Y?
            if X_names is None:
                X_names = [X.columns for X in XX]
            XX = [X.values for X in XX]
            Y = Y.values

        # Check dimensions.
        X_shapes0 = tuple(X.shape[0] for X in XX)
        X_shapes1 = tuple(X.shape[1] for X in XX)

        if Y.shape != X_shapes0:
            raise ValueError(f'The lengths {X_shapes0} of each X in XX mus'
                             f't match Y.shape = {Y.shape}.')

        if X_names is None:
            X_names = [[f'[{i}]' for i in range(len(X))] for X in XX]
        else:
            X_names_shape = tuple(len(names) for names in X_names)

            if not all(isinstance(n[0], str) for n in X_names):
                raise ValueError('X_names must be a list-like of list-like'
                                 's of string labels.')
            if X_names_shape != X_shapes1:
                raise ValueError(f'The number of columns {X_shapes1} of ea'
                                  'ch X in XX must match number of names gi'
                                 f'ven for each X {X_names_shape}.')

        return XX, Y, X_names

    def fit(self, XX, Y, X_names=None):
        XX, Y, X_names = self._polish_input(XX, Y, X_names)
        self.tree = self._build_tree(XX, Y)

    def _predict_sample(self, xx, simple_mean=False, verbose=False):
        """Predict prob. of interaction given each object's attributes.

        Predict the probability of existing interaction between two ob-
        jects from sets (xrow and xcol) of each one's attribute values.
        """
        leaf = self.tree  # Initiate leaf search on root.

        while not leaf['is_leaf']:
            ax, attr_idx = leaf['coord']  # Split coordinates.
            right = xx[ax][attr_idx] >= leaf['cutoff']
            leaf = leaf['children'][right]

        if verbose:
            self._pbar.update()
        if simple_mean:
            return leaf['mean']

        # Search for known instances in training set (XX) and use their
        # interaction probability as result if any was found.
        Y_slice = []
        any_known = False

        for X, x in zip(leaf['XX'], xx):
            # TODO: Assume there is only one known instance at most, i.e.
            # use some "get_index" instead of a boolean mask.
            known = (X == x).all(axis=1)  # Better if it was int.
            if known.any():
                any_known = True
                Y_slice.append(kwown)
            else:
                Y_slice.append(slice(None))

        if any_known:
            return leaf['Y'][tuple(Y_slice)].mean()
        else:
            return ax_Ymean

    def predict(self, XX, simple_mean=False, verbose=False):
        """Predict prob. of interaction between rows and columns objects.

        Predict the probability of ocurring interaction between two arrays
        of column and row instances. Each array is assumed to contain o-
        ther array-like objects consisting of attribute values for each
        instance.

        Parameters
        ----------
            XX : array-like
                List of lists of instances' attributes, i.e. list of attri-
                butes tables, one table for each axis, one attributes row
                for each instance.
            simple_mean : boolean
                Average the whole Y submatrix at the leaf node in each pre-
                diction. Otherwise, search for known rows or cols and aver-
                age only them.
            verbose : boolean
                Use tqdm to display progress.

        Returns
        -------
            Ypred : ndarray
                Predictions arranged in a matrix, respecting each axis' i-
                nitial ordering.
        """
        if isinstance(XX[0], pd.DataFrame):
            XX = [X.values for X in XX]

        shape = tuple(len(X) for X in XX)
        ndim = len(XX)

        if verbose:
            self._pbar = tqdm(total=np.prod(shape))

        Y_pred = [self._predict_sample(xx, simple_mean=simple_mean,
                                       verbose=verbose)
                  for xx in product(*XX)]
        Y_pred = np.array(Y_pred).reshape(shape)

        if verbose:
            self._pbar.close()
            del self._pbar

        return Y_pred

    def save(self, path=None):
        if self.savepath is not None:
            path = self.savepath
        elif path is None:
            raise TypeError('self.savepath must be set or argument path'
                            'must be given.')

        with open(path, 'w') as fp:
            json.dump(self.tree, fp)

    def load(self, path=None):
        if self.savepath is not None:
            path = self.savepath
        elif path is None:
            raise TypeError('self.savepath must be set or argument path'
                            'must be given.')

        with open(path) as fp:
            self.tree = json.load(fp)


class PBCTClassifier(PBCT):
    """Equals to PBCT, except for assuming binary interaction matrix."""

    def _find_best_split(self, X, Y_means, Yvar):
        return _find_best_split_binY(X, Y_means, Yvar,
                                     self.min_samples_leaf)

    # TODO: Really repeat the whole method def?
    def _cut_Y(self, XX, Y):
        """Find the best row or column attribute to cut Y.

        Find the attribute of either Y's rows or columns which best splits
        Y in two parts, i. e. generates the lowest size-weighted average of
        the parts' variances after splitting.

        Returns
        -------
            Highest quality : float
                Quality of the best variance reducer split found. A quality
                value of 0.7 means the weighted mean variance of the two Y
                submatrices is 30% of Y's variance before splitting.
            Best cutoff : float
                The attribute's threshold value which best cuts Y.
            ids1, ids2 : array_like
                The row [column] indices of Y for that generates each split.
            axis : int
                The axis on which to perform Y splitting.
            Attribute's index : int
                The index of the attribute column used to split Y, 4 mean-
                ing it is the 5th attribute column of the selected axis.

            or

            None
                if no split was found to reduce the variance.
        """
        Y_ax_means = ax_means(Y)

        # FIXME: Inherit means. Speed improvement < 6% :(
        Ymean = Y_ax_means[0].mean()
        Yvar = Ymean - Ymean ** 2
        if Yvar == 0:  # If goal achieved.
            return Ymean, Y_ax_means, None

        # best_split = (highest_quality, best_cutoff,
        #               ids1, ids2, split_axis, attribute_index)
        best_split = 0,

        for axis, X in enumerate(XX):
            X_cols = enumerate(X.T)
            if self.verbose:
                X_cols = tqdm(X_cols, desc=f'ax={axis}',
                              total=X.shape[1])

            for attr_id, attr_col in X_cols:
                split = self._find_best_split(attr_col,
                                              Y_ax_means[axis], Yvar)
                if split and (split[0] > best_split[0]):
                    best_split = split + (axis, attr_id)
        
        if best_split == (0,):
            return Ymean, Y_ax_means, None
        return Ymean, Y_ax_means, best_split
