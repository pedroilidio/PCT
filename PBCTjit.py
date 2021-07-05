#!/bin/env python
"""Provides a PBCT classifier implementation and related tools.

Sandbox for Numba JIT compilation experiments.

Classes
-------
    PBCT*
"""
# TODO: Cost-complexity prunning.
# TODO: Lookahead.
# TODO: There are duplicate instances which receives different labels
#       in database. Preprocessing it (i.e. averaging) may enhance performance.
#       Changes in Node.predict must follow.

import numba
from itertools import product
import pandas as pd
import numpy as np
import joblib
import graphviz
from tqdm.auto import tqdm, trange
from matplotlib import colors
import colorsys
from matplotlib.cm import get_cmap
import PBCTv3 as P

PATH_MODEL = 'model.pickle'
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 20
PATH_RENDERING = 'model'


def ax_means(Y):
    dims = tuple(range(Y.ndim))
    all_but_i = [dims[:i] + dims[i+1:] for i in dims]
    return [Y.mean(axis=ax) for ax in all_but_i]


@numba.njit(fastmath=True)
def _find_best_split(X, Y_means, Y_sq_means, Yvar, min_samples_leaf):
    """Find the best cuttof value of a row or column attribute.

    For a row or column attribute, find the value wich most reduces the
    overall variance of the dataset when used as a threshold to split
    the intereaction matrix (Y) in two parts. The overall variance is
    the size-weighted average of the subsets' variances.

    Parameters
    ----------
        X : array_like
            All the values of a row or column attribute in the dataset.
        Y_means : array_like
            Row or column -wise means of the in- teraction matrix (Y),
            depending if X refers to a row or column attribute.
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
        try:
            Y1var = Y1_sq_means_sum - Y1_means_sum ** 2 / Y1size
            Y2var = Y2_sq_means_sum - Y2_means_sum ** 2 / Y2size
            quality = 1 - (Y1var + Y2var) / total_length / Yvar
        except:
            quality = 0

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
        return None
    else:
        ids1, ids2 = sorted_ids[:cut_id], sorted_ids[cut_id:]
        cutoff = (X[ids1[-1]] + X[ids2[0]]) / 2
        return highest_quality, cutoff, ids1, ids2


@numba.njit
def _predict_sample(xx, leaf, simple_mean=False):
    """Predict prob. of interaction given each object's attributes.

    Predict the probability of existing interaction between two ob-
    jects from sets (xrow and xcol) of each one's attribute values.
    """

    while not leaf['is_leaf']:
        ax, attr_idx = leaf['coord']  # Split coordinates.
        right = xx[ax][attr_idx] >= leaf['cutoff']
        leaf = leaf['children'][right]

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


@numba.njit
def _cut_Y(XX, Y, Y_ax_means, Ysq_ax_means, min_samples_leaf):
    """Find the best row or column attribute to cut Y.

    Find the attribute of either Y's rows or columns which best splits
    Y in two parts, i. e. generates the lowest size-weighted average of
    the parts' variances after splitting.

    Parameters
    ----------
        Xrows[cols] : array_like
            A matrix in which each line is an array of the correspond-
            ing Y's row [column].
        Yvar : float
            The precomputed variance of Y.
        Yrow[col]_means : array_like
            Row[column]-wise mean of the interaction matrix (Y).
        Yrow[col]_sq_means : array_like
            The same as Yrow[col]_means, but taking the square of Y's
            values before averaging them.

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
        axis : {'rows', 'cols'}
            'rows' if the best split is along Y's rows or 'cols' if it
            is along Y's columns.
        Attribute's index : int
            The index of the attribute column used to split Y, 4 mean-
            ing it is the 5th row or column attribute.

        or

        None
            if no split was found to reduce the variance.
    """
    Ymean, Ysq_mean = Y_ax_means[0].mean(), Ysq_ax_means[0].mean()
    Yvar = Ysq_mean - Ymean ** 2
    shape = Y.shape

    # best_split = (highest_quality, best_cutoff, ids1, ids2)
    base = 0., 0., np.array([0.]), np.array([0.])
    best_split = base
    coord = 0, 0

    for axis in numba.prange(len(XX)):
        X_cols = XX[axis].T

        for attr_id in numba.prange(len(X_cols)):
            attr_col = X_cols[attr_id]
            split = _find_best_split(attr_col, Y_ax_means[axis], Ysq_ax_means[axis], Yvar, min_samples_leaf)
            #first_time = (axis == 0) and (attr_id == 0)
            if split is not None and (split[0] > best_split[0]):
                best_split = split
                coord = axis, attr_id
    
    return Ymean, Y_ax_means


class PBCT(P.PBCT4):
    """self.tree is nested dicts, itertools.product over np broadcast, JIT
    compiled _find_best_split"""

    def _find_best_split(self, X, Y_means, Y_sq_means, Yvar):
        return _find_best_split(X, Y_means, Y_sq_means,
                                Yvar, self.min_samples_leaf)

    # def _cut_Y(self, XX, Y):
    #     # It is faster to resquare it here than to index the parent's Ysq.
    #     Y_ax_means = ax_means(Y)
    #     Ysq = Y ** 2
    #     Ysq_ax_means = ax_means(Ysq)
    #     return _cut_Y(XX, Y, Y_ax_means, Ysq_ax_means, self.min_samples_leaf)

    # def _predict_sample(self, xx, simple_mean=False):
    #     return _predict_sample(xx, self.tree, simple_mean=simple_mean)


################################################################# CLASS DEF

spec = [
    ('max_depth', numba.types.int32),
    ('min_samples_leaf', numba.types.int32),
    ('split_min_quality', numba.types.float64),
    ('leaf_target_quality', numba.types.float64),
    ('max_variance', numba.types.float64),
    ('verbose', numba.types.boolean),
]

@numba.experimental.jitclass(spec)
class PBCTjit():  # FIXME: Fails to infer type.
    """ no np broadcasting to achieve cartesian product. itertools.product
    instead."""
    def __init__(self, max_depth=MAX_DEPTH,
                 min_samples_leaf=MIN_SAMPLES_LEAF, split_min_quality=0,
                 leaf_target_quality=1, max_variance=0, verbose=False):
        """Instantiate new PBCT node."""
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.split_min_quality = split_min_quality
        self.leaf_target_quality = leaf_target_quality
        self.max_variance = max_variance
        self.verbose = verbose

    def _find_best_split(self, X, Y_means, Y_sq_means, Yvar):
        """Find the best cuttof value of a row or column attribute.

        For a row or column attribute, find the value wich most reduces the
        overall variance of the dataset when used as a threshold to split
        the intereaction matrix (Y) in two parts. The overall variance is
        the size-weighted average of the subsets' variances.

        Parameters
        ----------
            X : array_like
                All the values of a row or column attribute in the dataset.
            Y_means : array_like
                Row or column -wise means of the in- teraction matrix (Y),
                depending if X refers to a row or column attribute.
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
        cut_id = None

        # Sort Ymeans by X.
        sorted_ids = X.argsort()
        Y_means = Y_means[sorted_ids]
        Y_sq_means = Y_sq_means[sorted_ids]

        # Split Y in Y1 and Y2.
        i = self.min_samples_leaf
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
            try:
                Y1var = Y1_sq_means_sum - Y1_means_sum ** 2 / Y1size
                Y2var = Y2_sq_means_sum - Y2_means_sum ** 2 / Y2size
                quality = 1 - (Y1var + Y2var) / total_length / Yvar
            except:
                quality = 0

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

        if cut_id is None:
            return None
        else:
            ids1, ids2, = sorted_ids[:cut_id], sorted_ids[cut_id:]
            cutoff = (X[ids1[-1]] + X[ids2[0]]) / 2
            return highest_quality, cutoff, ids1, ids2

    def _cut_Y(self, XX, Y):
        """Find the best row or column attribute to cut Y.

        Find the attribute of either Y's rows or columns which best splits
        Y in two parts, i. e. generates the lowest size-weighted average of
        the parts' variances after splitting.

        Parameters
        ----------
            Xrows[cols] : array_like
                A matrix in which each line is an array of the correspond-
                ing Y's row [column].
            Yvar : float
                The precomputed variance of Y.
            Yrow[col]_means : array_like
                Row[column]-wise mean of the interaction matrix (Y).
            Yrow[col]_sq_means : array_like
                The same as Yrow[col]_means, but taking the square of Y's
                values before averaging them.

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
            axis : {'rows', 'cols'}
                'rows' if the best split is along Y's rows or 'cols' if it
                is along Y's columns.
            Attribute's index : int
                The index of the attribute column used to split Y, 4 mean-
                ing it is the 5th row or column attribute.

            or

            None
                if no split was found to reduce the variance.
        """
        # It is faster to resquare it here than to index the parent's Ysq.
        Ysq = Y ** 2

        dims = tuple(range(Y.ndim))
        all_but_i = [dims[:i] + dims[i+1:] for i in dims]
        Y_ax_means = [Y.mean(axis=ax) for ax in all_but_i]
        Ysq_ax_means = [Ysq.mean(axis=ax) for ax in all_but_i]

        Ymean, Ysq_mean = Y_ax_means[0].mean(), Ysq_ax_means[0].mean()
        Yvar = Ysq_mean - Ymean ** 2
        self.shape = Y.shape

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
            best_split = None
        return Ymean, Y_ax_means, best_split

    def _make_node(self, XX, Y, depth=0):
        is_leaf = (depth == self.max_depth)
        if not is_leaf:
            Ymean, Y_ax_means, best_split = self._cut_Y(XX, Y)
            # Node is leaf if no split was found to reduce Y variance or
            # if split quality already satisfies the defined target or
            # if no split satisfied minimum required quality.
            is_leaf = (best_split is None
                       or best_split[0] > self.leaf_target_quality
                       or best_split[0] < self.split_min_quality)

        if is_leaf:
            node = dict(
              # is_leaf=is_leaf,
                is_leaf=True,
                depth=depth,
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
                depth=depth,
                cutoff=cutoff,
                coord=(axis, attr_id),
                XXXX=(XX1, XX2),
                YY=(Y1, Y2),
                quality=q,
            )

        return node

    def _build_tree(self, XX, Y, depth=0):
        # TODO: use only Y indices in each node?

        node = self._make_node(XX, Y, depth=depth)

        if node['is_leaf']:
            return node

        XX1, XX2 = node['XXXX']
        Y1, Y2 = node['YY']
        del node['XXXX'], node['YY']

        node['children'] = (
            self._build_tree(XX1, Y1, depth=depth+1),
            self._build_tree(XX2, Y2, depth=depth+1))

        return node


    def fit(self, XX, Y, X_names=None):
        self.tree = self._build_tree(XX, Y)

    def _predict_sample(self, xx, simple_mean=False):
        """Predict prob. of interaction given each object's attributes.

        Predict the probability of existing interaction between two ob-
        jects from sets (xrow and xcol) of each one's attribute values.
        """
        leaf = self.tree  # Initiate leaf search on root.

        while not leaf['is_leaf']:
            ax, attr_idx = leaf['coord']  # Split coordinates.
            right = xx[ax][attr_idx] >= leaf['cutoff']
            leaf = leaf['children'][right]

        if self.verbose:
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

    def predict(self, XX, simple_mean=False):
        """Predict prob. of interaction between rows and columns objects.

        Predict the probability of ocurring interaction between two arrays
        of column and row instances. Each array is assumed to contain o-
        ther array-like objects consisting of attribute values for each
        instance.

        Parameters
        ----------
            XX : array-like
                List of lists of instances' attributes, for instances of
                the interaction matrix rows' type and to the columns' type,
                respectively.
        """
        if isinstance(XX[0], pd.DataFrame):
            XX = [X.values for X in XX]

        shape = tuple(len(X) for X in XX)
        # TODO: Check ndim.
        ndim = len(XX)

        if self.verbose:
            self._pbar = tqdm(total=np.prod(shape))

        Y_pred = [self._predict_sample(xx, simple_mean=simple_mean)
                  for xx in product(*XX)]
        Y_pred = np.array(Y_pred).reshape(shape)

        if self.verbose:
            self._pbar.close()
            del self._pbar

        return Y_pred

#     def _polish_input(self, XX, Y, X_names):
#         if isinstance(Y, pd.DataFrame):  # FIXME: Testing only Y?
#             if X_names is None:
#                 X_names = [X.columns for X in XX]
#             XX = [X.values for X in XX]
#             Y = Y.values
# 
#         # Check dimensions.
#         X_shapes0 = tuple(X.shape[0] for X in XX)
#         X_shapes1 = tuple(X.shape[1] for X in XX)
# 
#         if Y.shape != X_shapes0:
#             raise ValueError('The lengths {X_shapes0} of each X in XX mus'
#                              't match Y.shape = {Y.shape}.')
# 
#         if X_names is None:
#             X_names = [['[{i}]' for i in range(len(X))] for X in XX]
#         else:
#             X_names_shape = tuple(len(names) for names in X_names)
# 
#             if not all(isinstance(n[0], str) for n in X_names):
#                 raise ValueError('X_names must be a list-like of list-like'
#                                  's of string labels.')
#             if X_names_shape != X_shapes1:
#                 raise ValueError('The number of columns {X_shapes1} of ea'
#                                  'ch X in XX must match number of names gi'
#                                  'ven for each X {X_names_shape}.')
# 
#         return XX, Y, X_names
