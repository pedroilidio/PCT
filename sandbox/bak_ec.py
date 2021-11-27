import multiprocessing as mp
from itertools import product
import joblib
import pandas as pd
import numpy as np
import numba
from numba import types
from .classes import PBCT, PBCTClassifier, ax_means, DEFAULTS
from . import classes
from tqdm import tqdm

# From: https://github.com/numba/numba/issues/1269#issuecomment-472574352
@numba.njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@numba.njit
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)

@numba.njit
def np_std(array, axis):
    return np_apply_along_axis(np.std, axis, array)
####

@numba.njit(fastmath=True)
def cumdist1d(xx):
    """Cumulative squared distance to the mean."""
    m = xx[0]
    s = 0.
    # Don't go to the last one, since we start with the first+second
    # element and the reverse array must then be all but them.
    res = np.zeros(xx.shape[0]-3, dtype=np.float64)

    for i in range(1, xx.shape[0]-2):
        x = xx[i]
        xmold = x-m
        m += xmold / (i+1)
        s += xmold * (x-m)
        res[i-1] = s# / i  # unbiased estimation

    return res

@numba.njit(fastmath=True)
def cumdist2d(xx):
    """Cumulative squared distance to the mean."""
    # m = np.zeros(xx.shape[1], dtype=np.float64)
    m = xx[0]
    s = np.zeros(xx.shape[1], dtype=np.float64)
    xmold = s.copy()
    res = np.zeros((xx.shape[0]-3, xx.shape[1]), dtype=np.float64)

    # starts at the second element, k=2=i+
    for i in range(1, xx.shape[0]-2):
        x = xx[i]
        xmold[:] = x - m
        m += xmold / (i+1)
        s += xmold * (x-m)
        res[i-1][:] = s# / i  # S/(k-1) sample correction

    return res


@numba.njit(fastmath=True)
def _find_best_split_ss(
    attr_col, X, Xvar,
    Y_means, Yvar, min_samples_leaf,
    supervision):
    """It uses more memory and computes unnecessary stuff, but it is readable."""

    unsupervision = 1 - supervision
    total_length = attr_col.size
    msl = min_samples_leaf - 1

    # Sort Ymeans by the attribute values.
    sorted_ids = attr_col.argsort()
    Y_means = Y_means[sorted_ids]
    X = X[sorted_ids]

    # Cumulative squared euclidean distance to the mean.
    X1_loss = cumdist2d(X)
    X2_loss = cumdist2d(X[::-1])[::-1]
    Y1_loss = cumdist1d(Y_means)
    Y2_loss = cumdist1d(Y_means[::-1])[::-1]

    uq = 1 - np_mean(X1_loss + X2_loss, 1) / total_length / Xvar
    sq = 1 - (Y1_loss + Y2_loss) / total_length / Yvar

    total_q = (unsupervision * uq) + (supervision * sq)
    if msl != 0:
        total_q = total_q[msl:-msl]
    cut_id = np.argmax(total_q)
    quality = total_q[cut_id]

    if quality <= 0:
        return None  # FIXME: Is it OK to return different types?

    # +1 bellow because we have to include sorted_ids[cut_id] in ids1
    cut_id += msl + 2
    ids1, ids2 = sorted_ids[:cut_id], sorted_ids[cut_id:]
    cutoff = (attr_col[ids1[-1]] + attr_col[ids2[0]]) / 2

    return quality, cutoff, ids1, ids2


@numba.njit(fastmath=True)
def _find_best_split_welford(attr_col, Y_means, Yvar, min_samples_leaf):
    """It uses more memory and computes unnecessary stuff, but it is readable."""
    # FIXME: You've called it "min_samples_leaf" but it's actually the minimum
    # number of instances in the given axis, minimum lines or columns to have.

    # Variance of 1 element is 0, "great split you have".
    assert min_samples_leaf >= 2

    if Y_means.size < 2*min_samples_leaf:
        return None
    total_length = attr_col.size

    # Sort Ymeans by the attribute values.
    sorted_ids = attr_col.argsort()
    Y_means = Y_means[sorted_ids]

    # Cumulative squared euclidean distance to the mean.
    Y1_loss = cumdist1d(Y_means)
    Y2_loss = cumdist1d(Y_means[::-1])[::-1]

    q = 1 - (Y1_loss + Y2_loss) / total_length / Yvar

    # Remember min_samples_leaf >= 2 (first line).
    if min_samples_leaf != 2:
        msl = min_samples_leaf - 2
        q = q[msl:-msl]
    cut_id = np.argmax(q)
    quality = q[cut_id]

    if quality <= 0:
        return None  # FIXME: Is it OK to return different types?

    # +1 bellow because we have to include sorted_ids[cut_id] in ids1
    cut_id += min_samples_leaf
    ids1, ids2 = sorted_ids[:cut_id], sorted_ids[cut_id:]
    cutoff = (attr_col[ids1[-1]] + attr_col[ids2[0]]) / 2

    return quality, cutoff, ids1, ids2


'''
# BUG: SSPCT not passing test_PBCT_class.
@numba.njit(fastmath=True)
def _find_best_split_bin_Y_ss(attr_col, X_means, X_sq_means, Xvar,
                              Y_means, Yvar, min_samples_leaf,
                              supervision):
    """Equals to _find_best_split, but assumes Y content is binary and considers
       features when splitting.

    See _find_best_split().
    """
    unsupervision = 1 - supervision
    total_length = attr_col.size
    highest_quality = 0
    # The index right before the cutoff value in the sorted X.
    cut_id = -1

    # Sort Ymeans by the attribute values.
    sorted_ids = attr_col.argsort()
    Y_means = Y_means[sorted_ids]
    X_means = X_means[sorted_ids]
    X_sq_means = X_sq_means[sorted_ids]

    # Split Y in Y1 and Y2.
    i = min_samples_leaf
    Y1_means_sum = Y_means[:i].sum()
    Y2_means_sum = Y_means[i:].sum()
    Y1size, Y2size = i, total_length-i

    # Split X in X1 and X2.
    X1_means_sum = X_means[:i].sum()
    X2_means_sum = X_means[i:].sum()
    X1_sq_means_sum = X_sq_means[:i].sum()
    X2_sq_means_sum = X_sq_means[i:].sum()

    for j in range(i, Y2size+1):  # Y2size happens to be what I needed.
        Y1var = Y1_means_sum - Y1_means_sum ** 2 / Y1size
        Y2var = Y2_means_sum - Y2_means_sum ** 2 / Y2size
        supervised_q = 1 - (Y1var + Y2var) / total_length / Yvar

        X1var = X1_sq_means_sum - X1_means_sum ** 2 / Y1size
        X2var = X2_sq_means_sum - X2_means_sum ** 2 / Y2size
        unsupervised_q = 1 - (X1var + X2var) / total_length / Xvar

        quality = (supervision * supervised_q) + \
                  (unsupervision * unsupervised_q)

        if quality > highest_quality:
            highest_quality = quality
            cut_id = j

        Y_current_mean = Y_means[j]
        X_current_mean = X_means[j]
        X_current_sq_mean = X_sq_means[j]

        Y1_means_sum += Y_current_mean
        Y2_means_sum -= Y_current_mean
        X1_means_sum += X_current_mean
        X2_means_sum -= X_current_mean
        X1_sq_means_sum += X_current_sq_mean
        X2_sq_means_sum -= X_current_sq_mean

        Y1size += 1
        Y2size -= 1

    if cut_id == -1:
        return None  # FIXME: Is it OK to return different types?
    else:
        ids1, ids2 = sorted_ids[:cut_id], sorted_ids[cut_id:]
        cutoff = (attr_col[ids1[-1]] + attr_col[ids2[0]]) / 2
        return highest_quality, cutoff, ids1, ids2
'''

@numba.njit(
    types.Tuple((types.int64, types.float64))(
        types.DictType(
            keyty=types.int64,
            valty=types.containers.Tuple((
                types.boolean,
                types.float64,
                types.int64,
                types.int64,
            ))
        ),
        types.UniTuple(
            types.Array(
                dtype=types.float64,
                ndim=1,
                layout='A',
            ), 2
        )
    ),
    fastmath=True,
)
def _get_to_leaf(tree, xx):
        """Predict prob. of interaction given each object's attributes.

        Predict the probability of existing interaction between two ob-
        jects from sets (xrow and xcol) of each one's attribute values.
        """
        # if node['is_leaf']:
        #     return True, node['mean'], 0, 0
        # else:
        #     return False, node['cutoff'], *node['coord']

        pos = 0  # Start at root node.
        is_leaf, cutoff, ax, attr_idx = tree[0]

        while not is_leaf:  # Get to a leaf.
            pos = (pos << 1) + (xx[ax][attr_idx] >= cutoff)
            is_leaf, cutoff, ax, attr_idx = tree[pos]

        # cutoff here is actually the submatrix's mean.
        return pos, cutoff  


class SSPBCT(PBCTClassifier):
    """Semi-supervised PBCT, attributes' variance are accounted in split qua\
    lity."""

    def __init__(self, savepath=None, verbose=False, max_depth=DEFAULTS['max_depth'],
                 min_samples_leaf=DEFAULTS['min_samples_leaf'], split_min_quality=0,
                 leaf_target_quality=1, max_variance=0, supervision=1):
        """Instantiate new PBCT."""

        self.parameters = dict(
            savepath=savepath,
            verbose=verbose,
            max_depth=max_depth,
            # TODO: make it one value per axis.
            min_samples_leaf=min_samples_leaf,
            split_min_quality=split_min_quality,
            leaf_target_quality=leaf_target_quality,
            max_variance=max_variance,
            supervision=supervision,
        )
        for k, v in self.parameters.items():
            setattr(self, k, v)

    def _find_best_split(self, *args, **kwargs):
            return _find_best_split_ss(
                *args, **kwargs,
                min_samples_leaf=self.min_samples_leaf,
                supervision=self.supervision,
            )

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
            if X.shape[0] < 4:
                # Can't cut if there is only one row,
                # cumdist would return empty array and
                # _find_best_split would raise ValueError.
                # TODO: find a better place to test that?
                continue
            Xvar = X.var(axis=0).mean()

            for attr_id, attr_col in enumerate(X.T):
                split = self._find_best_split(
                    attr_col=attr_col,
                    X=X,
                    Xvar=Xvar,
                    Y_means=Y_ax_means[axis],
                    Yvar=Yvar,
                )

                if split and (split[0] > best_split[0]):
                    best_split = split + (axis, attr_id)
        
        if best_split == (0,):
            return Ymean, Y_ax_means, None
        return Ymean, Y_ax_means, best_split


def _node_to_tuple(node):
    if node['is_leaf']:
        return True, node['mean'], 0, 0
    else:
        return False, node['cutoff'], *node['coord']


def _numbize_tree(tree, verbose=False):
    typed_tree = numba.typed.Dict.empty(
        key_type=types.int64,
        value_type=types.containers.Tuple((
            types.boolean,
            types.float64,
            types.int64,
            types.int64,
        ))
    )

    items = tree.items()
    if verbose:
        items = tqdm(items, total=len(tree))

    for pos, node in items:
        typed_tree[pos] = _node_to_tuple(node)

    return typed_tree


class PBCTFastPredict(PBCTClassifier):
    """tree is dict of tuples, worse fit, faster JIT compiled predict_sample."""
    # TODO: Parallelize predict. We are failing to pickle numba typed Dict.
    # FIXME: This class is not passing test_PBCT_class.py!

    def _build_tree(self, XX, Y):
        """Build decision tree dict[pos: tuple]."""

        root = self._make_node(
            XX, Y, XXY=(XX, Y), pos=0, depth=0,
            ids = [np.arange(i) for i in Y.shape],
        )

        tree = {0: _node_to_tuple(root)}
        if root['is_leaf']:
            return tree
        node_queue = [root]

        while node_queue:
            parent_node = node_queue.pop(0)  # With a zero is BFS, without is DFS.
            # Interestingly, pos is simply loop count in binary, for BFS in a
            # complete tree.
            # pos == 0b01101 means left right right left right
            pos = parent_node['pos'] << 1
            dep = parent_node['depth'] + 1
            parent_ids = parent_node['ids']
            axis = parent_node['coord'][0]
            # local_ids are based on the considered submatrix, while ids refers
            # to the whole initial Y.

            # FIXME: Gambiarra alert: XXY key is just used temporarily.
            XYi1, XYi2 = classes.select_from_local_ids(
                *parent_node['XXY'],
                parent_ids,
                parent_node['local_ids'],
                axis,
            )
            XX1, Y1, ids1, XX2, Y2, ids2 = *XYi1, *XYi2
            del parent_node['XXY']

            # TODO: Use a loop here?
            child1 = self._make_node(
                XX1, Y1,
                XXY=(XX1, Y1),  
                depth=dep,
                pos=pos,
                ids=ids1,
                Yshape=Y1.shape,
            )
            # Minimal progress report.
            print(classes.CLEAN_AND_RETURN + format(pos, 'b'), end='')

            child2 = self._make_node(
                XX2, Y2,
                XXY=(XX2, Y2),
                depth=dep,
                pos=pos+1,
                ids=ids2,
                Yshape=Y2.shape,
            )
            print(classes.CLEAN_AND_RETURN + format(pos+1, 'b'), end='')

            tree[pos] = child1
            tree[pos+1] = child2

            if not child1['is_leaf']:
                node_queue.append(child1)
            if not child2['is_leaf']:
                node_queue.append(child2)

        print()
        return tree

    def _predict_sample(self, xx, verbose=False):
        return _get_to_leaf(self._typed_tree, xx)[1]

    def predict(self, XX, ax=None, verbose=False, n_jobs=1):
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
        size = np.prod(shape)
        chunksize = int(size/n_jobs) + 1  # Ceil it.

        if not hasattr(self, '_typed_tree'):
            self._typed_tree = _numbize_tree(self.tree)

        prod_iter = product(*XX)
        if verbose:
            prod_iter = tqdm(prod_iter, total=np.prod(shape))

        # Y_pred = joblib.Parallel(n_jobs, verbose=1)(
        #     joblib.delayed(self._predict_sample)(xx, verbose=verbose)
        #     for xx in prod_iter
        # )
        # Y_pred = [
        #     self._predict_sample(xx, verbose=verbose)
        #     for xx in prod_iter
        # ]
        with mp.Pool(n_jobs) as pool:
            Y_pred = pool.imap(self._predict_sample, prod_iter,
                               chunksize=chunksize)
            Y_pred = np.fromiter(Y_pred, float).reshape(shape)

        return Y_pred


# class PBCTWelford(PBCTClassifier):
#     """Uses Welford's online variance method."""
#     def _find_best_split(self, X, Y_means, Yvar):
#         return _find_best_split_welford(X, Y_means, Yvar, self.min_samples_leaf)

# FIXME: Produces bigger trees. Why??
class PBCTWelford(PBCT):
    """Uses Welford's online variance method."""
    def _find_best_split(self, X, Y_means, Y_sqmeans, Yvar):
        wret = _find_best_split_welford(
            X,
            Y_means,
            Yvar,
            self.min_samples_leaf,
        )
        sret = classes._find_best_split(
            X,
            Y_means,
            Y_sqmeans,
            Yvar,
            self.min_samples_leaf,
        )
        breakpoint()
        return sret
