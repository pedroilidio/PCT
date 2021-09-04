#!/bin/env python
"""Provides a PBCT classifier implementation, as proposed by Pliakos, Geurts and Vens in 2018 (DOI: 10.1007/s10994-018-5700-x).

Classes
-------
    PBCT, PBCTClassifier
"""
# TODO: Cost-complexity prunning.
# TODO: Lookahead.
# TODO: There are duplicate instances which receives different labels
#       in database. Preprocessing it (i.e. averaging) may enhance performance.
#       Changes in Node.predict must follow. You can also force to not happen.

import joblib
from itertools import product
import numba
import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange

DEFAULTS = dict(
    path_model='trained_model.json',
    min_samples_leaf=20,
    max_depth=-1,
    path_rendering='model_visualization',
)
CLEAN_AND_RETURN = '\x1b[1K\r'


def load_model(path):
    model_data = joblib.load(path)
    model = PBCT(**model_data['parameters'])
    model.tree = model_data['tree']
    return model


def ax_means(Y):
    dims = tuple(range(Y.ndim))
    all_but_i = (dims[:i] + dims[i+1:] for i in dims)
    return [Y.mean(axis=ax) for ax in all_but_i]


def select_from_local_ids(XX, Y, ids, local_ids, axis):
    """Given indices to go to each side, split XX and Y in 2."""

    local_ids1, local_ids2 = local_ids

    ids1, ids2 = ids.copy(), ids.copy()
    ids1[axis] = ids[axis][local_ids1]
    ids2[axis] = ids[axis][local_ids2]

    XX1, XX2 = XX.copy(), XX.copy()
    XX1[axis] = XX[axis][local_ids1]
    XX2[axis] = XX[axis][local_ids2]

    Y_slice1 = [slice(None)] * Y.ndim
    Y_slice2 = [slice(None)] * Y.ndim
    Y_slice1[axis] = local_ids1
    Y_slice2[axis] = local_ids2
    Y1, Y2 = Y[tuple(Y_slice1)], Y[tuple(Y_slice2)]

    return [(XX1, Y1, ids1), (XX2, Y2, ids2)]


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

    def __init__(self, savepath=None, verbose=False, max_depth=DEFAULTS['max_depth'],
                 min_samples_leaf=DEFAULTS['min_samples_leaf'], split_min_quality=0,
                 leaf_target_quality=1, max_variance=0):
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
        )
        for k, v in self.parameters.items():
            setattr(self, k, v)

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

            node = dict(
                is_leaf=False,
                cutoff=cutoff,
                coord=(axis, attr_id),
                local_ids=(ids1, ids2),
                quality=q,
            )

        kwargs.update(node)
        return kwargs

    def _build_tree(self, XX, Y):
        """Build decision tree as nested dicts."""

        tree = self._make_node(
            XX, Y, XXY=(XX, Y), pos=0, depth=0,
            ids = [np.arange(i) for i in Y.shape],
        )

        if tree['is_leaf']: return tree
        node_queue = [tree]

        while node_queue:
            parent_node = node_queue.pop(0)  # With a zero is BFS, else is DFS.
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
            XYi1, XYi2 = select_from_local_ids(
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
            print(CLEAN_AND_RETURN + format(pos, 'b'), end='')

            child2 = self._make_node(
                XX2, Y2,
                XXY=(XX2, Y2),
                depth=dep,
                pos=pos+1,
                ids=ids2,
                Yshape=Y2.shape,
            )
            print(CLEAN_AND_RETURN + format(pos+1, 'b'), end='')

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

    def _predict_sample(self, xx, simple_mean=True, verbose=False):
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
        else:
            raise NotImplementedError(
                'Please use simple_mean=True when predicting.'
            )

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
            return ax_Ymean  # FIXME

    def predict(self, XX, ax=None, simple_mean=True, verbose=False):
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

    def _ax_predict_sample(self, x, axis=0):
        """For an instance of one axis, predict prob. of interaction with train
        ing data of the other.
        """
        leafs = []
        queue = [self.tree]  # Initiate leaf search on root.

        while queue:
            node = queue.pop()
            if node['is_leaf']:
                leafs.append(node)
            else:
                ax, attr_idx = node['coord']  # Split coordinates.
                if ax != axis:
                    for child in node['children']:
                        queue.append(child)
                else:
                    right = x[attr_idx] >= node['cutoff']
                    queue.append(node['children'][right])

        ids = np.hstack([leaf['ids'][not axis] for leaf in leafs])
        pred = np.hstack([leaf['axmeans'][not axis] for leaf in leafs])
        return pred[ids]

    def ax_predict(self, X, axis=0, verbose=False):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if verbose:
            X = tqdm(X)
        ret = np.array([self._ax_predict_sample(x, axis=axis) for x in X])
        return ret

    def save(self, path=None):
        if self.savepath is not None:
            path = self.savepath
        elif path is None:
            raise TypeError('self.savepath must be set or argument path'
                            'must be given.')

        model_data = dict(
            parameters=self.parameters,
            tree=self.tree,
        )
        joblib.dump(model_data, path)

    # TODO: Decide wether to normalize by total # of leaves.
    # TODO: Determine if calculating on each axis really makes sense.
    def feature_importances(self, node=None, ret=None):
        """Calculate Mean Decrease in Impurity for a trained tree, in each axis."""
        
        is_root_call = ret is None
        if is_root_call:
            node = self.tree
            ret = dict(cols={}, rows={}, total={})
        #   feature_importances.n_nodes = dict(rows=0, cols=0, total=0)
        
        if node['is_leaf']:
            return ret
        
        # feature_importances.n_nodes['total'] += 1
        # feature_importances.n_nodes[node.split_axis] += 1
        print(bin(node['pos'])[2:], end='\x1b[1K\r')
        shape = node['Yshape']
        name = node['coord']
        
        # of rows, # of cols, total items.
        sizes = dict(rows=shape[0], cols=shape[1], total=shape[0]*shape[1])
        
        for key, size in sizes.items():
            ret[key][name] = ret[key].get(name, 0) + size * node['quality']
        
        for child in node['children']:
            ret = self.feature_importances(child, ret=ret)
        
        if is_root_call:
            ret = pd.DataFrame(ret).sort_values('total', ascending=False)
            ret /= sizes
            # ret /= feature_importances.n_nodes  # sizes already decrease with node number?
        return ret


class PBCTClassifier(PBCT):
    """It's PBCT (see class def), but assuming the interaction matrix to be binary."""

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
