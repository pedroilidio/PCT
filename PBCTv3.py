#!/bin/env python
"""Provides a PBCT classifier implementation and related tools.

Third version, experiment using loops over recursion and build-
iing tree data as pure python objects, json compatible.

Classes
-------
    PBCT*
"""
# TODO: Cost-complexity prunning.
# TODO: Lookahead.
# TODO: There are duplicate instances which receives different labels
#       in database. Preprocessing it (i.e. averaging) may enhance performance.
#       Changes in Node.predict must follow.
import argparse
import json
from itertools import product
#from copy import copy, deepcopy
import pandas as pd
import numpy as np
import joblib
import graphviz
from tqdm.auto import tqdm, trange
# from sklearn.base import ClassifierMixin
from matplotlib import colors
import colorsys
from matplotlib.cm import get_cmap
# np.random.seed(42)

PATH_MODEL = 'model.pickle'
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 20
PATH_RENDERING = 'model'


def ax_means(Y):
    dims = tuple(range(Y.ndim))
    all_but_i = [dims[:i] + dims[i+1:] for i in dims]
    return [Y.mean(axis=ax) for ax in all_but_i]


class PBCT():
    #"""Create a new predictive bi-clustering tree (PBCT) node."""
    """Nested dict tree, recursive tree building"""

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

    def _make_node(self, XX, Y, depth=0):
        is_leaf = (depth == self.max_depth)
        if not is_leaf:
            Ymean, Y_ax_means, best_split = self._cut_Y(XX, Y)
            # Node is leaf if no split was found to reduce Y variance or
            # if no split satisfied minimum required quality.
            is_leaf = (best_split is None
                       or best_split[0] < self.split_min_quality)
        else:
            Ymean = Y.mean()
            Y_ax_means = ax_means(Y)

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

    def fit(self, XX, Y=None, X_names=None):
        """Find recursively the best splits, building the tree.

        Iterate through rows' and columns' attributes of the interaction
        matrix Y, searching for the attribute and cutoff value which most
        reduces the size-weighted average of the Y's split out parts.

        Then, instantiate self's children nodes for each Y part as
        self.little_child and self.big_child, for attribute values lower
        or greater than the cutoff, respectively.

        Either Y or Y*_means arguments must be given.

        Parameters
        ----------
            Xrows[cols] : array_like
                A matrix in which each line is an array of the correspond-
                ing Y's row [column].
            Y : array_like
                The interaction matrix (labels).
            Yrow[col]_means : array_like
                Row[column]-wise mean of the interaction matrix (Y).
            Yrow[col]_sq_means : array_like
                The same as Yrow[col]_means, but taking the square of Y's
                values before averaging them.
            Xrows[cols]_names : list_like
                List-like of string identifiers for the rows [columns] at-
                tributes.
            store_ids : bool, default=False
                If True, create self.ids1 and self.ids2 variables contain-
                ing split indices for posterior inspection or visualiza-
                tion.
        """
        XX, Y, X_names = self._polish_input(XX, Y, X_names)

        # Line bellow is kinda cumbersome. It vectorizes the predict
        # function, so that the predicted interaction matrix will be
        # created with numpy's broadcasting. It needs to happen here,
        # after we know the number of dimensions we are dealing with,
        # because the signature argument needs it and we need the
        # signature argument for the predict function to approach the
        # data row-wisely.

        # (a0),(a1),...,(a{ndim-1})->()
        sig = ','.join(f'(a{i})' for i in range(Y.ndim)) + '->()'
        self._v_predict_sample = np.vectorize(
            self._predict_sample,
            signature=sig,
            excluded=['simple_mean', 'leaf'],
        )
        self.tree = self._build_tree(XX, Y)

        # TODO: On the split_axis, you can only reindex means, no need
        # to calculate again at child nodes.

        # TODO: when multiple values are equal there will be less cutoff
        #       values than expected.

    def _predict_sample(self, *xx, simple_mean=False):
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

        for X, x in zip(node['XX'], xx):
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

        if self.verbose:
            total = 1
            for X in XX:
                total *= len(X)
            self._pbar = tqdm(total=total)

        # TODO: Check ndim.
        ndim = len(XX)
        slices = np.tile(np.newaxis, (ndim, ndim)).astype(object)
        np.fill_diagonal(slices, slice(None))
        XXnewax = [X[tuple(s)] for X, s in zip(XX, slices)]

        Y_pred = self._v_predict_sample(*XXnewax, simple_mean=simple_mean)

        if self.verbose:
            self._pbar.close()
            del self._pbar

        return Y_pred

class PBCTClassifier(PBCT):
    """Assumes 1 or 0 interaction matrix."""

    def _find_best_split(self, X, Y_means, Yvar):
        total_length = X.size
        highest_quality = 0
        # The index right before the cutoff value in the sorted X.
        cut_id = None

        # Sort Ymeans by X.
        sorted_ids = X.argsort()
        Y_means = Y_means[sorted_ids]

        # Split Y in Y1 and Y2.
        i = self.min_samples_leaf
        Y1_means_sum = Y_means[:i].sum()
        Y2_means_sum = Y_means[i:].sum()
        Y1size, Y2size = i, total_length-i

        for j in range(i, Y2size+1):  # Y2size happens to be what I needed.
            try:
                Y1var = Y1_means_sum - Y1_means_sum ** 2 / Y1size
                Y2var = Y2_means_sum - Y2_means_sum ** 2 / Y2size
                quality = 1 - (Y1var + Y2var) / total_length / Yvar
                quality = 1 - (Y1var + Y2var) / total_length / Yvar
            except:
                quality = 0

            if quality > highest_quality:
                highest_quality = quality
                cut_id = j

            current_mean = Y_means[j]
            Y1_means_sum += current_mean
            Y2_means_sum -= current_mean
            Y1size += 1
            Y2size -= 1

        if cut_id is None:
            return None
        else:
            ids1, ids2, = sorted_ids[:cut_id], sorted_ids[cut_id:]
            cutoff = (X[ids1[-1]] + X[ids2[0]]) / 2
            return highest_quality, cutoff, ids1, ids2

    def _cut_Y(self, XX, Y):
        dims = tuple(range(Y.ndim))
        all_but_i = [dims[:i] + dims[i+1:] for i in dims]
        Y_ax_means = [Y.mean(axis=ax) for ax in all_but_i]

        Ymean = Y_ax_means[0].mean()
        Yvar = Ymean - Ymean ** 2

        # best_split = (highest_quality, best_cutoff,
        #               ids1, ids2, split_axis, attribute_index)
        best_split = 0,

        for axis, X in enumerate(XX):
            X_cols = enumerate(X.T)
            if self.verbose:
                X_cols = tqdm(X_cols, desc=f'ax={axis}',
                              total=X.shape[1])

            for attr_id, attr_col in X_cols:
                split = self._find_best_split(attr_col, Y_ax_means[axis], Yvar)
                if split and (split[0] > best_split[0]):
                    best_split = split + (axis, attr_id)
        
        if best_split == (0,):
            best_split = None
        return Ymean, Y_ax_means, best_split


class PBCT2(PBCT):
    """self.tree is list of dicts, iteration over recursion."""

    def _build_tree(self, XX, Y):
        depth = 0
        root = self._make_node(XX, Y)
        root.update(parent=None, idx=0)

        tree = [root]
        tree_len = 1
        node_queue = [root] if not root['is_leaf'] else []

        while node_queue:
            parent_node = node_queue.pop()

            dep = parent_node['depth'] + 1
            XX1, XX2 = parent_node['XXXX']
            Y1, Y2 = parent_node['YY']
            del parent_node['XXXX'], parent_node['YY']

            child1 = self._make_node(XX1, Y1, depth=dep)
            child2 = self._make_node(XX2, Y2, depth=dep)

            children_idx = tree_len, tree_len + 1
            tree += [child1, child2]
            tree_len += 2

            child1['idx'], child2['idx'] = children_idx
            child1['parent'] = child2['parent'] = parent_node['idx']
            parent_node['children_idx'] = children_idx

            if not child1['is_leaf']:
                node_queue.append(child1)
            if not child2['is_leaf']:
                node_queue.append(child2)

        return tree
        
    def _predict_sample(self, *xx, simple_mean=False):
        """Predict prob. of interaction given each object's attributes.

        Predict the probability of existing interaction between two ob-
        jects from sets (xrow and xcol) of each one's attribute values.
        """
        leaf = self.tree[0]  # Initiate leaf search on root.

        while not leaf['is_leaf']:
            ax, attr_idx = leaf['coord']  # Split coordinates.
            right = xx[ax][attr_idx] >= leaf['cutoff']
            child_idx = leaf['children_idx'][right]
            leaf = self.tree[child_idx]

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

class PBCT3(PBCT):
    """self.tree nested dicts, iteration over recursion."""

    def _build_tree(self, XX, Y):
        tree = self._make_node(XX, Y)
        node_queue = [tree] if not tree['is_leaf'] else []

        while node_queue:
            parent_node = node_queue.pop()

            dep = parent_node['depth'] + 1
            XX1, XX2 = parent_node['XXXX']
            Y1, Y2 = parent_node['YY']
            del parent_node['XXXX'], parent_node['YY']

            child1 = self._make_node(XX1, Y1, depth=dep)
            child2 = self._make_node(XX2, Y2, depth=dep)
            parent_node['children'] = (child1, child2)

            if not child1['is_leaf']:
                node_queue.append(child1)
            if not child2['is_leaf']:
                node_queue.append(child2)

        return tree
        

class PBCT4(PBCT):
    """self.tree is nested dicts, itertools.product over np broadcast"""

    def fit(self, XX, Y, X_names=None):
        XX, Y, X_names = self._polish_input(XX, Y, X_names)
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


def lightness(c):
    """Calculate lighnex of hex-formatted color."""
    return colorsys.rgb_to_hls(*colors.hex2color(c))[1]


def add_node(node, graph, node_cmap, leaf_cmap, parent_id=None):
    node_id = str(hash(node))

    if node.is_leaf:
        fillcolor = leaf_cmap(node.Ymean)
        fontcolor = ('white', 'black')[lightness(fillcolor) > .5]
        graph.attr('node', shape='ellipse',
                   fillcolor=fillcolor, fontcolor=fontcolor)
    else:
        fillcolor = node_cmap(node.split_quality)
        fontcolor = ('white', 'black')[lightness(fillcolor) > .5]
        graph.attr('node', shape='box',
                   fillcolor=fillcolor, fontcolor=fontcolor)

    graph.node(node_id, str(node))

    if parent_id is not None:
        graph.edge(parent_id, node_id)

    if not node.is_leaf:
        for child in (node.little_child, node.big_child):
            add_node(child, graph, parent_id=node_id,
                     node_cmap=node_cmap, leaf_cmap=leaf_cmap)


def render_tree(node, leaf_cmap='RdYlGn', node_cmap='YlOrBr',
                format='png', *args, **kwargs):
    if isinstance(leaf_cmap, str):
        leaf_cmap=get_cmap(leaf_cmap)
    if isinstance(node_cmap, str):
        node_cmap=get_cmap(node_cmap)
    leaf_cmap_hex = lambda x: colors.rgb2hex(leaf_cmap(x))
    node_cmap_hex = lambda x: colors.rgb2hex(node_cmap(x))

    graph = graphviz.Digraph(*args, format=format, **kwargs)
    graph.attr('node', style='filled')

    add_node(node, graph,
             leaf_cmap=leaf_cmap_hex,
             node_cmap=node_cmap_hex)

    # Use name attribute to set output location.
    outpath = graph.render()
    return outpath


def gen_imatrix(shape, nattrs, func=None, nrules=5):
    if func is None:
        func, strfunc = gen_interaction_func(nattrs, nrules)
        print('Generated interaction function \n\t', strfunc)
    # shape contains the number of instances in each axis database, i.e.
    # its number of rows. nattrs contains their numbers of columns, i.e.
    # how many attributes per axis.
    XX = [np.random.rand(ni, nj) for ni, nj in zip(shape, nattrs)]
    # Create index tuples such as (np.newaxis, np.newaxis, :, np.newaxis).
    # That's because Y will be made usin numpy's broadcasting to explore
    # all combinations of x.
    ndim = len(shape)
    slices = np.tile(np.newaxis, (ndim, ndim)).astype(object)
    np.fill_diagonal(slices, slice(None))
    XXnewax = [X[tuple(s)] for X, s in zip(XX, slices)]
    # To ensure it is row-wise, not element-wise.
    # (a0),(a1),...,(a{ndim-1})->()
    sig = ','.join(f'(a{i})' for i in range(ndim)) + '->()'
    vfunc = np.vectorize(func, signature=sig)
    return XX, vfunc(*XXnewax).astype(int)


def gen_interaction_func(nattrs, nrules=10, popen=.5, pclose=.2, pand=.5):
    axes = np.random.choice(len(nattrs), nrules)
    attrs = [np.random.randint(nattrs[ax]) for ax in axes]
    cutoffs = np.random.rand(nrules)
    orands = ['and ' if i else 'or '
              for i in np.random.rand(nrules-1) < pand]
    orands.append('')

    strf = ''
    nopen = 0
    for ax, attr, cutoff, orand in zip(axes, attrs, cutoffs, orands):
       if np.random.rand() < popen: 
           strf += '( '
           nopen += 1
       strf += f'xx[{ax}][{attr}] < {cutoff} '
       if nopen and (np.random.rand() < pclose):
           strf += ') '
           nopen -= 1
       strf += orand

    strf += ')' * nopen
    return eval('lambda *xx: ' + strf), strf


def test_PBCT(shape, nattrs, classes=[PBCT], min_samples_leaf=1, **class_args):
    from time import time
    func, strfunc = gen_interaction_func(nattrs)
    print('Using following interaction rule:', strfunc)
    print('Generating synthetic data...')
    XX, Y  = gen_imatrix(shape, nattrs, func)
    times = []

    for class_ in classes:
        pbct = class_(min_samples_leaf=min_samples_leaf, **class_args)
        name = class_.__name__
        module = class_.__module__
        desc = class_.__doc__
        print(name + ': ' + str(desc))
        print('Fitting model...')
        t0 = time()
        pbct.fit(XX, Y)
        tf = time()-t0
        print(f'It took {tf} s.')

        print('Predicting...')
        t0 = time()
        Yp = pbct.predict(XX, simple_mean=True)
        tp = time()-t0
        print(f'It took {tp} s.')

        acc = ((Yp > Y.mean()) == Y).mean()
        verified = acc == 1
        if not verified:
            # raise RuntimeError(name + ' did not performed perfectly! (acc'
            #                    f'uracy == {acc} != 1, chance == {Y.mean()})')
            print(name + ' did not performed perfectly! (acc'
                  f'uracy == {acc} != 1, chance == {Y.mean()})')
        print(f'ACC == 1 : {verified}')

        times.append(dict(module=module, name=name, time_fit=tf, time_pred=tp,
                          accuracy=acc, desc=desc))

    return XX, Y, times


# Doesn't help much.
def tree_sort(node, matrix):
    if node.is_leaf:
        return matrix
    if node.axis == 'cols':
        m1, m2 = matrix[:, node.ids[0]], matrix[:, node.ids[1]]
    else:
        m1, m2 = matrix[node.ids[0]], matrix[node.ids[1]]

    m1 = tree_sort(node.little_child, m1)
    m2 = tree_sort(node.big_child, m2)

    return np.stack((m1, m2), axis=axis)


def split_LT(Xrow, Xcol, Y, Trows, Tcols):
    X_Lr, X_Tr = np.delete(Xrow, Trows, axis=0), Xrow[Trows]
    X_Lc, X_Tc = np.delete(Xcol, Tcols, axis=0), Xcol[Tcols]

    Y_TrTc = Y[Trows][:, Tcols]
    Y_LrTc = np.delete(Y, Trows, axis=0)[:, Tcols]
    Y_TrLc = np.delete(Y, Tcols, axis=1)[Trows]
    Y_LrLc = np.delete(np.delete(Y, Tcols, axis=1), Trows, axis=0)

    ret = dict(
        TrTc = (X_Tr, X_Tc, Y_TrTc),
        LrTc = (X_Lr, X_Tc, Y_LrTc),
        TrLc = (X_Tr, X_Lc, Y_TrLc),
        LrLc = (X_Lr, X_Lc, Y_LrLc),
    )

    return ret


def split_train_test(Xrows, Xcols, Y, fraction=.1):
    """Split data between train and test datasets."""

    if isinstance(fraction, float):
        fraction = fraction, fraction
    frow, fcol = fraction
    nrows, ncols = Xrows.shape[0], Xcols.shape[0]
    nrows_test, ncols_test = round(nrows * frow), round(ncols * fcol)

    # Select test indices irow and icol, respectively for each axis.
    Trows = np.random.choice(nrows, nrows_test, replace=False)
    Tcols = np.random.choice(ncols, ncols_test, replace=False)

    return split_LT(Xrows, Xcols, Y, Trows, Tcols)


def split_kfold(Xrows, Xcols, Y, k=5):
    if isinstance(k, int):
        k = k, k
    nrows, ncols = Xrows.shape[0], Xcols.shape[0]
    Xrows_idx, Xcols_idx = np.arange(nrows), np.arange(ncols)
    np.random.shuffle(Xrows_idx)
    np.random.shuffle(Xcols_idx)
    Xrows_folds_idx = np.array_split(Xrows_idx, k[0])
    Xcols_folds_idx = np.array_split(Xcols_idx, k[1])
    splits = []

    for Tcols in Xcols_folds_idx:
        for Trows in Xrows_folds_idx:
            splits.append(split_LT(Xrows, Xcols, Y, Trows, Tcols))

    return splits


def feature_importances(node,
                        ret=dict(cols={}, rows={}, total={}),
                        is_root_call=True):
    if node.is_leaf:
        return ret
    # if is_root_call:
    #     feature_importances.n_nodes = dict(rows=0, cols=0, total=0)
    #     
    # feature_importances.n_nodes['total'] += 1
    # feature_importances.n_nodes[node.split_axis] += 1
    shape = node.shape
    name = (node.split_axis, node.attr_name)
    
    # of rows, # of cols, total items.
    sizes = dict(rows=shape[0], cols=shape[1], total=shape[0]*shape[1])
    
    for key, size in sizes.items():
        ret[key][name] = ret[key].get(name, 0) + size * node.split_quality
    
    ret = feature_importances(node.big_child, ret=ret,
                              is_root_call=False)
    ret = feature_importances(node.little_child, ret=ret,
                              is_root_call=False)
    
    if is_root_call:
        ret = pd.DataFrame(ret).sort_values('total', ascending=False)
        ret /= sizes
        # ret /= feature_importances.n_nodes  # sizes already decrease with node number?
    return ret
    
    
def parse_args():
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_group = arg_parser.add_mutually_exclusive_group(required=True)
    arg_group.add_argument(
        '--fit', action='store_true',
        help='Use input data to train a PBCT.')
    arg_group.add_argument(
        '--predict', action='store_true',
        help='Predict interaction between input instances.')

    arg_parser.add_argument(
        '--XX', nargs='+',
        help=('Paths to .csv files containing rows of numerical attributes'
              ' for each axis\' instance. The first column must consist of'
              ' labels for each of them and, the first line must present n'
              'ames for each attribute.'))
    arg_parser.add_argument(
        '--Y', required=True,
        help=('If fitting the model to data (--fit), it represents the pat'
              'h to a .csv file containing the known interaction matrix be'
              'tween rows and columns data (--Xrows and --Xcols), with fir'
              'st line and first column being the respective identifiers. '
              'If --predict is given, Y is the destination path for the pr'
              'edicted values, formatted as an interaction matrix in the s'
              'ame way described for --fit.'))
    arg_parser.add_argument(
        '--model', default=PATH_MODEL,
        help=('When fitting: path to the location where the model will be '
              'saved. When predicting: the saved model to be used.'))
    arg_parser.add_argument(
        '--max_depth', default=MAX_DEPTH,
        help='Maximum PBCT depth allowed.')
    arg_parser.add_argument(
        '--min_samples_leaf', default=MIN_SAMPLES_LEAF,
        help=('Minimum number of sample pairs in the training set required'
              ' to arrive at each leaf.'))
    arg_parser.add_argument(
        '--visualize', default=PATH_RENDERING,
        help=('If provided, path to where a visualization of the trained t'
              'ree will be saved.'))

    return arg_parser.parse_args()


# FIXME: Not n-dimensional yet!
def main(path_Xrows, path_Xcols, path_Y, fit, predict,
         path_model=PATH_MODEL, max_depth=MAX_DEPTH,
         min_samples_leaf=MIN_SAMPLES_LEAF, path_render=PATH_RENDERING):
    """
    Train a PBCT or predict values from the command-line. See `parse_args()`
    or use --help for parameters' descriptions.
    """

    if fit:
        print('Loading data...')
        Xrows, Xcols, Y = [pd.read_csv(p, index_col=0)
                           for p in (path_Xrows, path_Xcols, path_Y)]
        Tree = PBCT(min_samples_leaf=min_samples_leaf, max_depth=max_depth)
        print('Training PBCT...')
        Tree.fit(Xrows, Xcols, Y)
        print('Saving model...')
        joblib.dump(Tree, path_model)
        if path_render:
            print('Rendering tree...')
            render_tree(Tree, name=path_render)
        print('Done.')

    elif predict:
        print('Loading data...')
        Xrows, Xcols = [pd.read_csv(p, index_col=0)
                           for p in (path_Xrows, path_Xcols)]
        print('Loading model...')
        Tree = joblib.load(path_model)
        print('Predicting values...')
        predictions = Tree.predict(Xrows, Xcols)
        predictions_df = pd.DataFrame(predictions,
                                      index=Xrows.index,
                                      columns=Xcols.index)
        predictions_df.to_csv(path_Y)
        print('Done.')

    else:
        raise ValueError("Either 'fit' or 'predict' must be given.")


if __name__ == '__main__':
    args = parse_args()
    main(path_Xrows=args.Xrows, path_Xcols=args.Xcols, path_Y=args.Y,
         fit=args.fit, predict=args.predict, path_model=args.model,
         max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
         path_render=args.visualize)
