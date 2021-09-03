#!/bin/env python
"""Provides a PBCT classifier implementation and related tools.

Second version, implements n-dimensional interaction matrix support.

Classes
-------
    Node, PBCT
"""
# TODO: Cost-complexity prunning.
# TODO: Lookahead.
# TODO: There are duplicate instances which receives different labels
#       in database. Preprocessing it (i.e. averaging) may enhance performance.
#       Changes in Node.predict must follow.
import argparse
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


# class Node(ClassifierMixin):  # Reduce dependencies.
class Node():
    """Create a new predictive bi-clustering tree (PBCT) node."""

    def __init__(self, parent=None, max_depth=MAX_DEPTH,
                 min_samples_leaf=MIN_SAMPLES_LEAF, split_min_quality=0,
                 leaf_target_quality=1, max_variance=0, verbose=False):
        """Instantiate new PBCT node."""
        if parent is None:
            self._depth = 0
            self.min_samples_leaf = min_samples_leaf
            self.max_depth = max_depth
            self.split_min_quality = split_min_quality
            self.leaf_target_quality = leaf_target_quality
            self.max_variance = max_variance
            self.verbose = verbose
        else:
            self.parent = parent
            self._depth = parent._depth + 1
            self.min_samples_leaf = parent.min_samples_leaf
            self.max_depth = parent.max_depth
            self.split_min_quality = parent.split_min_quality
            self.leaf_target_quality = parent.leaf_target_quality
            self.max_variance = parent.max_variance
            self.verbose = parent.verbose

            self.X_names = parent.X_names
            self.root = parent.root

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

        if cut_id is None:
            return None
        else:
            ids1, ids2, = sorted_ids[:cut_id], sorted_ids[cut_id:]
            cutoff = (X[ids1[-1]] + X[ids2[0]]) / 2
            return highest_quality, cutoff, ids1, ids2

    def _find_best_attr(self, XX, Yvar, _Y_ax_means, _Ysq_ax_means):
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

        # best_split = (highest_quality, best_cutoff,
        #               ids1, ids2, split_axis, attribute_index)
        best_split = 0,
        if self.verbose:
            indent = '  ' * self._depth + str(self._depth) + '|'

        for axis, X in enumerate(XX):
            X_cols = enumerate(X.T)
            if self.verbose:
                X_cols = tqdm(X_cols, desc=f'{indent}ax={axis}',
                              total=X.shape[1])

            for attr_id, attr_col in X_cols:
                split = self._find_best_split(attr_col, _Y_ax_means[axis],
                                             _Ysq_ax_means[axis], Yvar)
                if split and (split[0] > best_split[0]):
                    best_split = split + (axis, attr_id)

        if best_split == (0,):
            return None
        else:
            return best_split

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

    def fit(self, XX, Y=None, X_names=None,
            _Y_ax_means=None, _Ysq_ax_means=None, store_ids=False):
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
        if self._depth == 0:
            XX, Y, X_names = self._polish_input(XX, Y, X_names)
            self.X_names = X_names
            self.root = self
            self.n_nodes = 1

            # Line bellow is kinda cumbersome. It vectorizes the predict
            # function, so that the predicted interaction matrix will be
            # created with numpy's broadcasting. It needs to happen here,
            # after we know the number of dimensions we are dealing with,
            # because the signature argument needs it and we need the
            # signature argument for the predict function to approach the
            # data row-wisely.

            # TODO: Use dims variable?
            # (a0),(a1),...,(a{ndim-1})->()
            sig = ','.join(f'(a{i})' for i in range(Y.ndim)) + '->()'
            self._v_predict_sample = np.vectorize(self._predict_sample,
                                                  signature=sig,
                                                  excluded=['simple_mean'])

        # It is faster to resquare it here than to index the parent's Ysq.
        Ysq = Y ** 2

        dims = tuple(range(Y.ndim))  # TODO: Could share with children.
        all_but_i = [dims[:i] + dims[i+1:] for i in dims]
        _Y_ax_means = [Y.mean(axis=ax) for ax in all_but_i]
        _Ysq_ax_means = [Ysq.mean(axis=ax) for ax in all_but_i]

        Ymean, Ysq_mean = _Y_ax_means[0].mean(), _Ysq_ax_means[0].mean()
        Yvar = Ysq_mean - Ymean ** 2
        self.shape = Y.shape

        if (self._depth == self.max_depth) or (Yvar <= self.max_variance):
            self.is_leaf = True
        else:
            best_split = self._find_best_attr(XX, Yvar,
                                              _Y_ax_means, _Ysq_ax_means)
            # Node is leaf if no split was found to reduce Y variance or
            # if split quality already satisfies the defined target or
            # if no split satisfied minimum required quality.
            self.is_leaf = best_split is None
            self.is_leaf |= best_split[0] > self.leaf_target_quality
            self.is_leaf |= best_split[0] < self.split_min_quality

        if self.is_leaf:
            self.Ymean = Ymean
            self._Y_ax_means = _Y_ax_means
            self.XX = XX
            return

        # else:
        q, cutoff, ids1, ids2, axis, attr_id = best_split

        self.split_axis = axis
        self.attr_id = attr_id
        self.cutoff = cutoff
        self.split_quality = q
        self.attr_name = self.X_names[axis][attr_id]

        if store_ids:
            self.ids = ids1, ids2

        XX1, XX2 = XX.copy(), XX.copy()
        XX1[axis] = XX[axis][ids1]
        XX2[axis] = XX[axis][ids2]

        Y_slice1 = [slice(None)] * Y.ndim
        Y_slice2 = [slice(None)] * Y.ndim
        Y_slice1[axis] = ids1
        Y_slice2[axis] = ids2
        Y1, Y2 = Y[tuple(Y_slice1)], Y[tuple(Y_slice2)]

        # TODO: On the split_axis, you can only reindex means, no need
        # to calculate again at child nodes.

        self.little_child = Node(parent=self)
        self.big_child = Node(parent=self)

        self.little_child.fit(XX=XX1, Y=Y1)
        self.big_child.fit(XX=XX2, Y=Y2)
        self.root.n_nodes += 2

    def _predict_sample(self, *xx, simple_mean=False):
        """Predict prob. of interaction given each object's attributes.

        Predict the probability of existing interaction between two ob-
        jects from sets (xrow and xcol) of each one's attribute values.
        """
        if self.is_leaf:
            if self.verbose:
                self.root._pbar.update()
            if simple_mean:
                return self.Ymean

            # Search for known instances in training set (XX) and use their
            # interaction probability as result if any was found.
            # TODO: Assume there is only one known instance at most, i.e.
            # use some "get_index" instead of a boolean mask.
            Y_slice = []
            any_known = False

            for X, x in zip(self.XX, xx):
                known = (X == x).all(axis=1)
                if known.any():
                    any_known = True
                    Y_slice.append(kwown)
                else:
                    Y_slice.append(slice(None))

            if any_known:
                return self.Y[tuple(Y_slice)].mean()
            else:
                return self.Ymean

        else:
            big = xx[self.split_axis][self.attr_id] >= self.cutoff
            child = self.big_child if big else self.little_child
            return child._predict_sample(*xx, simple_mean=simple_mean)

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

    def __str__(self):
        if self.is_leaf:
            return (f'Mean={self.Ymean:.6f}\n'
                    f'Shape={self.shape}')

        else:
            return (f'axis={self.split_axis}\n'
                    f'{self.attr_name} >= {self.cutoff:.6f}\n'
                    f'Quality={self.split_quality:.6f}\n'
                    f'Shape={self.shape}\n')

    def __repr__(self):
        return '\n\t'.join(('PBCT.Node(', *str(self).split(), ')'))


# Alias.
class PBCT(Node):
    pass


def load_data():
    print('Loading data...')
    mrna_cols = ('#GeneID', 'ORFloglength', '5\'UTRloglength', '3\'UTRloglenth',
                 'OrfAUPercentage', '5\'UTRAUPercentage', '3\'UTRAUPercentage')
    mrna = pd.read_csv('FeatureInformation.csv', sep=';', usecols=mrna_cols,
                       index_col='#GeneID')
    int_matrix = pd.read_csv('hSa_MicroRNADataset.csv', sep=';', index_col=0)
    mirna = pd.read_table('miRNA_features.tsv', header=[0, 1, 2], index_col=0)

    print('Intersecting...')
    common_mirna = mirna.index.intersection(int_matrix.columns)
    common_mrna = mrna.index.intersection(int_matrix.index)

    mirna = mirna.loc[common_mirna]
    mrna = mrna.loc[common_mrna]
    int_matrix = int_matrix.loc[common_mrna, common_mirna]

    print('Done.')
    return mrna, mirna, int_matrix


def load_mini_data(nt=1000, nm=1000):
    t, m, i = load_data()
    minit = t.iloc[:nt]
    minim = m.iloc[:nm]
    minii = i.loc[minit.index, minim.index]
    return minit, minim, minii


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


def gen_imatrix(func, shape, nattrs):
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
        '--Xrows', required=True,
        help=('Path to a .csv file containing rows of numerical attributes'
              ' for each ROW instance. The first column must consist of la'
              'bels for each of them and, the first line must present name'
              's for each attribute.'))
    arg_parser.add_argument(
        '--Xcols', required=True,
        help=('Path to a .csv file containing rows of numerical attributes'
              ' for each COLUMN instance. The first column must consist of'
              ' labels for each of them, and the first line must present n'
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
