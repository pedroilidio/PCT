#!/bin/env python
"""Provides a PBCT classifier implementation and related tools.

First version, uses Node classes referencing eachother to build
the tree. Only 2D interaction matrices supported.

Classes
-------
    Node, PBCT
"""
# TODO: Cost-complexity prunning.
# TODO: Lookahead.
# TODO: There are duplicate instances which receives different labels
#       in database. Preprocessing it (i.e. averaging) may enhance performance.
#       Changes in Node.predict must follow.
import pandas as pd
import numpy as np
import joblib
import graphviz
import argparse
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

    def __init__(self, parent=None, depth=0, max_depth=MAX_DEPTH,
                 min_samples_leaf=MIN_SAMPLES_LEAF, min_quality=0,
                 verbose=False):
        """Instantiate new PBCT node."""
        self.parent = parent
        self.depth = depth
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_quality = min_quality
        self.verbose = verbose

    def find_best_split(self, X, Y_means, Y_sq_means, Yvar):
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
            print('\r', len(ids1), len(ids2), end=' | ', flush=True)
            cutoff = (X[ids1[-1]] + X[ids2[0]]) / 2
            return highest_quality, cutoff, ids1, ids2

    def find_best_attr(self, Xrows, Xcols, Yvar,
                       Yrow_means, Yrow_sq_means,
                       Ycol_means, Ycol_sq_means):
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
            Direction : {'rows', 'cols'}
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
        #               ids1, ids2, direction, attribute_index)
        best_split = 0,
        indent = '  ' * self.depth + str(self.depth) + '|'
        Xrows_iter = enumerate(Xrows.T)
        Xcols_iter = enumerate(Xcols.T)
        if self.verbose:
            Xrows_iter = tqdm(Xrows_iter, desc=indent+'rows', total=Xrows.shape[1])
            Xcols_iter = tqdm(Xcols_iter, desc=indent+'cols', total=Xcols.shape[0])

        for attr_id, attr in Xrows_iter:
            split = self.find_best_split(attr, Yrow_means, Yrow_sq_means, Yvar)
            if split and (split[0] > best_split[0]):
                best_split = split + ('rows', attr_id)

        for attr_id, attr in Xcols_iter:
            split = self.find_best_split(attr, Ycol_means, Ycol_sq_means, Yvar)
            if split and (split[0] > best_split[0]):
                best_split = split + ('cols', attr_id)

        if best_split == (0,):
            return None
        else:
            return best_split

    def fit(self, Xrows, Xcols, Y=None,
            Yrow_means=None, Yrow_sq_means=None,
            Ycol_means=None, Ycol_sq_means=None,
            Xrows_names=None, Xcols_names=None,
            store_ids=False):
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

        # TODO: should we ignore pandas at all?
        if isinstance(Y, pd.DataFrame):
            if Xrows_names is None:
                Xrows_names = Xrows.columns
            if Xcols_names is None:
                Xcols_names = Xcols.columns
            Y = Y.values
            Xrows = Xrows.values
            Xcols = Xcols.values

        if (Xrows_names is not None) and not isinstance(Xrows_names[0], str):
            raise ValueError('Xrows_names must be a list-like of string labels.')
        if (Xcols_names is not None) and not isinstance(Xcols_names[0], str):
            raise ValueError('Xcols_names must be a list-like of string labels.')

        # If a Y-related argument is missing, obtain it from Y.

        # It is faster to resquare it here than to index the parent's Ysq.
        Ysq = Y ** 2

        if (Yrow_means is None) or (Yrow_sq_means is None):
            Yrow_means, Yrow_sq_means = Y.mean(axis=1), Ysq.mean(axis=1)
        if (Ycol_means is None) or (Ycol_sq_means is None):
            Ycol_means, Ycol_sq_means = Y.mean(axis=0), Ysq.mean(axis=0)

        Ymean, Ysq_mean = Yrow_means.mean(), Yrow_sq_means.mean()
        Yvar = Ysq_mean - Ymean ** 2
        self.shape = Xrows.shape[0], Xcols.shape[0]

        if (self.depth == self.max_depth) or (Yvar == 0):
            self.is_leaf = True
        else:
            best_split = self.find_best_attr(Xrows, Xcols, Yvar,
                                             Yrow_means, Yrow_sq_means,
                                             Ycol_means, Ycol_sq_means)
            self.is_leaf = (best_split is None) or (best_split[0] < self.min_quality)

        if self.is_leaf:
            self.Ymean = Ymean
            self.Yrow_means = Yrow_means
            self.Ycol_means = Ycol_means
            self.Xrows = Xrows
            self.Xcols = Xcols
            return

        # else:
        q, cutoff, ids1, ids2, direction, attr_id = best_split

        self.split_direction = direction
        self.attr_id = attr_id
        self.cutoff = cutoff
        self.split_quality = q
        if store_ids:  # TODO: Store in each child instead?
            self.ids = ids1, ids2

        # TODO: Consider using dict, code seems duplicated.
        if direction == 'cols':
            if Xcols_names is None:
                self.attr_name = f'[{attr_id}]'
            else:
                self.attr_name = Xcols_names[attr_id]
            Xcols1, Xcols2 = Xcols[ids1], Xcols[ids2]
            Xrows1, Xrows2 = Xrows, Xrows
            Ycol_means1, Ycol_means2 = Ycol_means[ids1], Ycol_means[ids2]
            Ycol_sq_means1, Ycol_sq_means2 = Ycol_sq_means[ids1], Ycol_sq_means[ids2]
            Yrow_means1, Yrow_means2, Yrow_sq_means1, Yrow_sq_means2 = (None,)*4
            Y1, Y2 = Y[:, ids1], Y[:, ids2]

        elif direction == 'rows':
            if Xrows_names is None:
                self.attr_name = f'[{attr_id}]'
            else:
                self.attr_name = Xrows_names[attr_id]
            Xcols1, Xcols2 = Xcols, Xcols
            Xrows1, Xrows2 = Xrows[ids1], Xrows[ids2]
            Yrow_means1, Yrow_means2 = Yrow_means[ids1], Yrow_means[ids2]
            Yrow_sq_means1, Yrow_sq_means2 = Yrow_sq_means[ids1], Yrow_sq_means[ids2]
            Ycol_means1, Ycol_means2, Ycol_sq_means1, Ycol_sq_means2 = (None,)*4
            Y1, Y2 = Y[ids1], Y[ids2]

        # Could use copy.deepcopy.
        self.little_child = Node(
            parent=self,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            min_quality=self.min_quality,
            depth=self.depth + 1)
        self.big_child = Node(
            parent=self,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            min_quality=self.min_quality,
            depth=self.depth + 1)

        self.little_child.fit(
            store_ids=store_ids,
            Xrows=Xrows1, Xcols=Xcols1, Y=Y1,
            Yrow_means=Yrow_means1, Yrow_sq_means=Yrow_sq_means1,
            Ycol_means=Ycol_means1, Ycol_sq_means=Ycol_sq_means1,
            Xrows_names=Xrows_names, Xcols_names=Xcols_names)
        self.big_child.fit(
            store_ids=store_ids,
            Xrows=Xrows2, Xcols=Xcols2, Y=Y2,
            Yrow_means=Yrow_means2, Yrow_sq_means=Yrow_sq_means2,
            Ycol_means=Ycol_means2, Ycol_sq_means=Ycol_sq_means2,
            Xrows_names=Xrows_names, Xcols_names=Xcols_names)

    def predict_sample(self, xrow, xcol, simple_mean=False):
        """Predict prob. of interaction given each object's attributes.

        Predict the probability of existing interaction between two ob-
        jects from sets (xrow and xcol) of each one's attribute values.
        """

        if self.is_leaf:
            if simple_mean:
                return self.Ymean
            known_rows = (self.Xrows == xrow).all(axis=1)
            known_cols = (self.Xcols == xcol).all(axis=1)

            if known_rows.any():
                return self.Yrow_means[known_rows].mean()
            elif known_cols.any():
                return self.Ycol_means[known_cols].mean()
            else:
                return self.Ymean

        else:
            direction = self.split_direction == 'cols'
            a = (xrow, xcol)[direction][self.attr_id] >= self.cutoff
            child = (self.little_child, self.big_child)[int(a)]
            return child.predict_sample(xrow, xcol)

    def predict(self, Xrow, Xcol):
        """Predict prob. of interaction between rows and columns objects.

        Predict the probability of ocurring interaction between two arrays
        of column and row instances. Each array is assumed to contain o-
        ther array-like objects consisting of attribute values for each
        instance.

        Parameters
        ----------
            Xrow, Xcol : array-like
                List of lists of instances' attributes, for instances of
                the interaction matrix rows' type and to the columns' type,
                respectively.
        """
        range_ = (range, trange)[self.verbose]

        if isinstance(Xrow, pd.DataFrame):
            Xrow = Xrow.values
        if isinstance(Xcol, pd.DataFrame):
            Xcol = Xcol.values

        I, J = Xrow.shape[0], Xcol.shape[0]
        Y = np.zeros((I, J))
        for i in range_(I):
            for j in range_(J):
                v = self.predict_sample(Xrow[i], Xcol[j])
                Y[i, j] = v
        return Y

    def __str__(self):
        if self.is_leaf:
            return (f'Mean={self.Ymean:.6f}\n'
                    f'Shape={self.shape}')

        else:
            return (f'Direction={self.split_direction}\n'
                    f'{self.attr_name} >= {self.cutoff:.6f}\n'
                    f'Quality={self.split_quality:.6f}\n'
                    f'Shape={self.shape}\n')

    def __repr__(self):
        return '\n\t'.join('PBCT.Node(', str(self), ')')


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


def gen_imatrix(func, shape, xwidth, ywidth, xrange=1, yrange=1):
    Y = np.zeros(shape)
    Xx = np.random.rand(shape[0], xwidth) * xrange
    Xy = np.random.rand(shape[1], ywidth) * yrange
    for ix, x in enumerate(Xx):
        for iy, y in enumerate(Xy):
            Y[ix, iy] = func(x, y)
    return Xx, Xy, Y


# Doesn't help much.
def tree_sort(node, matrix):
    if node.is_leaf:
        return matrix
    if node.direction == 'cols':
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
    # feature_importances.n_nodes[node.split_direction] += 1
    shape = node.shape
    name = (node.split_direction, node.attr_name)
    
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
