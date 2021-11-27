import numpy as np

DEFAULT_K = 3, 3


# NOTE: We could simply use sklearn.model_selection._split._validate_shuffle_spl
# it in test_size and train_size, but it would add another dependency.
def parse_test_train_size(test_size, train_size, shape):
    """Determine number of samples in test set.

    Parameters
    ----------

    shape : Tuple of two integers
        Interaction matrix shape.

    test_size : float or int or tuple of two floats or ints, default=None
        If between 0.0 and 1.0, represents the proportion of the dataset to inc
        lude in the TrTc split for each axis, e.g. (.3, .5) means 30% of the ro
        ws and 50% of the columns will be used as the TrTc set. If >= 1, repres
        ents the absolute number of test samples in each axis. If None, the val
        ues are set to the complements of train_size. If a single value v is gi
        ven, it will be interpreted as (v, v). If train_size is also None, it w
        ill be set to 0.25.

    train_size : float or int or tuple of two floats or ints, default=None
        Same as test_size, but refers to the LrLc set dimensions.

    Returns
    -------
    nrows_test, ncols_test : int
        Number of rows and columns in TrTc set.
    """
    nrows, ncols = shape

    if type(test_size) in (float, int):
        nrows_test, ncols_test = test_size, test_size
    elif type(test_size) in (list, tuple):
        nrows_test, ncols_test = test_size
    elif type(train_size) in (float, int):
        nrows_test, ncols_test = train_size, train_size
    elif type(train_size) in (list, tuple):
        nrows_test, ncols_test = train_size
    else:
        test_size = True
        nrows_test, ncols_test = .25, .25

    if not (0 < nrows_test < nrows) or \
       not (0 < ncols_test < ncols):
        raise ValueError('Invalid test_size or train_size specified.')

    if test_size is not None:
        if 0 < nrows_test < 1:
            nrows_test = round(nrows_test * nrows)
        if 0 < ncols_test < 1:
            ncols_test = round(ncols_test * ncols)

    elif train_size is not None:
        if 0 < nrows_test < 1:
            nrows_test = round((1-nrows_test) * nrows)
        else:
            nrows_test = nrows - nrows_test
        if 0 < ncols_test < 1:
            ncols_test = round((1-ncols_test) * ncols)
        else:
            ncols_test = ncols - ncols_test

    return nrows_test, ncols_test


def split_LT(Xrow, Xcol, Y, test_rows, test_cols):
    """Select train/test datasets from X attribute tables, from test indices."""

    X_Lr, X_Tr = np.delete(Xrow, test_rows, axis=0), Xrow[test_rows]
    X_Lc, X_Tc = np.delete(Xcol, test_cols, axis=0), Xcol[test_cols]

    Y_TrTc = Y[test_rows][:, test_cols]
    Y_LrTc = np.delete(Y, test_rows, axis=0)[:, test_cols]
    Y_TrLc = np.delete(Y, test_cols, axis=1)[test_rows]
    Y_LrLc = np.delete(np.delete(Y, test_cols, axis=1), test_rows, axis=0)

    ret = dict(
        TrTc = ((X_Tr, X_Tc), Y_TrTc),
        LrTc = ((X_Lr, X_Tc), Y_LrTc),
        TrLc = ((X_Tr, X_Lc), Y_TrLc),
        LrLc = ((X_Lr, X_Lc), Y_LrLc),
    )

    return ret


def train_test_split(Xrows, Xcols, Y, test_size=None, train_size=None):
    """Split data between train and test datasets.

    Parameters
    ----------
    Xrows, Xcols : NDArray
        Attribute matrices for row and column instances, respectively.

    Y : NDArray
        Interaction matrix.

    test_size : float or int or tuple of two floats or ints, default=None
        If between 0.0 and 1.0, represents the proportion of the dataset to inc
        lude in the TrTc split for each axis, e.g. (.3, .5) means 30% of the ro
        ws and 50% of the columns will be used as the TrTc set. If >= 1, repres
        ents the absolute number of test samples in each axis. If None, the val
        ues are set to the complements of train_size. If a single value v is gi
        ven, it will be interpreted as (v, v). If train_size is also None, it w
        ill be set to 0.25.

    train_size : float or int or tuple of two floats or ints, default=None
        Same as test_size, but refers to the LrLc set dimensions.
    """
#    shuffle : bool, default=True
#        Whether or not to shuffle the data before splitting.

    nrows, ncols = Y.shape
    nrows_test, ncols_test = parse_test_train_size(test_size, train_size, Y.shape)
    # Select test indices test_rows and test_cols, respectively for each axis.
    test_rows = np.random.choice(nrows, nrows_test, replace=False)
    test_cols = np.random.choice(ncols, ncols_test, replace=False)

    return split_LT(Xrows, Xcols, Y, test_rows, test_cols)


def kfold_split(Xrows, Xcols, Y, k=None, diag=False):
    """Split 2D data into folds for cross validation.
    
    Parameters
    ----------
    Xrows, Xcols : NDArray
        Attribute matrices for row and column instances, respectively.

    Y : NDArray
        Interaction matrix.

    k : int or Tuple[int, int]
        If diag is True, is the number of folds. Otherwise it's the number of
        divisions in each axis. For instance, k = (4, 6) means the matrix will
        be divided in 4 groups of rows and 6 groups of columns. If k is integer,
        it will be interpreted as (k, k).

    diag : bool
        If True, use independent TrTc sets, with no overlapping rows or columns.
    """
    if k is None:
        k = DEFAULT_K
    elif isinstance(k, int):
        k = k, k
    elif diag:
        raise ValueError('k must be an integer if diag is given.')

    nrows, ncols = Xrows.shape[0], Xcols.shape[0]
    Xrows_idx, Xcols_idx = np.arange(nrows), np.arange(ncols)
    np.random.shuffle(Xrows_idx)
    np.random.shuffle(Xcols_idx)
    Xrows_folds_idx = np.array_split(Xrows_idx, k[0])
    Xcols_folds_idx = np.array_split(Xcols_idx, k[1])
    splits = []

    if diag:
        for test_cols, test_rows in zip(Xcols_folds_idx, Xrows_folds_idx):
            splits.append(split_LT(Xrows, Xcols, Y, test_rows, test_cols))
    else:
        for test_cols in Xcols_folds_idx:
            for test_rows in Xrows_folds_idx:
                splits.append(split_LT(Xrows, Xcols, Y, test_rows, test_cols))

    return splits