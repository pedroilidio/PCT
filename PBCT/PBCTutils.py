# TODO: improve documentation.
import numpy as np


def split_LT(Xrow, Xcol, Y, Trows, Tcols):
    """Select train/test datasets from X attribute tables, from test indices."""
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

    # Select test indices Trows and Tcols, respectively for each axis.
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
