import contextlib
from copy import deepcopy
from joblib import Parallel, delayed
# import joblib
# from tqdm.auto import tqdm
import numpy as np
import PBCT


def fit_and_test(model, split, predict_lrlc=False):
    model.fit(*split['LrLc'])
    return {LT: model.predict(XX)
            for LT, (XX, Y) in split.items()
            if predict_lrlc or LT != 'LrLc'}


def split_fit_test(XX, Y, model, **kwargs):
    split = PBCT.split_data.train_test_split(*XX, Y, **kwargs)
    return split, fit_and_test(model, split)


def cross_validate_2D(
        XX, Y, model, k=None, diag=False,
        n_jobs=None, prefer=None,
        seed=None, verbose=0):
    splits = PBCT.split_data.kfold_split(*XX, Y, k=k, diag=diag, seed=seed)
    models = [deepcopy(model) for _ in splits]

    predictions = Parallel(n_jobs, verbose=verbose, prefer=prefer)(
        delayed(fit_and_test)(model, split)
        for model, split in zip(models, splits)
    )

    return dict(folds=splits, models=models, predictions=predictions)


def save_split(split, dir_data, fmt_x='%f', fmt_y='%f'):
    dir_data.mkdir()
    for LT, data in split.items():
        dir_LT = dir_data/LT
        dir_LT.mkdir()
        (x1, x2), y = data
        np.savetxt(dir_LT/'X1.csv', x1, delimiter=',', fmt=fmt_x)
        np.savetxt(dir_LT/'X2.csv', x2, delimiter=',', fmt=fmt_x)
        np.savetxt(dir_LT/'Y.csv', y, delimiter=',', fmt=fmt_y)
