from copy import deepcopy
import multiprocessing as mp
import numpy as np
import PBCT


def fit_and_test(model, split):
    model.fit(*split['LrLc'])
    return {LT: model.predict(XX)
            for LT, (XX, Y) in split.items() if LT != 'LrLc'}


def split_fit_test(XX, Y, model, **kwargs):
    split = PBCT.split_data.train_test_split(*XX, Y, **kwargs)
    return split, fit_and_test(model, split)


def cross_validate_2D(XX, Y, model, k=3, diag=False, njobs=None):
    splits = PBCT.split_data.kfold_split(*XX, Y, k=k, diag=diag)
    models = [deepcopy(model) for _ in splits]
    # predictions = [
    #     fit_and_test(model, split)
    #     for model, split in zip(models, splits)
    # ]
    with mp.Pool(njobs) as pool:
        predictions = pool.starmap(fit_and_test, zip(models, splits))

    return splits, models, predictions


def save_split(split, dir_data):
    dir_data.mkdir()
    for LT, data in split.items():
        dir_LT = dir_data/LT
        dir_LT.mkdir()
        (x1, x2), y = data
        np.savetxt(dir_LT/'X1.csv', x1, delimiter=',')
        np.savetxt(dir_LT/'X2.csv', x2, delimiter=',')
        np.savetxt(dir_LT/'Y.csv', y, delimiter=',', fmt='%d')
