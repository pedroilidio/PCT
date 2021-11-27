import contextlib
from copy import deepcopy
from joblib import Parallel, delayed
# import joblib
# from tqdm.auto import tqdm
import numpy as np
import PBCT


# https://stackoverflow.com/a/58936697/11286509
# @contextlib.contextmanager
# def tqdm_joblib(tqdm_object):
#     """Context manager to patch joblib to report into tqdm progress bar given as argument"""
#     class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
# 
#         def __call__(self, *args, **kwargs):
#             tqdm_object.update(n=self.batch_size)
#             return super().__call__(*args, **kwargs)
# 
#     old_batch_callback = joblib.parallel.BatchCompletionCallBack
#     joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
#     try:
#         yield tqdm_object
#     finally:
#         joblib.parallel.BatchCompletionCallBack = old_batch_callback
#         tqdm_object.close()
        

def fit_and_test(model, split, predict_lrlc=False):
    model.fit(*split['LrLc'])
    return {LT: model.predict(XX)
            for LT, (XX, Y) in split.items()
            if predict_lrlc or LT != 'LrLc'}


def split_fit_test(XX, Y, model, **kwargs):
    split = PBCT.split_data.train_test_split(*XX, Y, **kwargs)
    return split, fit_and_test(model, split)


def cross_validate_2D(XX, Y, model, k=None, diag=False,
                      n_jobs=None, prefer=None, verbose=0):
    splits = PBCT.split_data.kfold_split(*XX, Y, k=k, diag=diag)
    models = [deepcopy(model) for _ in splits]
    
    # with tqdm_joblib(tqdm(total=len(splits))):
    #     predictions = Parallel(n_jobs, prefer=prefer)(
    #         delayed(fit_and_test)(model, split)
    #         for model, split in zip(models, splits)
    #     )
    predictions = Parallel(n_jobs, verbose=verbose, prefer=prefer)(
        delayed(fit_and_test)(model, split)
        for model, split in zip(models, splits)
    )
    
    return dict(folds=splits, models=models, predictions=predictions)


def save_split(split, dir_data):
    dir_data.mkdir()
    for LT, data in split.items():
        dir_LT = dir_data/LT
        dir_LT.mkdir()
        (x1, x2), y = data
        np.savetxt(dir_LT/'X1.csv', x1, delimiter=',')
        np.savetxt(dir_LT/'X2.csv', x2, delimiter=',')
        np.savetxt(dir_LT/'Y.csv', y, delimiter=',', fmt='%d')
