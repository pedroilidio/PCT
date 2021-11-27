from pathlib import Path
from time import time
from warnings import warn
from PBCT import PBCT, PBCTClassifier
from PBCT.experimental_classes import SSPBCT, PBCT2
from make_examples import gen_interaction_func, gen_imatrix
import numpy as np


def test_PBCT(shape, nattrs, classes=[PBCT], class_args=None,
              global_class_args=None, saveload=True, n_jobs_pred=1):

    if class_args is None:
        class_args = [{} for c in classes]
    if global_class_args is None:
        global_class_args = {}

    # Set min_samples_leaf=1 as default.
    global_class_args = dict(min_samples_leaf=1) | global_class_args

    func, strfunc = gen_interaction_func(nattrs)
    print('Using following interaction rule:', strfunc)
    print('Generating synthetic data...', end=' ')
    XX, Y  = gen_imatrix(shape, nattrs, func)
    print('Done.')
    times = []
    models = []

    for class_, cargs in zip(classes, class_args):
        params = global_class_args | cargs
        if type(class_) is type:
            pbct = class_(**params)
        else:
            pbct = class_
            class_ = pbct.__class__
        name = class_.__name__
        module = class_.__module__
        desc = class_.__doc__
        print('\n' + name + ': ' + str(desc))
        print('Fitting model...')
        t0 = time()
        pbct.fit(XX, Y)
        tf = time()-t0
        print(f'It took {tf} s.')

        if saveload:
            print('Testing save/load...', end=' ')
            path_model = Path('model.dict.joblib')
            pbct.save(path_model)
            pbct = class_.load(path_model)
            path_model.unlink()
            print('OK.')

        print('Predicting...', end=' ')
        t0 = time()
        Yp = pbct.predict(XX, n_jobs=n_jobs_pred)
        tp = time()-t0
        print(f'It took {tp} s.')

        acc = ((Yp > Y.mean()) == Y).mean()
        verified = acc == 1

        if not verified:
            density = Y.mean()
            tol = 1e-6
            print('TEST FAILED --------------')
            if (density-tol) < acc < (density+tol):
                print('\tMIXED')
                print('\t' + name, ('mixed the labels instead of '
                      'separating them!'))
                print(f'\td: {density} == {acc} :acc')
            elif (1-tol) < (acc + density) < (1+tol):
                print('\tSWAPPED')
                print('\t' + name, 'apparently swapped labels!') 
                print(f'\tacc + density == {acc+density}')
            else:
                print('\t' + name, 'did not performed perfectly!')

            print(f'\taccuracy == {acc} != 1')
            print(f'\tY density (baseline) ==', density)
            print('--------------------------')
        else:
            print(f'SUCESS - ACC: {acc} == 1 : {verified}')

        print(f'# of nodes: {pbct.n_nodes}')
        print(f'# of leaves: {pbct.n_leaves}')

        models.append(pbct)
        times.append(dict(module=module, name=name, time_fit=tf, time_pred=tp,
                          accuracy=acc, desc=desc, params=params))

    return XX, Y, times, models


def main():
    np.random.seed(0)
    #print('Always remember to set min_samples_leaf=1.')
    return test_PBCT(
        shape=(1000, 1000),
        nattrs=(50, 50),
        # shape=(100, 100),
        # nattrs=(5, 5),
        classes=[
            #PBCT2,
            SSPBCT(supervision=0, min_samples_leaf=5),
            #SSPBCT(supervision=.5, min_samples_leaf=1),
            #SSPBCT(supervision=.1, min_samples_leaf=1),
        ],
        global_class_args=dict(min_samples_leaf=1),
        saveload=False,
    )


if __name__ == "__main__":
    main()
