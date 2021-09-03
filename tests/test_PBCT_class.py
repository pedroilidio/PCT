from time import time
from PBCT import PBCT, PBCTClassifier
from make_examples import gen_interaction_func, gen_imatrix


def test_PBCT(shape, nattrs, classes=[PBCT], min_samples_leaf=1, **class_args):
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
            raise RuntimeError(name + ' did not performed perfectly! (acc'
                               f'uracy == {acc} != 1, chance == {Y.mean()})')
        print(f'ACC == 1 : {verified}')

        times.append(dict(module=module, name=name, time_fit=tf, time_pred=tp,
                          accuracy=acc, desc=desc))

    return XX, Y, times


def main():
    test_PBCT(
        shape=(1000, 1000),
        nattrs=(500, 500),
        classes=[PBCT, PBCTClassifier],
        min_samples_leaf=1
    )


if __name__ == "__main__":
    main()
