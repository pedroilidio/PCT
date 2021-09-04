from time import time
from PBCT import PBCT, PBCTClassifier, load_model
from PBCT.classes import ax_means
from make_examples import gen_interaction_func, gen_imatrix


def test_PBCT_ax_predict(shape, nattrs, classes=[PBCT], min_samples_leaf=1, **class_args):
    func, strfunc = gen_interaction_func(nattrs)
    print('Using following interaction rule:', strfunc)
    print('Generating synthetic data...', end=' ')
    XX, Y  = gen_imatrix(shape, nattrs, func)
    print('Done.')
    times = []

    for class_ in classes:
        pbct = class_(min_samples_leaf=min_samples_leaf, **class_args)
        name = class_.__name__
        module = class_.__module__
        desc = class_.__doc__
        print('\n' + name + ': ' + str(desc))
        print('Fitting model...')
        t0 = time()
        pbct.fit(XX, Y)
        tf = time()-t0
        print(f'It took {tf} s.')

        ax_preds = []
        for ax in (0, 1):
            print(f'Predicting with axis {ax}...', end=' ')
            t0 = time()
            ax_preds.append(pbct.ax_predict(XX[ax], axis=ax))
            tp = time()-t0
            print(f'It took {tp} s.')

        distances = []
        for pred, means in zip(ax_preds, ax_means(Y)):
            distances.append((pred-means) ** 2)

        verified = all(d.all() == 0 for d in distances)

        if not verified:
            raise RuntimeError(name + ' did not performed perfectly! There are'
                               f' non-zero distances: {distances})')
        print('Everything OK!')

        times.append(dict(module=module, name=name, time_fit=tf, time_pred=tp,
                          distances=distances, desc=desc))

    return XX, Y, times


def main():
    test_PBCT_ax_predict(
        shape=(1000, 1000),
        nattrs=(500, 500),
        min_samples_leaf=1
    )


if __name__ == "__main__":
    main()
