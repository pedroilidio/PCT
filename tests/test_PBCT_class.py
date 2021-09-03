from time import time
from PBCT import PBCT, PBCTClassifier


def gen_imatrix(shape, nattrs, func=None, nrules=5):
    if func is None:
        func, strfunc = gen_interaction_func(nattrs, nrules)
        print('Generated interaction function \n\t', strfunc)
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


def gen_interaction_func(nattrs, nrules=10, popen=.5, pclose=.2, pand=.5):
    axes = np.random.choice(len(nattrs), nrules)
    attrs = [np.random.randint(nattrs[ax]) for ax in axes]
    cutoffs = np.random.rand(nrules)
    orands = ['and ' if i else 'or '
              for i in np.random.rand(nrules-1) < pand]
    orands.append('')

    strf = ''
    nopen = 0
    for ax, attr, cutoff, orand in zip(axes, attrs, cutoffs, orands):
       if np.random.rand() < popen: 
           strf += '( '
           nopen += 1
       strf += f'xx[{ax}][{attr}] < {cutoff} '
       if nopen and (np.random.rand() < pclose):
           strf += ') '
           nopen -= 1
       strf += orand

    strf += ')' * nopen
    return eval('lambda *xx: ' + strf), strf


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
            # raise RuntimeError(name + ' did not performed perfectly! (acc'
            #                    f'uracy == {acc} != 1, chance == {Y.mean()})')
            print(name + ' did not performed perfectly! (acc'
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
