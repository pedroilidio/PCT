from timeit import repeat
import numpy as np
import numba
from PBCT.experimental_classes import (
    _find_best_split_ss,
    _find_best_split as _find_best_split_ec)
from PBCT.classes import _find_best_split
from make_examples import gen_interaction_func, gen_imatrix

@numba.njit(fastmath=True)
def _find_best_split_welford_q(attr_col, Y_means, Yvar, min_samples_leaf):
    """It uses more memory and computes unnecessary stuff, but it is readable."""
    # FIXME: You've called it "min_samples_leaf" but it's actually the minimum
    # number of instances in the given axis, minimum lines or columns to have.

    # Variance of 1 element is 0, "great split you have".
    assert min_samples_leaf >= 2

    if Y_means.size < 2*min_samples_leaf:
        return None
    total_length = attr_col.size

    # Sort Ymeans by the attribute values.
    sorted_ids = attr_col.argsort()
    Y_means = Y_means[sorted_ids]

    # Cumulative squared euclidean distance to the mean.
    Y1_loss = cumdist1d(Y_means)
    Y2_loss = cumdist1d(Y_means[::-1])[::-1]

    q = 1 - (Y1_loss + Y2_loss) / total_length / Yvar

    # Remember min_samples_leaf >= 2 (first line).
    if min_samples_leaf != 2:
        msl = min_samples_leaf - 2
        q = q[msl:-msl]
    cut_id = np.argmax(q)
    quality = q[cut_id]

    if quality <= 0:
        return None  # FIXME: Is it OK to return different types?

    # +1 bellow because we have to include sorted_ids[cut_id] in ids1
    cut_id += min_samples_leaf
    ids1, ids2 = sorted_ids[:cut_id], sorted_ids[cut_id:]
    cutoff = (attr_col[ids1[-1]] + attr_col[ids2[0]]) / 2

    return q, quality, cutoff, ids1, ids2


@numba.njit(fastmath=True)
def _find_best_split_q(attr_col, Y_means, Y_sq_means, Yvar, min_samples_leaf):
    """PBCT.classes._find_best_split version."""
    total_length = attr_col.size
    highest_quality = 0
    qualities = []
    # The index right before the cutoff value in the sorted X.
    cut_id = -1

    # Sort Ymeans by the attribute values.
    sorted_ids = attr_col.argsort()
    Y_means = Y_means[sorted_ids]
    Y_sq_means = Y_sq_means[sorted_ids]

    # Split Y in Y1 and Y2.
    i = min_samples_leaf
    Y1_means_sum = Y_means[:i].sum()
    Y2_means_sum = Y_means[i:].sum()
    Y1_sq_means_sum = Y_sq_means[:i].sum()
    Y2_sq_means_sum = Y_sq_means[i:].sum()
    Y1size, Y2size = i, total_length-i

    for j in range(i, Y2size+1):  # Y2size happens to be what I needed.
        # sq_res is the sum of squared residuals (differences to the mean).
        Y1sq_res = Y1_sq_means_sum - Y1_means_sum ** 2 / Y1size
        Y2sq_res = Y2_sq_means_sum - Y2_means_sum ** 2 / Y2size
        quality = 1 - (Y1sq_res + Y2sq_res) / total_length / Yvar
        qualities.append(quality)

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

    #if cut_id == -1:
    if False:
        return None  # FIXME: Is it OK to return different types?
    else:
        ids1, ids2 = sorted_ids[:cut_id], sorted_ids[cut_id:]
        cutoff = (attr_col[ids1[-1]] + attr_col[ids2[0]]) / 2
        return np.array(qualities), highest_quality, cutoff, ids1, ids2


@numba.njit(fastmath=True)
def iter_cumdist_1d_z(xx):
    """Cumulative squared distance to the mean."""
    z = xx[0]
    m = z
    s = 0.

    for i in range(1, xx.shape[0]-2):
        x = xx[i]
        z += x
        m_old = m
        m = z / (i+1)
        s += (x-m_old) * (x-m)
        yield s


@numba.njit(fastmath=True)
def iter_cumdist_1d(xx):
    """Cumulative squared distance to the mean."""
    m = xx[0]
    s = 0.

    for i in range(1, xx.shape[0]-2):
        x = xx[i]
        last_xm = x-m
        m += last_xm / (i+1)
        s += last_xm * (x-m)
        yield s


def test_semisupervised_split():
    """Test SSPBCT's split function with Welford's online variance."""
    axis, attr = 1, 0
    np.random.seed(1)
    XX, Y = gen_imatrix((50, 20), (5, 5), quiet=True)
    X = XX[axis]
    Xcol = X[:, attr]
    Ymeans = Y.mean(axis=1)
    Ysqmeans = (Y**2).mean(axis=1)

    split = _find_best_split_ss(
        attr_col=Xcol,
        X=X,
        Xvar=X.var(0).mean(),
        Y_means=Ymeans,
        Ysq_means=Ysqmeans,
        Yvar=Y.var(),
        min_samples_leaf=1,
        supervision=1,
    )

    assert split is not None
    q, thresh, ids1, ids2 = split
    print(Y[ids1], Y[ids2])


def test_sq_sum_variance_split():
    """Compare original PBCT implementation with Welford's."""
    rng = np.random.default_rng()
    Xcol = rng.random(1000)
    Y = (Xcol > .1).astype(np.float64)
    Y = (Y + rng.random(1000)) / 2
    Yvar = Y.var()
    Ysq = Y**2

    # Run once to precompile.
    _find_best_split(Xcol, Y, Ysq, Yvar, min_samples_leaf=4)
    _find_best_split_welford(Xcol, Y, Yvar, min_samples_leaf=4)

    t1 = repeat('_find_best_split(Xcol, Y, Ysq, Yvar, min_samples_leaf=4)',
                globals=locals()|globals(), number=1000, repeat=10)
    t2 = repeat('_find_best_split_welford(Xcol, Y, Yvar, min_samples_leaf=4)',
                globals=locals()|globals(), number=1000, repeat=10)
    print('Original:', np.mean(t1), '+/-', np.std(t1))
    print('Welford: ', np.mean(t2), '+/-', np.std(t2))
    

def compare_qualities():
    np.random.seed(1)
    XX, Y = gen_imatrix((50, 20), (5, 5), quiet=True)
    Xcol = XX[1][:, 0]
    Ymeans = Y.mean(axis=1)
    Ysqmeans = (Y**2).mean(axis=1)

    # n = 6
    # rng = np.random.default_rng()
    # Xcol = rng.random(n)
    # Y = (Xcol > (.1 * rng.random(n))).astype(np.float64)
    # Y = (Y + rng.random(n)) / 2
    # Ymeans = Y

    Yvar = Y.var()
    if Yvar == 0:
        print('Yvar == 0, try another seed.')
        exit()

    splits = (
    )

    q = [np.array(s[0]) for s in splits]

    for i in zip(*splits):
        print(*i, sep='\n')
        print()
    print(q[1]/q[0])
      

def compare_splits():
    axis, attr = 1, 0
    np.random.seed(3)
    XX, Y = gen_imatrix((50, 2), (5, 5), quiet=True)
    X = XX[axis]
    Xcol = X[:, attr]
    Ymeans = Y.mean(axis=1)
    Ysqmeans = (Y**2).mean(axis=1)

    splits = [
        _find_best_split_ss(
            attr_col=Xcol,
            X=X,
            Xvar=X.var(0).mean(),
            Y_means=Ymeans,
            Ysq_means=Ysqmeans,
            Yvar=Y.var(),
            min_samples_leaf=1,
            supervision=1,
        ),
        _find_best_split_ec(
            attr_col=Xcol,
            Y_means=Ymeans,
            Ysq_means=Ysqmeans,
            Yvar=Y.var(),
            min_samples_leaf=1,
        ),
        _find_best_split(
            attr_col=Xcol,
            Y_means=Ymeans,
            Y_sq_means=Ysqmeans,
            Yvar=Y.var(),
            min_samples_leaf=1,
        )
    ]

    assert None not in splits

    q = (np.array(s[0]) for q in splits)
    for i in zip(*splits):
        print(*i, sep='\n')


def main():
    #test_sq_sum_variance_split()
    #test_semisupervised_split()
    #compare_qualities()
    compare_splits()


if __name__ == '__main__':
    main()
