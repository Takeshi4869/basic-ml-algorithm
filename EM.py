import numpy as np
from scipy import stats


def em(observations, thetas, tol=1e-4, max_iter=100):
    thetas_old = thetas.copy()
    for _ in range(max_iter):
        # E step
        at = bt = af = bf = 0
        for ob in observations:
            contribution_a = stats.binom.pmf(ob.sum(), len(ob), thetas[0])
            contribution_b = stats.binom.pmf(ob.sum(), len(ob), thetas[1])
            w_a = contribution_a / (contribution_a + contribution_b)
            w_b = contribution_b / (contribution_a + contribution_b)
            at += w_a * ob.sum()
            af += w_a * (len(ob) - ob.sum())
            bt += w_b * ob.sum()
            bf += w_b * (len(ob) - ob.sum())

        # M step
        thetas_old[0] = thetas[0]
        thetas_old[1] = thetas[1]
        thetas[0] = at / (at + af)
        thetas[1] = bt / (bt + bf)
        if np.sum(abs(thetas-thetas_old)) < tol:
            break

    return thetas

