import numpy as np

def linear_sum_assignment(costs, solver='scipy'):
    solvers = {
        'scipy' : lambda costs: lsa_solve_scipy(costs),
    }
    return solvers[solver.lower()](costs)

def lsa_solve_scipy(costs):
    from scipy.optimize import linear_sum_assignment as scipy_solve
    return scipy_solve(safe_costs(costs))

def safe_costs(costs):
    """Replace infinite/NaN distances by safe finite values."""
    costs = np.copy(costs)
    
    # Note there is an issue in scipy.optimize.linear_sum_assignment where
    # it runs forever if an entire row/column is infinite or nan. We therefore
    # make a copy of the distance matrix and compute a safe value that indicates
    # 'cannot assign'. Also note + 1 is necessary in below inv-dist computation
    # to make invdist bigger than max dist in case max dist is zero.
    
    valid = costs[np.isfinite(costs)]
    INVDIST = 2 * valid.max() + 1 if valid.shape[0] > 0 else 1.
    costs[~np.isfinite(costs)] = INVDIST  

    return costs