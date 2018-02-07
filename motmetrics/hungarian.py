import numpy as np

def linear_sum_assignment(costs, solver=None):
    """Solve a linear sum assignment problem (LSA).

    For large datasets solving the minimum cost assignment becomes the dominant runtime part. 
    We therefore support various solvers out of the box (currently scipy, ortools, munkres)
    
    Params
    ------
    costs : np.array
        numpy matrix containing costs. Use NaN/Inf values for unassignable
        row/column pairs.
    
    Kwargs
    ------
    solver : str, optional
        Name of solver to use. If None uses the first available solver
        for smaller problem sizes and if available 'ortools' solver
        for larger problem sizes.
    """

    if solver is None:
        solver = default_solver
        #if costs.size > 10000 and 'ortools' in available_solvers:
        #    solver = 'ortools'

    solvers = {
        'fast_hungarian' : lambda costs: lsa_solve_fast_hungarian(costs),
        'scipy' : lambda costs: lsa_solve_scipy(costs),
        'munkres' : lambda costs: lsa_solve_munkres(costs),
        'ortools' : lambda costs: lsa_solve_ortools(costs)
    }
    return solvers[solver.lower()](costs)

def lsa_solve_scipy(costs):
    """Solves the LSA problem using the scipy library."""

    from scipy.optimize import linear_sum_assignment as scipy_solve
   
    # Note there is an issue in scipy.optimize.linear_sum_assignment where
    # it runs forever if an entire row/column is infinite or nan. We therefore
    # make a copy of the distance matrix and compute a safe value that indicates
    # 'cannot assign'. Also note + 1 is necessary in below inv-dist computation
    # to make invdist bigger than max dist in case max dist is zero.
    
    inv = ~np.isfinite(costs)
    if inv.any():
        costs = costs.copy()
        valid = costs[~inv]
        INVDIST = 2 * valid.max() + 1 if valid.shape[0] > 0 else 1.
        costs[inv] = INVDIST

    return scipy_solve(costs)

def lsa_solve_fast_hungarian(costs):
    from fast_hungarian import solve_minimum_cost
    return solve_minimum_cost(costs)

def lsa_solve_munkres(costs):
    """Solves the LSA problem using the Munkres library."""
    from munkres import Munkres, DISALLOWED
    m = Munkres()

    costs = costs.copy()
    inv = ~np.isfinite(costs)
    if inv.any():
        costs = costs.astype(object)
        costs[inv] = DISALLOWED       

    indices = np.array(m.compute(costs), dtype=np.int64)
    return indices[:,0], indices[:,1]


def lsa_solve_ortools(costs):
    """Solves the LSA problem using Google's optimization tools."""
    from ortools.graph import pywrapgraph


    # Google OR tools only support integer costs. Here's our attempt
    # to convert from floating point to integer:
    #
    # We search for the minimum difference between any two costs and
    # compute the first non-zero digit after the decimal place. Then
    # we compute a factor,f, that scales all costs so that the difference
    # is integer representable in the first digit.
    # 
    # Example: min-diff is 0.001, then first non-zero digit place -3, so
    # we scale by 1e3.
    #
    # For small min-diffs and large costs in general there is a change of
    # overflowing.

    valid = np.isfinite(costs)

    min_e = -8
    unique = np.unique(costs[valid])

    if unique.shape[0] == 1:
        min_diff = unique[0]
    elif unique.shape[0] > 1:
        min_diff = np.diff(unique).min()
    else:
        min_diff = 1

    min_diff_e = 0
    if min_diff != 0.0:
        min_diff_e = int(np.log10(np.abs(min_diff)))
        if min_diff_e < 0:
            min_diff_e -= 1
    
    e = min(max(min_e, min_diff_e), 0)
    f = 10**abs(e)

    assignment = pywrapgraph.LinearSumAssignment()
    for r in range(costs.shape[0]):
        for c in range(costs.shape[1]):
            if valid[r,c]:
                assignment.AddArcWithCost(r, c, int(costs[r,c]*f))
    
    if assignment.Solve() != assignment.OPTIMAL:
        assert default_solver != 'ortools'
        return linear_sum_assignment(costs, solver=default_solver)

    if assignment.NumNodes() == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    pairings = []
    for i in range(assignment.NumNodes()):        
        pairings.append([i, assignment.RightMate(i)])

    indices = np.array(pairings, dtype=np.int64)
    return indices[:,0], indices[:,1]


import importlib
available_solvers = [s for s in ['fast_hungarian', 'scipy', 'ortools', 'munkres'] if importlib.util.find_spec(s) is not None]
assert len(available_solvers) > 0
default_solver = available_solvers[0]