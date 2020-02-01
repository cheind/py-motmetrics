"""Tools for solving linear assignment problems."""

# pylint: disable=import-outside-toplevel

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import warnings

import numpy as np


def _module_is_available_py2(name):
    try:
        imp.find_module(name)
        return True
    except ImportError:
        return False


def _module_is_available_py3(name):
    return importlib.util.find_spec(name) is not None


try:
    import importlib.util
except ImportError:
    import imp
    _module_is_available = _module_is_available_py2
else:
    _module_is_available = _module_is_available_py3


def linear_sum_assignment(costs, solver=None):
    """Solve a linear sum assignment problem (LSA).

    For large datasets solving the minimum cost assignment becomes the dominant runtime part.
    We therefore support various solvers out of the box (currently lapsolver, scipy, ortools, munkres)

    Params
    ------
    costs : np.array
        numpy matrix containing costs. Use NaN/Inf values for unassignable
        row/column pairs.

    Kwargs
    ------
    solver : callable or str, optional
        When str: name of solver to use.
        When callable: function to invoke
        When None: uses first available solver
    """
    costs = np.asarray(costs)
    if not costs.size:
        return np.empty([0], dtype=int), np.empty([0], dtype=int)

    solver = solver or default_solver

    if isinstance(solver, str):
        # Try resolve from string
        solver = solver_map.get(solver, None)

    assert callable(solver), 'Invalid LAP solver.'
    rids, cids = solver(costs)
    rids = np.asarray(rids).astype(int)
    cids = np.asarray(cids).astype(int)
    return rids, cids


def add_expensive_edges(costs):
    """Replaces non-edge costs (nan, inf) with large number.

    If the optimal solution includes one of these edges,
    then the original problem was infeasible.

    Kwargs
    ------
    costs : np.ndarray
    """
    # The graph is probably already dense if we are doing this.
    assert isinstance(costs, np.ndarray)
    # The linear_sum_assignment function in scipy does not support missing edges.
    # Replace nan with a large constant that ensures it is not chosen.
    # If it is chosen, that means the problem was infeasible.
    valid = np.isfinite(costs)
    if valid.all():
        return costs.copy()
    if not valid.any():
        return np.zeros_like(costs)
    r = min(costs.shape)
    # Assume all edges costs are within [-c, c], c >= 0.
    # The cost of an invalid edge must be such that...
    # choosing this edge once and the best-possible edge (r - 1) times
    # is worse than choosing the worst-possible edge r times.
    # l + (r - 1) (-c) > r c
    # l > r c + (r - 1) c
    # l > (2 r - 1) c
    # Choose l = 2 r c + 1 > (2 r - 1) c.
    c = np.abs(costs[valid]).max() + 1  # Doesn't hurt to add 1 here.
    large_constant = 2 * r * c + 1
    return np.where(valid, costs, large_constant)


def _assert_solution_is_feasible(costs, rids, cids):
    ijs = list(zip(rids, cids))
    if len(ijs) != min(costs.shape):
        raise AssertionError('infeasible solution: not enough edges')
    elems = [costs[i, j] for i, j in ijs]
    if not np.all(np.isfinite(elems)):
        raise AssertionError('infeasible solution: includes non-finite edges')


def lsa_solve_scipy(costs):
    """Solves the LSA problem using the scipy library."""

    from scipy.optimize import linear_sum_assignment as scipy_solve

    finite_costs = add_expensive_edges(costs)
    rids, cids = scipy_solve(finite_costs)
    _assert_solution_is_feasible(costs, rids, cids)
    return rids, cids


def lsa_solve_lapsolver(costs):
    """Solves the LSA problem using the lapsolver library."""
    from lapsolver import solve_dense

    # Note that lapsolver will add expensive finite edges internally.
    # However, older versions did not add a large enough edge.
    finite_costs = add_expensive_edges(costs)
    rids, cids = solve_dense(finite_costs)
    _assert_solution_is_feasible(costs, rids, cids)
    return rids, cids


def lsa_solve_munkres(costs):
    """Solves the LSA problem using the Munkres library."""
    from munkres import Munkres

    num_rows, num_cols = costs.shape
    m = Munkres()
    # The munkres package may hang if the problem is not feasible.
    # Therefore, add expensive edges instead of using munkres.DISALLOWED.
    finite_costs = add_expensive_edges(costs)
    # Ensure that matrix is square.
    finite_costs = _zero_pad_to_square(finite_costs)
    indices = np.array(m.compute(finite_costs), dtype=np.int)
    indices = indices[(indices[:, 0] < num_rows)
                      & (indices[:, 1] < num_cols)]
    rids, cids = indices[:, 0], indices[:, 1]
    _assert_solution_is_feasible(costs, rids, cids)
    return rids, cids


def _zero_pad_to_square(costs):
    num_rows, num_cols = costs.shape
    if num_rows == num_cols:
        return costs
    n = max(num_rows, num_cols)
    padded = np.zeros((n, n), dtype=costs.dtype)
    padded[:num_rows, :num_cols] = costs
    return padded


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
            if valid[r, c]:
                assignment.AddArcWithCost(r, c, int(costs[r, c] * f))

    if assignment.Solve() != assignment.OPTIMAL:
        return linear_sum_assignment(costs, solver='scipy')

    if assignment.NumNodes() == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    pairings = []
    for i in range(assignment.NumNodes()):
        pairings.append([i, assignment.RightMate(i)])

    indices = np.array(pairings, dtype=np.int64)
    return indices[:, 0], indices[:, 1]


def lsa_solve_lapjv(costs):
    """Solves the LSA problem using lap.lapjv()."""

    from lap import lapjv

    # The lap.lapjv function supports +inf edges but there are some issues.
    # https://github.com/gatagat/lap/issues/20
    # Therefore, replace nans with large finite cost.
    finite_costs = add_expensive_edges(costs)
    row_to_col, _ = lapjv(finite_costs, return_cost=False, extend_cost=True)
    indices = np.array([np.arange(costs.shape[0]), row_to_col], dtype=np.int).T
    # Exclude unmatched rows (in case of unbalanced problem).
    indices = indices[indices[:, 1] != -1]  # pylint: disable=unsubscriptable-object
    rids, cids = indices[:, 0], indices[:, 1]
    # Ensure that no missing edges were chosen.
    _assert_solution_is_feasible(costs, rids, cids)
    return rids, cids


available_solvers = None
default_solver = None
solver_map = None


def _init_standard_solvers():
    global available_solvers, default_solver, solver_map  # pylint: disable=global-statement

    solvers = [
        ('lapsolver', lsa_solve_lapsolver),
        ('lap', lsa_solve_lapjv),
        ('scipy', lsa_solve_scipy),
        ('munkres', lsa_solve_munkres),
        ('ortools', lsa_solve_ortools),
    ]

    solver_map = dict(solvers)

    available_solvers = [s[0] for s in solvers if _module_is_available(s[0])]
    if len(available_solvers) == 0:
        default_solver = None
        warnings.warn('No standard LAP solvers found. Consider `pip install lapsolver` or `pip install scipy`', category=RuntimeWarning)
    else:
        default_solver = available_solvers[0]


_init_standard_solvers()


@contextmanager
def set_default_solver(newsolver):
    """Change the default solver within context.

    Intended usage

        costs = ...
        mysolver = lambda x: ... # solver code that returns pairings

        with lap.set_default_solver(mysolver):
            rids, cids = lap.linear_sum_assignment(costs)

    Params
    ------
    newsolver : callable or str
        new solver function
    """

    global default_solver  # pylint: disable=global-statement

    oldsolver = default_solver
    try:
        default_solver = newsolver
        yield
    finally:
        default_solver = oldsolver
