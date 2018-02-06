import numpy as np
from motmetrics.hungarian import linear_sum_assignment as lsa, available_solvers

def run():
    import time

    sizes = [(3,3), (10,10), (100,100), (200,200), (500,500), (1000,1000)]

    for s in sizes:
        costs = np.random.uniform(low=-1000, high=1000, size=s)
        ass = []
        for solver in available_solvers:
            t0 = time.time()
            ass.append(lsa(costs, solver=solver))
            t1 = time.time()
            print(f'{solver} on {s} took {t1-t0} seconds')
        np.testing.assert_allclose(ass[0], ass[1])

if __name__ == '__main__':
    run()