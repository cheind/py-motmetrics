import numpy as np
import pandas as pd
import sys
from motmetrics.hungarian import linear_sum_assignment as lsa, available_solvers


def benchmark(sizes, exclude_above):
    import time

    index = []
    data = []

    for idx, s in enumerate(sizes):
        costs = np.random.uniform(low=-1000, high=1000, size=s)
        ass = []
        for solver in available_solvers:            
             # warm-up                

            if s[0] <= exclude_above.get(solver, sys.maxsize) and s[1] <= exclude_above.get(solver, sys.maxsize):
                if idx == 0:
                    lsa(costs, solver=solver)

                t0 = time.time()
                ass.append(lsa(costs, solver=solver))
                t1 = time.time()
                runtime = t1-t0

                index.append((f'{s[0]}x{s[1]}', solver))
                data.append([f'{runtime:.3f}'])

                print(f'{solver} on {s} took {runtime:.3f} seconds')
            else:                
                print(f'Skipping {solver} on {s}')
                index.append((f'{s[0]}x{s[1]}', solver))
                data.append(['-'])
        
        for i in range(1, len(ass)):
            print('comparing',0,i,costs[ass[0][0], ass[0][1]].sum(), costs[ass[i][0], ass[i][1]].sum())
            np.testing.assert_allclose(ass[0], ass[i])
    
    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index, names=['Matrix', 'Solver']), columns=['Runtime [sec]'])
    return df

def run():
    import time

    square_sizes = [(3,3), (10,10), (100,100), (200,200), (500,500), (1000,1000), (5000,5000), (10000,10000)]
    nonsquare_sizes = [(3,2), (10,5), (100,10), (200,20), (500,50), (1000,100), (5000,500), (10000,1000)]
    
    # Some solvers are too slow for larger problem sizes
    exclude_above = {
        'scipy': 500,
        'ortools': 5000,
        'munkres': 200
    }    
    
    print(benchmark(square_sizes, exclude_above))

    exclude_above = {
        'munkres': 0, #does not work for non-square matrices
        'scipy': 500,
        'ortools': 5000
    }    


    print(benchmark(nonsquare_sizes, exclude_above))


if __name__ == '__main__':
    run()