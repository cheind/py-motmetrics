import numpy as np
import pandas as pd
import sys
from motmetrics.hungarian import linear_sum_assignment as lsa, available_solvers

def run():
    import time

    sizes = [(3,3), (10,10), (100,100), (200,200), (500,500), (1000,1000), (5000,5000)]
    exclude_above = {
        'munkres': 200,     
        'scipy': 500
    }
    

    index = []
    data = []

    for idx, s in enumerate(sizes):
        costs = np.random.uniform(low=-1000, high=1000, size=s)
        ass = []
        for solver in available_solvers:            
            if idx == 0:
                lsa(costs, solver=solver) # warm-up                

            if s[0] <= exclude_above.get(solver, sys.maxsize):
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
            np.testing.assert_allclose(ass[0], ass[i])
    
    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index, names=['Matrix', 'Solver']), columns=['Runtime [sec]'])
    print(df)

if __name__ == '__main__':
    run()