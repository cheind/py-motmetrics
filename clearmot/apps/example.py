
import clearmot as cm
import numpy as np

if __name__ == '__main__':

    # Create an accumulator that will be updated during each frame
    acc = cm.new_accumulator(auto_id=True)

    # Each frame a list of object / hypotheses ids and pairwise distances
    # is passed to the accumulator. Assume that distance matrix given for 
    # now to us.

    cm.update_mot(acc,
        ['a', 'b'],                 # Object ids in this frame reported by ground truth
        [1, 2, 3],                  # Hypothesis ids in this frame reported by detector
        [[0.1, np.nan, 0.3],        # Distances from object 'a' to hypotheses 1, 2, 3
         [0.5, 0.2, 0.3]]           # Distances from object 'b' to hypotheses 1, 2, 
    )
    print(acc.events)

    cm.update_mot(acc,
        ['a', 'b'],
        [1],
        [[0.2], [0.4]]
    )
    print(acc.events)

    cm.update_mot(acc,
        ['a', 'b'],
        [1, 3],
        [[0.6, 0.2],
         [0.1, 0.6]]
    )
    print(acc.events)

    print(cm.compute_stats(acc))

    
