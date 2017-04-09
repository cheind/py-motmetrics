
import motmetrics as mm
import numpy as np

if __name__ == '__main__':

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Each frame a list of ground truth object / hypotheses ids and pairwise distances
    # is passed to the accumulator. For now assume that the distance matrix given to us.

    # 2 Matches, 1 False alarm
    acc.update(
        ['a', 'b'],                 # Ground truth objects in this frame
        [1, 2, 3],                  # Detector hypotheses in this frame
        [[0.1, np.nan, 0.3],        # Distances from object 'a' to hypotheses 1, 2, 3
         [0.5,  0.2,   0.3]]        # Distances from object 'b' to hypotheses 1, 2, 
    )

    # 1 Match, 1 Miss
    acc.update(
        ['a', 'b'],
        [1],
        [[0.2], [0.4]]
    )

    # 1 Match, 1 Switch
    acc.update(
        ['a', 'b'],
        [1, 3],
        [[0.6, 0.2],
         [0.1, 0.6]]
    )
    
    #print(acc.events)

    print('Summary')
    x = acc.events.loc[0:1]
    summary = mm.metrics.summarize(x)
    #print(mm.io.render_summary(summary))

    

    
