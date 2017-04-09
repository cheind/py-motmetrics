## py-motmetrics

The **py-motmetrics** library provides a Python implementation of metrics for benchmarking multiple object trackers (MOT).

While benchmarking single object trackers is rather straightforward, measuring the performance of multiple object trackers needs careful design as multiple correspondence constellations can arise (see image below). A variety of methods have been proposed in the past and while there is no general agreement on a single method, the methods of [[1,2,3]](#References) have received considerable attention in recent years. **py-motmetrics** implements these [metrics](#Metrics).

<div style="text-align:center;">

![](etc/mot.png)<br/>
*Pictures courtesy of Bernardin, Keni, and Rainer Stiefelhagen [[1]](#References)*
</div>

### Features at a glance
- *Variety of metrics* <br/>
Provides MOTA, MOTP, track quality measures and more. The results are [comparable](#MOTChallengeCompatibility) with the popular [MOTChallenge][MOTChallenge] benchmarks.
- *Distance agnostic* <br/>
Supports Euclidean, Intersection over Union and other distances measures.
- *Complete event history* <br/> 
Tracks all relevant per-frame events suchs as correspondences, misses, false alarms and switches.
- *Easy to extend* <br/> 
Events and summaries are utilizing [pandas][pandas] for data structures and analysis.

<a name="Metrics"></a>
### Metrics

**py-motmetrics** implements the following metrics. The metrics have been aligned with what is reported by [MOTChallenge][MOTChallenge] benchmarks.

Metric  | Unit   | Description |
------- | ------ | ----------- |
Frames  | Count  | Total number of frames|
Match  | Count  | Total number matches|
Switch  | Count  | Total number track switches (see [[1]](#References))|
FalsePos  | Count  | Total number false positive hypothesis (see [[1]](#References))|
Miss  | Count  | Total number missed objects (see [[1]](#References))|
MOTP  | Distance  | Multiple Object Tracking Precision. Average distance error of correctly detected objects (see [[1]](#References)). Note, MOTChallenge compatibility is given by `MOTP=1-MOTP`, distance IoU with threshold 0.5|
MOTA  | Percentage  | Multiple object tracking accuracy (see [[1]](#References)). Accounts for object configuration errors of tracker.|
|Precision | Percentage | Percent of correct detections to total tracker detections (see [[2]](#References)).|
|Recall | Percentage | Percent of correct detections to total number of objects (see [[2]](#References)).|
|Frag | Count | Total number of track fragmentations (see [[2,3]](#References)). |
|Objs | Count | Total number unique objects. |
|MT | Count | Mostly tracked objects (see [[2,3]](#References)). Count of trajectories covered by hypothesis for at least 80% of track-lifespan.|
|PT | Count | Partially tracked objects (see [[2,3]](#References)). Count of trajectories covered by hypothesis between 20% and 80% of track-lifespan. |
|ML | Count | Mostly lost objects (see [[2,3]](#References)). Count of trajectories covered by hypothesis for less than 20% of track-lifespan.|

### Usage

#### Populating the accumulator

```python
import motmetrics as mm

# Create an accumulator that will be updated during each frame
acc = mm.MOTAccumulator(auto_id=True)

# Call update once for per frame. For now, assume distances between
# frame objects / hypotheses are given.
acc.update(
    ['a', 'b'],                 # Ground truth objects in this frame
    [1, 2, 3],                  # Detector hypotheses in this frame
    [
        [0.1, np.nan, 0.3],     # Distances from object 'a' to hypotheses 1, 2, 3
        [0.5,  0.2,   0.3]      # Distances from object 'b' to hypotheses 1, 2, 3
    ]
)
```

The code above updates an event accumulator with data from a single frame. Here we assume that pairwise object / hypothesis distance have already been computed. Note the `np.nan` inside the distance matrix. It signals that `a` cannot be paired with hypothesis `2`. To inspect the current event state simple print the events associated with the accumulator

```python
print(acc.events) # a pandas DataFrame
"""
                Type  OId HId    D
FrameId Event
0       0      MATCH    a   1  0.1
        1      MATCH    b   2  0.2
        2         FP  NaN   3  NaN
"""
```

Meaning object `a` was matched to hypothesis `1` with distance 0.1. Similarily, `b` was matched to `2` with distance 0.2. Hypothesis `3` could not be matched to any remaining object and generated a false positive (FP).

Continuing from above
```python
df = acc.update(
    ['a', 'b'],
    [1],
    [
        [0.2], 
        [0.4]
    ]
)
print(df)
"""
0      MATCH   a    1  0.2
1       MISS   b  NaN  NaN
"""
```

While `a` was matched, `b` couldn't be matched because of lacking hypotheses.

```python
df = acc.update(
    ['a', 'b'],
    [1, 3],
    [
        [0.6, 0.2],
        [0.1, 0.6]
    ]
)
print(df)
"""
0       MATCH   a   1  0.6
1      SWITCH   b   3  0.6
"""
```
`b` is now tracked by hypothesis `3` leading to a track switch.

#### Computing metrics
Once the accumulator has been populated you can compute and display metrics. Continuing the example from above

```python
summary = mm.metrics.summarize(acc)
print(mm.io.render_summary(summary))
"""
   Frames  Match  Switch  FalsePos  Miss  MOTP   MOTA Precision Recall  Frag  Objs  MT  PT  ML
0       3      4       1         1     1 0.340 50.00%    83.33% 83.33%     1     2   1   1   0
"""

# Summarize multiple accumulators or accumulator parts
summaries = mm.metrics.summarize([acc, acc.events.loc[0:1]], names=['full', 'part'])
print(mm.io.render_summary(summaries))
"""
      Frames  Match  Switch  FalsePos  Miss  MOTP   MOTA Precision Recall  Frag  Objs  MT  PT  ML
full       3      4       1         1     1 0.340 50.00%    83.33% 83.33%     1     2   1   1   0
part       2      3       0         1     1 0.167 50.00%    75.00% 75.00%     0     2   1   1   0
"""
```

#### Computing distances
Up until this point we assumed the pairwise object/hypothesis distances to be known. Usually this is not the case. You are mostly given either rectangles or points (centroids) of related
objects. To compute a distance matrix from them you can use `motmetrics.distance` module as shown below.

##### Euclidean norm squared on points

```python
# Object related points
o = np.array([
    [1., 2],
    [2., 2],
    [3., 2],
])

# Hypothesis related points
h = np.array([
    [0., 0],
    [1., 1],      
])

C = mm.distances.norm2squared_matrix(o, h, max_d2=5.)
"""
[[  5.   1.]
 [ nan   2.]
 [ nan   5.]]
"""
```

##### Intersection over union norm for 2D rectangles
```python
a = np.array([
    [0, 0, 20, 100],    # Format X, Y, Width, Height
    [0, 0, 0.8, 1.5],
])

b = np.array([
    [0, 0, 1, 2],
    [0, 0, 1, 1],
    [0.1, 0.2, 2, 2],
])
mm.distances.iou_matrix(a, b, max_iou=0.5)
"""
[[ 0.          0.5                nan]
 [ 0.4         0.42857143         nan]]
"""
```

<a name="MOTChallengeCompatibility"></a>
### MOTChallenge compatibility

**py-motmetrics** produces results compatible with popular [MOTChallenge][MOTChallenge] benchmarks. Below are two results taken from MOTChallenge [Matlab devkit][devkit] corresponding to the results of the CEM tracker on the training set of the 2015 MOT 2DMark.

```
        ... TUD-Campus
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL
 58.2  94.1  0.18|  8   1   6   1|   13   150    7    7|  52.6  72.3  54.3

         ... TUD-Stadtmitte
 Rcll  Prcn   FAR| GT  MT  PT  ML|   FP    FN  IDs   FM|  MOTA  MOTP MOTAL
 60.9  94.0  0.25| 10   5   4   1|   45   452    7    6|  56.4  65.4  56.9
```

In comparison to **py-motmetrics**

```
                Frames  Match  Switch  FalsePos  Miss  MOTP   MOTA Precision Recall  Frag  Objs  MT  PT  ML
TUD-Campus          71    202       7        13   150 0.277 52.65%    94.14% 58.22%     7     8   1   6   1
TUD-Stadtmitte     179    697       7        45   452 0.346 56.40%    93.99% 60.90%     6    10   5   4   1
```

Besides naming differences, the only obvious differences are
- Metric `FAR` is missing. This metric is given implicitly and can be recovered by `FalsePos / Frames * 100`.
- Metric `MOTP` seems to be off. To convert compute `(1 - MOTP) * 100`. [MOTChallenge][MOTChallenge] benchmarks compute `MOTP` as percentage, while **py-motmetrics** sticks to the original average distance of assigned objects definition [[1]](#References).

All results are asserted by unit tests.

### Running tests
**py-motmetrics** uses the pytest framework. To run the tests, simply `cd` into the source directy and run `pytest`.

### Installation
To install **py-motmetrics** clone this repository and use `pip` to install
from local sources.

```
pip install -e <path/to/setup.py>
```

Python 3.5/3.6 and numpy, pandas and scipy is required.

### Continuous Integration

Branch  | Status
------- | ------
master  | ![](https://travis-ci.org/cheind/py-motmetrics.svg?branch=master)


<a name="References"></a>
### References
1. Bernardin, Keni, and Rainer Stiefelhagen. "Evaluating multiple object tracking performance: the CLEAR MOT metrics." 
EURASIP Journal on Image and Video Processing 2008.1 (2008): 1-10.
2. Milan, Anton, et al. "Mot16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016).
3. Li, Yuan, Chang Huang, and Ram Nevatia. "Learning to associate: Hybridboosted multi-target tracker for crowded scene." 
Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on. IEEE, 2009.


### License

```
MIT License

Copyright (c) 2017 Christoph Heindl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


[Pandas]: http://pandas.pydata.org/
[MOTChallenge]: https://motchallenge.net/
[devkit]: https://motchallenge.net/devkit/
