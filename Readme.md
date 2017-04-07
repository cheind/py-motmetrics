
## py-clearmot - CLEAR MOT metrics for multiple object tracker evaluation

This library provides a Python implementation of CLEAR MOT metrics for evaluation object tracker performances based on 

> Bernardin, Keni, and Rainer Stiefelhagen.
> "Evaluating multiple object tracking performance: the CLEAR MOT metrics." 

Main features are
- Distance agnostic. Supports Euclidean, Intersection over Union and other distances measures.
- Complete event history. Tracks all relevant per-frame events suchs as correspondences, misses, false alarms and switches. 
- Uses Python [pandas][1] for data structures and analysis.
- Supports MOTA and MOTP metrics.
- Global minimum cost assignments are accomplished through Kuhn-Munkres algorithm.

## Usage

```python
import clearmot as cm

# TODO
```

### Installation
To install **py-clearmot** clone this repository and use `pip` to install
from local sources.

```
pip install -e <path/to/setup.py>
```

Python 3.5/3.6 and numpy, pandas and scipy is required.

### Continuous Integration

Branch  | Status
------- | ------
master  | ![](https://travis-ci.org/cheind/py-clearmot.svg?branch=master)
develop | ![](https://travis-ci.org/cheind/py-clearmot.svg?branch=develop)

### License
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

[1]: http://pandas.pydata.org/
