"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
"""

import argparse
import glob
import os
import logging
import motmetrics as mm
import pandas as pd
from collections import OrderedDict
from pathlib import Path
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="""
Compute MOT metrics.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in 

Milan, Anton, et al. 
"Mot16: A benchmark for multi-object tracking." 
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Usage
---------
python -m motmetrics.apps.eval_my_dataset --groundtruth <path_to_gt_file> --test <path_to_test_file>
""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--groundtruth', type=str, help='Directory containing ground truth files.')   
    parser.add_argument('--test', type=str, help='Directory containing tracker result files')
    parser.add_argument('--iou', type=str, help='IoU', default=0.5)
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
    return parser.parse_args()

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for test_key, test_value in ts.items():
        for gt_key, gt_value in gts.items():
            accs.append(mm.utils.compare_to_groundtruth(gt_value, test_value, 'iou', distth=0.5))
            names.append(test_key)
    return accs, names

if __name__ == '__main__':
    args = parse_args()

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))        
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    gtfiles = [str(args.groundtruth)]
    tsfiles = [str(args.test)]

    logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logging.info('Loading files.')
    
    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt=args.fmt, min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=args.fmt)) for f in tsfiles])   

    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    logging.info('Running metrics')
    
    summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')