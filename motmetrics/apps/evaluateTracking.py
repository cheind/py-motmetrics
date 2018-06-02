"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking with RESULT PREPROCESS.

TOKA, 2018
ORIGIN: https://github.com/cheind/py-motmetrics
EXTENDED: <reposity>
"""

import argparse
import glob
import os
import logging
import motmetrics as mm
import pandas as pd
from collections import OrderedDict
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using MOTChallenge ground-truth data with data preprocess.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in 

Milan, Anton, et al. 
"Mot16: A benchmark for multi-object tracking." 
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...

Seqmap for test data
    [name]
    <SEQUENCE_1>
    <SEQUENCE_2>
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string in the seqmap.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('groundtruths', type=str, help='Directory containing ground truth files.')   
    parser.add_argument('tests', type=str, help='Directory containing tracker result files')
    parser.add_argument('seqmap', type=str, help='Text file containing all sequences name')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
    return parser.parse_args()

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logging.info('Evaluating {}...'.format(k))
            accs.append(mm.utils.CLEAR_MOT_M(gts[k][0], tsacc, gts[k][1], 'iou', distth=0.5))
            names.append(k)
        else:
            logging.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

def parseSequences(seqmap):
    assert os.path.isfile(seqmap), 'Seqmap %s not found.'%seqmap
    fd = open(seqmap)
    res = []
    for row in fd.readlines():
        row = row.strip()
        if row=='' or row=='name' or row[0]=='#': continue
        res.append(row)
    fd.close()
    return res

if __name__ == '__main__':

    args = parse_args()

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))        
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    seqs = parseSequences(args.seqmap)
    gtfiles = [os.path.join(args.groundtruths, i, 'gt/gt.txt') for i in seqs]
    tsfiles = [os.path.join(args.tests, '%s.txt'%i) for i in seqs]

    for gtfile in gtfiles:
        if not os.path.isfile(gtfile):
            logging.error('gt File %s not found.'%gtfile)
            exit(1)
    for tsfile in tsfiles:
        if not os.path.isfile(tsfile):
            logging.error('res File %s not found.'%tsfile)
            exit(1)

    logging.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logging.info('Loading files.')
    
    gt = OrderedDict([(seqs[i], (mm.io.loadtxt(f, fmt=args.fmt), os.path.join(args.groundtruths, seqs[i], 'seqinfo.ini')) ) for i, f in enumerate(gtfiles)])
    ts = OrderedDict([(seqs[i], mm.io.loadtxt(f, fmt=args.fmt)) for i, f in enumerate(tsfiles)])    

    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    
    logging.info('Running metrics')
    
    summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')
