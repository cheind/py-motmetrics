# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Functions for loading data and writing summaries."""

import io
from enum import Enum
from pathlib import Path

import numpy as np


class Format(Enum):
    """Enumerates supported file formats."""

    AUTO = 'auto'
    """Infer the format from the file extension and a small data sample."""

    MOT16 = 'mot16'
    """Milan, Anton, et al. "Mot16: A benchmark for multi-object tracking." arXiv preprint arXiv:1603.00831 (2016)."""

    MOT15_2D = 'mot15-2D'
    """Leal-Taixe, Laura, et al. "MOTChallenge 2015: Towards a benchmark for multi-target tracking." arXiv preprint arXiv:1504.01942 (2015)."""

    VATIC_TXT = 'vatic-txt'
    """Vondrick, Carl, Donald Patterson, and Deva Ramanan. "Efficiently scaling up crowdsourced video annotation." International Journal of Computer Vision 101.1 (2013): 184-204.
    https://github.com/cvondrick/vatic
    """

    DETRAC_MAT = 'detrac-mat'
    """Wen, Longyin et al. "UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking." arXiv preprint arXiv:arXiv:1511.04136 (2016).
    http://detrac-db.rit.albany.edu/download
    """

    DETRAC_XML = 'detrac-xml'
    """Wen, Longyin et al. "UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking." arXiv preprint arXiv:arXiv:1511.04136 (2016).
    http://detrac-db.rit.albany.edu/download
    """


class _SequenceData(object):
    """Compact columnar detections consumed directly by the metric engine."""

    __slots__ = ("frame_ids", "ids", "_fields")

    def __init__(self, frame_ids, ids, fields):
        self.frame_ids = np.asarray(frame_ids, dtype=np.int64)
        self.ids = np.asarray(ids, dtype=np.int64)
        self._fields = {name: np.asarray(values) for name, values in fields.items()}
        row_count = len(self.frame_ids)
        if len(self.ids) != row_count or any(len(values) != row_count for values in self._fields.values()):
            raise ValueError("Every sequence column must have the same length.")

    def __len__(self):
        return len(self.frame_ids)

    def column(self, name):
        """Return one stored column without copying it."""
        return self._fields[name]

    def values(self, names):
        """Return selected fields as the floating matrix used for distances."""
        try:
            columns = [self._fields[name] for name in names]
        except KeyError as exc:
            raise ValueError("Unknown distance field: {}".format(exc.args[0])) from exc
        if not columns:
            return np.empty((len(self), 0), dtype=float)
        return np.column_stack(columns).astype(float, copy=False)


def load_motchallenge(fname, **kwargs):
    r"""Load MOT challenge data.

    Params
    ------
    fname : str
        Filename to load data from

    Kwargs
    ------
    sep : str
        Allowed field separators, defaults to '\s+|\t+|,'
    min_confidence : float
        Rows with confidence less than this threshold are removed.
        Defaults to -1. You should set this to 1 when loading
        ground truth MOTChallenge data, so that invalid rectangles in
        the ground truth are not considered during matching.

    Returns
    -------
    _SequenceData
        Compact columns indexed by frame and identity arrays.
    """

    sep = kwargs.pop('sep', None)
    min_confidence = kwargs.pop('min_confidence', -1)
    text = _read_text(fname)
    first_line = next((line for line in text.splitlines() if line.strip()), '')
    if not first_line:
        return _empty_mot_sequence()

    if sep is None:
        sep = ',' if ',' in first_line else r'\s+'
    if sep == ',':
        normalized = text.replace(',', ' ')
        width = len(first_line.split(','))
    elif sep in (r'\s+', ' ', '\t'):
        normalized = text
        width = len(first_line.split())
    else:
        import re

        normalized = re.sub(sep, ' ', text)
        width = len(re.split(sep, first_line.strip()))

    flat = np.fromstring(normalized, sep=' ', dtype=float)
    if width < 2 or flat.size % width:
        raise ValueError("Invalid MOTChallenge rows in {}".format(fname))
    raw = flat.reshape(-1, width)
    if width >= 9:
        data = raw[:, :9]
    else:
        data = np.full((len(raw), 9), np.nan, dtype=float)
        data[:, :width] = raw

    data[:, 2:4] -= 1
    keep = data[:, 6] >= min_confidence
    if not keep.all():
        data = data[keep]
    fields = {
        name: data[:, column]
        for name, column in {
            'X': 2,
            'Y': 3,
            'Width': 4,
            'Height': 5,
            'Confidence': 6,
            'ClassId': 7,
            'Visibility': 8,
        }.items()
    }
    return _SequenceData(data[:, 0], data[:, 1], fields)


def _read_text(source):
    if hasattr(source, 'read'):
        data = source.read()
    else:
        with io.open(source, encoding='utf-8', errors='ignore') as file:
            data = file.read()
    if isinstance(data, bytes):
        return data.decode('utf-8', errors='ignore')
    return data


def _empty_mot_sequence():
    empty = np.empty(0, dtype=float)
    return _SequenceData(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        {
            'X': empty,
            'Y': empty,
            'Width': empty,
            'Height': empty,
            'Confidence': empty,
            'ClassId': empty,
            'Visibility': empty,
        },
    )


def load_vatictxt(fname, **kwargs):
    """Load Vatic text format.

    Loads the vatic CSV text having the following columns per row

        0   Track ID. All rows with the same ID belong to the same path.
        1   xmin. The top left x-coordinate of the bounding box.
        2   ymin. The top left y-coordinate of the bounding box.
        3   xmax. The bottom right x-coordinate of the bounding box.
        4   ymax. The bottom right y-coordinate of the bounding box.
        5   frame. The frame that this annotation represents.
        6   lost. If 1, the annotation is outside of the view screen.
        7   occluded. If 1, the annotation is occluded.
        8   generated. If 1, the annotation was automatically interpolated.
        9  label. The label for this annotation, enclosed in quotation marks.
        10+ attributes. Each column after this is an attribute set in the current frame

    Params
    ------
    fname : str
        Filename to load data from

    Returns
    -------
    _SequenceData
        Compact detection columns consumed by the metric engine.
    """
    import shlex

    kwargs.pop('sep', ' ')
    rows = [shlex.split(line) for line in _read_text(fname).splitlines() if line.strip()]
    activities = sorted({activity for row in rows for activity in row[10:]})

    frame_ids = np.fromiter((int(row[5]) for row in rows), dtype=np.int64, count=len(rows))
    ids = np.fromiter((int(row[0]) for row in rows), dtype=np.int64, count=len(rows))
    x = np.fromiter((float(row[1]) for row in rows), dtype=float, count=len(rows))
    y = np.fromiter((float(row[2]) for row in rows), dtype=float, count=len(rows))
    xmax = np.fromiter((float(row[3]) for row in rows), dtype=float, count=len(rows))
    ymax = np.fromiter((float(row[4]) for row in rows), dtype=float, count=len(rows))
    fields = {
        'X': x,
        'Y': y,
        'Width': xmax - x,
        'Height': ymax - y,
        'Lost': np.fromiter((bool(int(row[6])) for row in rows), dtype=bool, count=len(rows)),
        'Occluded': np.fromiter((bool(int(row[7])) for row in rows), dtype=bool, count=len(rows)),
        'Generated': np.fromiter((bool(int(row[8])) for row in rows), dtype=bool, count=len(rows)),
        'ClassId': np.asarray([row[9] for row in rows], dtype=object),
    }
    for activity in activities:
        fields[activity.capitalize()] = np.fromiter(
            (activity in row[10:] for row in rows),
            dtype=bool,
            count=len(rows),
        )
    return _SequenceData(frame_ids, ids, fields)


def load_detrac_mat(fname, **kwargs):
    """Loads UA-DETRAC annotations data from mat files

    Competition Site: http://detrac-db.rit.albany.edu/download

    File contains a nested structure of 2d arrays for indexed by frame id
    and Object ID. Separate arrays for top, left, width and height are given.

    Params
    ------
    fname : str
        Filename to load data from

    Kwargs
    ------
    Currently none of these arguments used.

    Returns
    -------
    _SequenceData
        Compact detection columns consumed by the metric engine.
    """

    from scipy.io import loadmat

    mat_data = loadmat(fname)

    frame_list = mat_data['gtInfo'][0][0][4][0]
    left_array = mat_data['gtInfo'][0][0][0].astype(np.float32)
    top_array = mat_data['gtInfo'][0][0][1].astype(np.float32)
    width_array = mat_data['gtInfo'][0][0][3].astype(np.float32)
    height_array = mat_data['gtInfo'][0][0][2].astype(np.float32)

    parsed_gt = []
    for f in frame_list:
        f = int(f)
        ids = [i + 1 for i, v in enumerate(left_array[f - 1]) if v > 0]
        for i in ids:
            parsed_gt.append((
                f,
                i,
                left_array[f - 1, i - 1] - width_array[f - 1, i - 1] / 2 - 1,
                top_array[f - 1, i - 1] - height_array[f - 1, i - 1] - 1,
                width_array[f - 1, i - 1],
                height_array[f - 1, i - 1],
                1,
                -1,
                -1,
            ))
    return _sequence_from_mot_rows(parsed_gt)


def load_detrac_xml(fname, **kwargs):
    """Loads UA-DETRAC annotations data from xml files

    Competition Site: http://detrac-db.rit.albany.edu/download

    Params
    ------
    fname : str
        Filename to load data from

    Kwargs
    ------
    Currently none of these arguments used.

    Returns
    -------
    _SequenceData
        Compact detection columns consumed by the metric engine.
    """
    import xml.etree.ElementTree

    root = xml.etree.ElementTree.parse(fname).getroot()
    frame_list = root.findall('frame')

    parsed_gt = []
    for frame in frame_list:
        fid = int(frame.attrib['num'])
        target_list = frame.find('target_list')
        if target_list is None:
            continue

        for target in target_list.findall('target'):
            box = target.find('box')
            if box is None:
                continue
            parsed_gt.append((
                fid,
                int(target.attrib['id']),
                float(box.attrib['left']) - 1,
                float(box.attrib['top']) - 1,
                float(box.attrib['width']),
                float(box.attrib['height']),
                1,
                -1,
                -1,
            ))
    return _sequence_from_mot_rows(parsed_gt)


def _sequence_from_mot_rows(rows):
    if not rows:
        return _empty_mot_sequence()
    matrix = np.asarray(rows, dtype=float)
    return _SequenceData(
        matrix[:, 0],
        matrix[:, 1],
        {
            'X': matrix[:, 2],
            'Y': matrix[:, 3],
            'Width': matrix[:, 4],
            'Height': matrix[:, 5],
            'Confidence': matrix[:, 6],
            'ClassId': matrix[:, 7],
            'Visibility': matrix[:, 8],
        },
    )


def infer_format(fname):
    """Infer a supported file format from the path and a small data sample."""
    path = Path(fname)
    suffix = path.suffix.lower()
    if suffix == '.mat':
        return Format.DETRAC_MAT
    if suffix == '.xml':
        return Format.DETRAC_XML

    with io.open(fname, encoding='utf-8', errors='ignore') as file:
        sample = file.read(4096)

    stripped = sample.lstrip()
    if stripped.startswith('<'):
        return Format.DETRAC_XML

    first_line = next((line.strip() for line in sample.splitlines() if line.strip()), '')
    if not first_line:
        raise ValueError('Cannot infer format from empty file: {}'.format(fname))

    fields = _split_text_fields(first_line)
    if _looks_like_vatic_fields(fields):
        return Format.VATIC_TXT
    return Format.MOT15_2D


def loadtxt(fname, fmt=Format.MOT15_2D, **kwargs):
    """Load data from any known format."""
    fmt = Format(fmt)
    if fmt == Format.AUTO:
        fmt = infer_format(fname)

    switcher = {
        Format.MOT16: load_motchallenge,
        Format.MOT15_2D: load_motchallenge,
        Format.VATIC_TXT: load_vatictxt,
        Format.DETRAC_MAT: load_detrac_mat,
        Format.DETRAC_XML: load_detrac_xml
    }
    func = switcher.get(fmt)
    return func(fname, **kwargs)


def _split_text_fields(line):
    return line.replace(',', ' ').split()


def _looks_like_vatic_fields(fields):
    if len(fields) < 10:
        return False
    return len(fields) > 10 or not _is_number(fields[9])


def _is_number(value):
    try:
        float(value)
    except ValueError:
        return False
    return True


def render_summary(rows, index, columns, formatters=None, namemap=None, buf=None):
    """Render metrics summary to console friendly tabular output.

    Params
    ------
    rows : sequence of mappings
        Metric values in display order.
    index : sequence of str
        Row labels.
    columns : sequence of str
        Metric names in display order.

    Kwargs
    ------
    buf : StringIO-like, optional
        Buffer to write to
    formatters : dict, optional
        Dictionary defining custom formatters for individual metrics, such as
        ``{'mota': '{:.2%}'.format}``.
    namemap : dict, optional
        Dictionary defining new metric names for display. I.e
        `{'num_false_positives': 'FP'}`.

    Returns
    -------
    string
        Formatted string
    """

    formatters = formatters or {}
    namemap = namemap or {}
    headers = [namemap.get(column, column) for column in columns]
    formatted_rows = [
        [formatters[column](row[column]) if column in formatters else str(row[column]) for column in columns]
        for row in rows
    ]
    index_strings = [str(value) for value in index]
    index_width = max([len(value) for value in index_strings] or [0])
    column_widths = [
        max([len(header)] + [len(row[column_index]) for row in formatted_rows])
        for column_index, header in enumerate(headers)
    ]
    lines = [
        " " * (index_width + 1)
        + " ".join(header.rjust(width) for header, width in zip(headers, column_widths))
    ]
    lines.extend(
        row_name.ljust(index_width)
        + " "
        + " ".join(value.rjust(width) for value, width in zip(row, column_widths))
        for row_name, row in zip(index_strings, formatted_rows)
    )
    output = "\n".join(lines)
    if buf is not None:
        buf.write(output)
        return None
    return output


motchallenge_metric_names = {
    'idf1': 'IDF1',
    'idp': 'IDP',
    'idr': 'IDR',
    'recall': 'Rcll',
    'precision': 'Prcn',
    'num_unique_objects': 'GT',
    'mostly_tracked': 'MT',
    'partially_tracked': 'PT',
    'mostly_lost': 'ML',
    'num_false_positives': 'FP',
    'num_misses': 'FN',
    'num_switches': 'IDs',
    'num_fragmentations': 'FM',
    'mota': 'MOTA',
    'motp': 'MOTP',
    'num_transfer': 'IDt',
    'num_ascend': 'IDa',
    'num_migrate': 'IDm',
}
"""A list mappings for metric names to comply with MOTChallenge."""
