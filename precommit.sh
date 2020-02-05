#!/bin/bash

set -o xtrace

pytest --benchmark-disable && \
flake8 . && \
pylint motmetrics && \
pylint --py3k motmetrics
