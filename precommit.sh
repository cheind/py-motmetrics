#!/bin/bash

set -o xtrace

pytest --benchmark-disable && \
flake8 .
