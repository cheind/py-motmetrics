FROM ubuntu:latest

MAINTAINER Avgerinos Christos <christosavg@gmail.com>

#ARG GT_DIR
#ARG TEST_DIR

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev vim \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN pip3 install --no-cache-dir numpy scipy
RUN pip install -Iv pandas==0.21.0
RUN mkdir -p /motmetrics/py-motmetrics
RUN mkdir -p /motmetrics/2DMOT2015

COPY ./py-motmetrics /motmetrics/py-motmetrics
COPY ./data /motmetrics/data

#RUN pip install motmetrics
RUN pip install -e ./motmetrics/py-motmetrics/

#RUN pip install -r motmetrics/py-motmetrics/requirements.txt

ENV GT_DIR motmetrics/data/train/
ENV TEST_DIR motmetrics/data/test/

#ENTRYPOINT python3 -m motmetrics.apps.eval_motchallenge motmetrics/data/train/ motmetrics/data/test/ && /bin/bash
CMD ["sh", "-c", "python3 -m motmetrics.apps.eval_motchallenge ${GT_DIR} ${TEST_DIR} && /bin/bash"]

