#!/bin/bash

#conda create --file requirements.txt --prefix /opt/nlpnd

conda create --prefix /opt/nlpnd python=3.6.3
source activate /opt/nlpnd
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt
