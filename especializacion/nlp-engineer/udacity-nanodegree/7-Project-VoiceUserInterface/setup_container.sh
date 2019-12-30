#!/bin/bash
pip install -r requirements.txt
KERAS_BACKEND=tensorflow python -c "from keras import backend"
apt-get install -y libav-tools
#jupyter lab &

