# Udacity Natural Language Processing Nanodegree

This repository contains the Tutorials and my solutions to the NLP Nanodegree's Projects.

See https://www.udacity.com/course/natural-language-processing-nanodegree--nd892

Original source code retrieved from Udacity online environments and from the following Udacity repository:
- https://github.com/udacity/hmm-tagger
- https://github.com/udacity/AIND-VUI-Capstone


## Installation notes

I used a Ubuntu 16.04 docker image to run this code (with a Nvidia GPU installed on the Ubuntu Docker host)

You should create a Conda Virtual environments using `conda env create -f conda-env.yml` command. Normally the created virtual environment should contain all the required Python libraries to execute the projects. (Don't forget to activate the conda environment with `conda activate nlpnd` before launching Jupyter Lab)

For the `7-Project-VoiceUserInterface` project, you should execute the `setup_container.sh` inside the virtual environment to meet the required configuration (this will downgrade the Keras version and install additional packages)
