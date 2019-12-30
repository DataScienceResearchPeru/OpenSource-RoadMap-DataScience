[//]: # (Image References)

[image1]: ./images/pipeline.png "ASR Pipeline"
[image2]: ./images/select_kernel.png "select aind-vui kernel"

## Project Overview

In this notebook, you will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline!  

![ASR Pipeline][image1]

We begin by investigating the [LibriSpeech dataset](http://www.openslr.org/12/) that will be used to train and evaluate your models. Your algorithm will first convert any raw audio to feature representations that are commonly used for ASR. You will then move on to building neural networks that can map these audio features to transcribed text. After learning about the basic types of layers that are often used for deep learning-based approaches to ASR, you will engage in your own investigations by creating and testing your own state-of-the-art models. Throughout the notebook, we provide recommended research papers for additional reading and links to GitHub repositories with interesting implementations.

## Project Instructions

### Amazon Web Services

This project requires GPU acceleration to run efficiently. Please refer to the Udacity instructions for setting up a GPU instance for this project, and refer to the project instructions in the classroom for setup. [link for AIND students](https://classroom.udacity.com/nanodegrees/nd889/parts/4550d1eb-a3e0-4e9b-9d3c-4f55aa6662b5/modules/c8419a1e-acd3-4463-9c01-a4c93f7c3b24/lessons/b27e9b6a-bb3b-4f3e-8993-bdfcb662a426/concepts/61c0743f-22f1-47db-a4d2-5616c25fc888)

1. Follow the Cloud Computing Setup instructions lesson to create an EC2 instance. (The lesson includes all the required package and library installation instructions.)

2. Obtain the appropriate subsets of the LibriSpeech dataset, and convert all flac files to wav format.
```
wget http://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzvf dev-clean.tar.gz
wget http://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzvf test-clean.tar.gz
mv flac_to_wav.sh LibriSpeech
cd LibriSpeech
./flac_to_wav.sh
```

3. Create JSON files corresponding to the train and validation datasets.
```
cd ..
python create_desc_json.py LibriSpeech/dev-clean/ train_corpus.json
python create_desc_json.py LibriSpeech/test-clean/ valid_corpus.json
```

4. Start Jupyter:
```
jupyter notebook --ip=0.0.0.0 --no-browser
```

5. Look at the output in the window, and find the line that looks like: `http://0.0.0.0:8888/?token=3156e...` Copy and paste the **complete** URL into the address bar of a web browser (Firefox, Safari, Chrome, etc). Before navigating to the URL, replace 0.0.0.0 in the URL with the "IPv4 Public IP" address from the EC2 Dashboard.


### Local Environment Setup

You should run this project with GPU acceleration for best performance.

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/udacity/AIND-VUI-Capstone.git
cd AIND-VUI-Capstone
```

2. Create (and activate) a new environment with Python 3.6 and the `numpy` package.

	- __Linux__ or __Mac__: 
	```
	conda create --name aind-vui python=3.5 numpy
	source activate aind-vui
	```
	- __Windows__: 
	```
	conda create --name aind-vui python=3.5 numpy scipy
	activate aind-vui
	```

3. Install TensorFlow.
	- Option 1: __To install TensorFlow with GPU support__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step and only need to install the `tensorflow-gpu` package:
	```
	pip install tensorflow-gpu==1.1.0
	```
	- Option 2: __To install TensorFlow with CPU support only__,
	```
	pip install tensorflow==1.1.0
	```

4. Install a few pip packages.
```
pip install -r requirements.txt
```

5. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
	```
	KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```
	- __Windows__: 
	```
	set KERAS_BACKEND=tensorflow
	python -c "from keras import backend"
	```
	- __NOTE:__ a Keras/Windows bug may give this error after the first epoch of training model 0: `‘rawunicodeescape’ codec can’t decode bytes in position 54-55: truncated \uXXXX `. 
To fix it: 
		- Find the file `keras/utils/generic_utils.py` that you are using for the capstone project. It should be in your environment under `Lib/site-packages` . This may vary, but if using miniconda, for example, it might be located at `C:/Users/username/Miniconda3/envs/aind-vui/Lib/site-packages/keras/utils`.
		- Copy `generic_utils.py` to `OLDgeneric_utils.py` just in case you need to restore it.
		- Open the `generic_utils.py` file and change this code line:</br>`marshal.dumps(func.code).decode(‘raw_unicode_escape’)`</br>to this code line:</br>`marshal.dumps(func.code).replace(b’\’,b’/’).decode(‘raw_unicode_escape’)`

6. Obtain the `libav` package.
	- __Linux__: `sudo apt-get install libav-tools`
	- __Mac__: `brew install libav`
	- __Windows__: Browse to the [Libav website](https://libav.org/download/)
		- Scroll down to "Windows Nightly and Release Builds" and click on the appropriate link for your system (32-bit or 64-bit).
		- Click `nightly-gpl`.
		- Download most recent archive file.
		- Extract the file.  Move the `usr` directory to your C: drive.
		- Go back to your terminal window from above.
	```
	rename C:\usr avconv
    set PATH=C:\avconv\bin;%PATH%
	```

7. Obtain the appropriate subsets of the LibriSpeech dataset, and convert all flac files to wav format.
	- __Linux__ or __Mac__: 
	```
	wget http://www.openslr.org/resources/12/dev-clean.tar.gz
	tar -xzvf dev-clean.tar.gz
	wget http://www.openslr.org/resources/12/test-clean.tar.gz
	tar -xzvf test-clean.tar.gz
	mv flac_to_wav.sh LibriSpeech
	cd LibriSpeech
	./flac_to_wav.sh
	```
	- __Windows__: Download two files ([file 1](http://www.openslr.org/resources/12/dev-clean.tar.gz) and [file 2](http://www.openslr.org/resources/12/test-clean.tar.gz)) via browser and save in the `AIND-VUI-Capstone` directory.  Extract them with an application that is compatible with `tar` and `gz` such as [7-zip](http://www.7-zip.org/) or [WinZip](http://www.winzip.com/). Convert the files from your terminal window.
	```
	move flac_to_wav.sh LibriSpeech
	cd LibriSpeech
	powershell ./flac_to_wav.sh
	```

8. Create JSON files corresponding to the train and validation datasets.
```
cd ..
python create_desc_json.py LibriSpeech/dev-clean/ train_corpus.json
python create_desc_json.py LibriSpeech/test-clean/ valid_corpus.json
```

9. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `aind-vui` environment.  Open the notebook.
```
python -m ipykernel install --user --name aind-vui --display-name "aind-vui"
jupyter notebook vui_notebook.ipynb
```

10. Before running code, change the kernel to match the `aind-vui` environment by using the drop-down menu.  Then, follow the instructions in the notebook.

![select aind-vui kernel][image2]

__NOTE:__ While some code has already been implemented to get you started, you will need to implement additional functionality to successfully answer all of the questions included in the notebook. __Unless requested, do not modify code that has already been included.__


### Evaluation

Your project will be reviewed by a Udacity reviewer against the CNN project [rubric](#rubric).  Review this rubric thoroughly, and self-evaluate your project before submission.  All criteria found in the rubric must meet specifications for you to pass.


### Project Submission

When you are ready to submit your project, collect the following files and compress them into a single archive for upload:
- The `vui_notebook.ipynb` file with fully functional code, all code cells executed and displaying output, and all questions answered.
- An HTML or PDF export of the project notebook with the name `report.html` or `report.pdf`.
- The `sample_models.py` file with all model architectures that were trained in the project Jupyter notebook.
- The `results/` folder containing all HDF5 and pickle files corresponding to trained models.

Alternatively, your submission could consist of the GitHub link to your repository.


<a id='rubric'></a>
## Project Rubric

#### Files Submitted

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Submission Files      | The submission includes all required files.		|

#### STEP 2: Model 0: RNN

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Trained Model 0         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_0.pickle` are undefined.  The trained weights for the model specified in `simple_rnn_model` are stored in `model_0.h5`.   	|

#### STEP 2: Model 1: RNN + TimeDistributed Dense

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Completed `rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `rnn_model` module containing the correct architecture.   	|
| Trained Model 1         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_1.pickle` are undefined.  The trained weights for the model specified in `rnn_model` are stored in `model_1.h5`.   	|

#### STEP 2: Model 2: CNN + RNN + TimeDistributed Dense

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Completed `cnn_rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `cnn_rnn_model` module containing the correct architecture.   	|
| Trained Model 2         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_2.pickle` are undefined.  The trained weights for the model specified in `cnn_rnn_model` are stored in `model_2.h5`.   	|

#### STEP 2: Model 3: Deeper RNN + TimeDistributed Dense

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Completed `deep_rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `deep_rnn_model` module containing the correct architecture.   	|
| Trained Model 3         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_3.pickle` are undefined.  The trained weights for the model specified in `deep_rnn_model` are stored in `model_3.h5`.   	|

#### STEP 2: Model 4: Bidirectional RNN + TimeDistributed Dense

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Completed `bidirectional_rnn_model` Module         		| The submission includes a `sample_models.py` file with a completed `bidirectional_rnn_model` module containing the correct architecture.   	|
| Trained Model 4         		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_4.pickle` are undefined.  The trained weights for the model specified in `bidirectional_rnn_model` are stored in `model_4.h5`.   	|

#### STEP 2: Compare the Models

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Question 1         		| The submission includes a detailed analysis of why different models might perform better than others.   	|

#### STEP 2: Final Model

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------:|:---------------------------------------------------------:| 
| Completed `final_model` Module         		| The submission includes a `sample_models.py` file with a completed `final_model` module containing a final architecture that is not identical to any of the previous architectures.   	|
| Trained Final Model        		| The submission trained the model for at least 20 epochs, and none of the loss values in `model_end.pickle` are undefined.  The trained weights for the model specified in `final_model` are stored in `model_end.h5`.   	|
| Question 2         		| The submission includes a detailed description of how the final model architecture was designed.   	|


## Suggestions to Make your Project Stand Out!

#### (1) Add a Language Model to the Decoder

The performance of the decoding step can be greatly enhanced by incorporating a language model.  Build your own language model from scratch, or leverage a repository or toolkit that you find online to improve your predictions.

#### (2) Train on Bigger Data

In the project, you used some of the smaller downloads from the LibriSpeech corpus.  Try training your model on some larger datasets - instead of using `dev-clean.tar.gz`, download one of the larger training sets on the [website](http://www.openslr.org/12/).

#### (3) Try out Different Audio Features

In this project, you had the choice to use _either_ spectrogram or MFCC features.  Take the time to test the performance of _both_ of these features.  For a special challenge, train a network that uses raw audio waveforms!

## Special Thanks

We have borrowed the `create_desc_json.py` and `flac_to_wav.sh` files from the [ba-dls-deepspeech](https://github.com/baidu-research/ba-dls-deepspeech) repository, along with some functions used to generate spectrograms.
