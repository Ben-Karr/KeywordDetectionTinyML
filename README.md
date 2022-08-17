# Keyword Detection for Arduino with TinyML
Builds on the [Deploy TinyML](https://learning.edx.org/course/course-v1:HarvardX+TinyML3+1T2022/home) course on [edX](https://edx.org). I used the great tools provided below to build my own dataset and you can see my application of the machine learning part to the Arduino [here](arduino/micro_speech).

## Notebooks:
* __[[TF]-Course-Baseline](https://github.com/Ben-Karr/KeywordDetectionTinyML/blob/master/%5BTF%5D-Course-Baseline.ipynb):__ 
This is a boiled down version of the [Custom Dataset Notebook](https://colab.research.google.com/github/tinyMLx/colabs/blob/master/4-6-8-CustomDatasetKWSModel.ipynb) from the [Deploying TinyML](https://learning.edx.org/course/course-v1:HarvardX+TinyML3+1T2022/home) course. It leaves out all annotations and assumes that all data (including custom datasets) are prepared and in an unified location.
* __[[Keras]-Preprocess-Functional](https://github.com/Ben-Karr/KeywordDetectionTinyML/blob/master/%5BKeras%5D-Preprocess-Functional.ipynb):__ 
Uses keras.utils.Squence to build a dataset and keras Functional API to build a model. It allows easy adjustment of the dataloading process including augmentation, rapid model development and transfer learning. Most of the functionality is kept as in the [Tensorflow example for speech commands](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands) (in particular: [input_data.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/input_data.py), [models.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/models.py), [train.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/train.py)) but avoids TF1 style / graphs / sessions.
* __[[Keras]KeywordDataset_Demo]():__ Demos the usage of [KeywordDataset]() as a module.

## Data Sources:
* [Speech Commands](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz): Collection of 1s .wav files of different keywords.

## Tools:
* [Open Speech Recording](https://github.com/petewarden/open-speech-recording) Record short audio clips as .ogg.
* [Extract Loudest Section](https://github.com/petewarden/extract_loudest_section.git) Extract the loudest part of a .wav file.
