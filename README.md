# Stonks

A python program that uses machine learning to predict future stock market prices based on historical data.

This project was only meant for fun and to explore ML and its potential uses.
Any predictions and/or results of Stongs should not be taken as fact and should only serve as a guide to potential trends.

## Requirements

Nvidia GPU: Not strictly necessary, but speeds up training and predictions

Nvidia cuDNN: Required for using Nvidia GPU

Python: I used python 3.10, and have not tested with other python versions


<br />

## Installation

### Install python requirements

`pip3 install -r requirements.txt`

<br />

### Install cuda requirements

Download and instal cuDNN for your desired GPU

https://developer.nvidia.com/cudnn

<br />

### Extra installation steps

Link libdevice.10.bc (this may need to be done if it is not detected by your installation): 

`export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda`

<br />

## Usage

**WIP**

Basic Usage `python3 stonks_new.py -m <mode> --ticker <ticker symbol> --days/date <num_days>/<date> --epochs <num_epochs>`

Modes:
 - train: trains a new model with given parameters
 - predict: use existing model with given parameters for a single prediction n days in the future
 - forecast: use existing model with given parameters for predictions from now until n days in the future
 
 Additional Arguments:
  - --crush: crush the price data to a normalized range of 0 - 100
  - --bidirections: use bidirectional neurons
  - --force: force retraining of model with given parameters
  - --epochs: epochs to train for (default 500)
  
<br />

## Credits

Stonks is based off of [this project](https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras) 
and it's related [code base](https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction) created by Rockikz 
