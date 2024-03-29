import json
import os
import time
from tensorflow.keras.layers import LSTM

# Window size or the sequence length
N_STEPS = 60
# Lookup step, 1 is the next day
#LOOKUP_STEP = 15
LOOKUP_STEP = 7

# whether to scale feature columns & output price as well
SCALE = True
scale_str = f"sc-{int(SCALE)}"
# whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"
# whether to split the training/testing set by date
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")

### model parameters

N_LAYERS = 2
#N_LAYERS = 4

# LSTM cell
CELL = LSTM

# 256 LSTM neurons
UNITS = 256
#UNITS = 512

# 40% dropout
DROPOUT = 0.4

# whether to use bidirectional RNNs
BIDIRECTIONAL = True

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
#BATCH_SIZE = 128
EPOCHS = 500

# Amazon stock market
#ticker = "AMZN"
#ticker = "SHOP"
ticker = "SPY"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"

def load_config(config_file, config_id):
	file = open(config_file)
	config = json.load(file)
	hyperparameters = config[str(config_id)]["hyperparameters"]
	N_STEPS = hyperparameters["window"]
	N_LAYERS = hyperparameters["layers"]
	UNITS = hyperparameters["neurons"]
	DROPOUT = hyperparameters["dropout"]
	BIDIRECTIONAL = hyperparameters["bidirectional"]
	BATCH_SIZE = hyperparameters["batch"]
	ticker = hyperparameters["ticker"]

load_config("configs/hyperparameters/config.json", 4)
