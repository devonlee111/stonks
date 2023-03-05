import json
import os
import random
import sys
import time

from collections import deque
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from yahoo_fin import stock_info as si

CURRENT_CONFIGS="configs/hyperparameters/config.json"
HISTORICAL_CONFIGS="configs/hyperparameters/historical.json"
MODELS_DIR="models/"

@dataclass
class Config():
    """
    structure for storing configs
    """

    n_steps = 130
    lookup_step = 7
    n_layers = 2
    units = 256
    dropout = 0.4
    bidirectional = True
    batch_size = 64
    epochs = 500
    ticker = "SPY"

    def __init__(self, n_steps, lookup_step, n_layers, units, dropout, bidirectional, batch_size, epochs, ticker):
        self.n_steps = n_steps
        self.lookup_step = lookup_step
        self.n_layers = n_layers
        self.units = units
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.epochs = epochs
        self.ticker = ticker

# Extra tickers/indeces for training
SP500 = "^GSPC"
DOW = "^DJI"
VIX = "^VIX"
NASDAQ = "^IXIC"
RUSSEL2000 = "^RUT"

#################### PARAMETERS ####################
# Define hyperparameters for models
# Default to base setting hyperparameters

# Window size or the sequence length
n_steps = 130
# Lookup step, 1 is the next day
#LOOKUP_STEP = 15
lookup_step = 7

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

n_layers = 2
#N_LAYERS = 4

# LSTM cell
CELL = LSTM

# 256 LSTM neurons
units = 256
#UNITS = 512

# 40% dropout
dropout = 0.4

# whether to use bidirectional RNNs
bidirectional = True

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
batch_size = 64
#BATCH_SIZE = 128
epochs = 500

# SPY (S&P500) ticker
ticker = "SPY"
ticker_data_filename = ""
# model name to save, making it as unique as possible based on parameters
model_name = ""

def load_config(config_file, config_id):
    """
    load hyperparameters from config file
    """
    global n_steps, n_layers, units, dropout, bidirectional, batch_size
    file = open(config_file)
    config = json.load(file)
    hyperparameters = config[str(config_id)]["hyperparameters"]
    n_steps = int(hyperparameters["window"])
    n_layers = int(hyperparameters["layers"])
    units = int(hyperparameters["neurons"])
    dropout = int(hyperparameters["dropout"])
    bidirectional = hyperparameters["bidirectional"]
    batch_size = int(hyperparameters["batch"])
    #ticker = hyperparameters["ticker"]



#################### TRAINING ####################
# Train a model with given hyperparameters to predict stock prices of specific stock based on historical data

# create these folders if they does not exist
def setup_environment():
    """
    create necessary directories if needed
    """
    if not os.path.isdir("results"):
        os.mkdir("results")

    if not os.path.isdir("logs"):
        os.mkdir("logs")

    if not os.path.isdir("data"):
        os.mkdir("data")

def train():
    """
    train a new model for the given ticker
    """
    # load the data
    data = load_data(ticker, n_steps=n_steps, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                    shuffle=SHUFFLE, lookup_step=lookup_step, test_size=TEST_SIZE,
                    feature_columns=FEATURE_COLUMNS)

    # save the dataframe
    data["df"].to_csv(ticker_data_filename)

    # construct the model
    model = create_model(n_steps, len(FEATURE_COLUMNS), loss=LOSS, units=units, cell=CELL, n_layers=n_layers,
                    dropout=dropout, optimizer=OPTIMIZER, bidirectional=bidirectional)

    # some tensorflow callbacks
    checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    # train the model and save the weights whenever we see
    # a new optimal model using ModelCheckpoint

    #history =
    model.fit(data["X_train"], data["y_train"],
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)



#################### PREDICTION ####################
# Use given data and recent stock prices to predict future prices

def plot_graph(test_df):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f'true_adjclose_{lookup_step}'], c='b')
    plt.plot(test_df[f'adjclose_{lookup_step}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


def get_final_df(model, data):
    """
    This function takes the `model` and `data` dict to
    construct a final dataframe that includes the features along
    with true and predicted prices of the testing dataset
    """
    # if predicted future price is higher than the current,
    # then calculate the true future price minus the current price, to get the buy profit
    buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    # if the predicted future price is lower than the current price,
    # then subtract the true future price from the current price
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    X_test = data["X_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    test_df = data["test_df"]
    # add predicted future prices to the dataframe
    test_df[f"adjclose_{lookup_step}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"true_adjclose_{lookup_step}"] = y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df
    # add the buy profit column
    final_df["buy_profit"] = list(map(buy_profit,
                                    final_df["adjclose"],
                                    final_df[f"adjclose_{lookup_step}"],
                                    final_df[f"true_adjclose_{lookup_step}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    # add the sell profit column
    final_df["sell_profit"] = list(map(sell_profit,
                                    final_df["adjclose"],
                                    final_df[f"adjclose_{lookup_step}"],
                                    final_df[f"true_adjclose_{lookup_step}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    return final_df


def predict(model, data):
    """calculate future price prdictions"""
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-n_steps:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price


def test():
    """predict future prices"""

    # load the data
    data = load_data(ticker, n_steps, scale=SCALE, split_by_date=SPLIT_BY_DATE,
        shuffle=SHUFFLE, lookup_step=lookup_step,
        test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)

    # construct the model
    model = create_model(n_steps, len(FEATURE_COLUMNS), loss=LOSS, units=units, cell=CELL, 
        n_layers=n_layers,dropout=dropout, optimizer=OPTIMIZER, bidirectional=bidirectional)

    # load optimal model weights from results folder
    model_path = os.path.join("results", model_name) + ".h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError

    model.load_weights(model_path)

    # evaluate the model
    loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    # calculate the mean absolute error (inverse scaling)
    if SCALE:
        mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae

    # get the final dataframe for the testing set
    final_df = get_final_df(model, data)

    # predict the future price
    future_price = predict(model, data)

    # we calculate the accuracy by counting the number of positive profits
    #accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)

    # calculating total buy & sell profit
    #total_buy_profit  = final_df["buy_profit"].sum()
    #total_sell_profit = final_df["sell_profit"].sum()

    # total profit by adding sell & buy together
    #total_profit = total_buy_profit + total_sell_profit

    # dividing total profit by number of testing samples (number of trades)
    #profit_per_trade = total_profit / len(final_df)
    # printing metrics
    final_price = f"${future_price:.2f}"
    print(f"Future price of {ticker} after {lookup_step} days is {final_price}")
    print(f"{LOSS} loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    #print("Accuracy score:", accuracy_score)
    #print("Total buy profit:", total_buy_profit)
    #print("Total sell profit:", total_sell_profit)
    #print("Total profit:", total_profit)
    #print("Profit per trade:", profit_per_trade)
    # plot true/pred prices graph
    plot_graph(final_df)
    #print(final_df.tail(10))
    # save the final dataframe to csv-results folder
    csv_results_folder = "csv-results"
    if not os.path.isdir(csv_results_folder):
        os.mkdir(csv_results_folder)
    csv_filename = os.path.join(csv_results_folder, model_name + ".csv")
    final_df.to_csv(csv_filename)

    return final_price



#################### ENGINE ####################
# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        split_by_date (bool): whether we split the dataset into training/testing by date, setting it 
            to False will split datasets in a random way
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    # add date as a column
    if "date" not in df.columns:
        df["date"] = df.index

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # drop NaNs
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence

    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                test_size=test_size, shuffle=shuffle)

    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result


def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model



#################### TUNING ####################
# Implement a genetic algorithm to tune hyperparameters over time



#################### PROGRAM ####################

def train_and_predict():
    """train on historical data and then predict future prices"""
    train()
    return test()

def stonk_help():
    """print out help on how to use program"""
    print("\n\n\n\nusage python stonks.py \"train\"/\"predict\" model_number/\"default\" lookup_step \"ticker\" epochs")

if len(sys.argv) != 5 and len(sys.argv) != 6:
    stonk_help()
    exit()

program_option = sys.argv[1]
model_number = sys.argv[2]
lookup_days = sys.argv[3]
stock_ticker = sys.argv[4]
num_epochs=str(epochs)
if len(sys.argv) == 6:
    num_epochs = sys.argv[5]

if not model_number.isdigit() and model_number != "default" or not lookup_days.isdigit() or not num_epochs.isdigit():
    print("model number must be integer or \"default\", lookup_step, and epochs must be integers")
    exit()

if model_number != "default":
    model_number = int(model_number)
    load_config("configs/hyperparameters/config.json", model_number)

lookup_days = int(lookup_days)

lookup_step = lookup_days
epochs = int(num_epochs)
ticker = stock_ticker

ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
model_name = f"{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}\
    -{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{n_steps}-step-{lookup_step}-layers\
    -{n_layers}-units-{units}--epochs{epochs}"

if bidirectional:
    model_name += "-b"

match program_option:
    case "train":
        train()
    case "predict":
        try:
            test()
        except FileNotFoundError:
            train_and_predict()
    case "forecast":
        all_predictions = []
        for i in range(1, lookup_days + 1):
            lookup_step = i
            model_name = f"{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
                {LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{n_steps}-step-{lookup_step}-layers-\
                {n_layers}-units-{units}--epochs{epochs}"
            try:
                all_predictions.append(test())
            except FileNotFoundError:
                all_predictions.append(train_and_predict())
        print(f"future prices for {ticker} for the next {lookup_step} are {all_predictions}")

    case "analyze":
        train()
        test()
    case _:
        stonk_help()
