"""
AI powered stock market predictor
Uses machine learning to predict future prices
all based or historical data and patterns
"""

import argparse
import datetime
import json
import os
import random
import sys
import time

import gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu.pick_gpu_lowest_memory())

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
PRED_DIR="predictions/"
GRAPHS_DIR="graphs/"

@dataclass
class Model():
    """
    datastructure for storing model information/meta-information
    """
    # name of the model: used for saving model information
    model_name = ""

    # file used to store ticker historical data
    ticker_data_filename = ""


@dataclass
class Config():
    """
    structure for storing model parameter configs
    """

    # Window size or the sequence length
    # 260 ~ 1 year
    # 130 ~ 6 months
    # 20 ~ 1 month
    n_steps = 60

    # Lookup step, 1 is the next day
    lookup_step = 7

    # number of layers in the model
    n_layers = 2

    # number of  LSTM neurons
    units = 256

    # dropout rate (0.4 = 40%)
    dropout = 0.4

    # whether to use bidirectional RNNs
    bidirectional = True

    # model batch size
    batch_size = 64

    # epochs to train over
    epochs = 500

    # ticker to train for
    ticker = "SPY"

    crush = False

@dataclass
class CrushedData:
    """
    dataclass for holding information on crushed data
    """
    data = pd.DataFrame
    crush_factor = -1

@dataclass
class Results:
    """
    dataclass for holding a series of predictions
    """
    predictions = []

    def print_result(self):
        """print all predictions"""
        for prediction in self.predictions:
            prediction.print_prediction()

    def write_to_csv(self, file):
        """write all predictions to csv file"""
        f = open(file, "w")
        f.write("date,price,error\n")
        for prediction in self.predictions:
            f.write(f"{prediction.date},{prediction.price},{prediction.error}\n")
        return ""

@dataclass
class Prediction:
    """
    class for holding a price prediction for a specific day
    """
    date = None
    price = None
    error = None

    def print_prediction(self):
        """return values of prediction in string format"""
        print(f"predicted price for {self.date}: {self.price}")

# Extra tickers/indeces for training
SP500 = "^GSPC"
DOW = "^DJI"
VIX = "^VIX"
NASDAQ = "^IXIC"
RUSSEL2000 = "^RUT"

#################### PARAMETERS ####################
# Static hyperparameters for models


# whether to scale feature columns & output price as well
SCALE = True
SCALE_STR = f"sc-{int(SCALE)}"

# whether to shuffle the dataset
SHUFFLE = True
SHUFFLE_STR = f"sh-{int(SHUFFLE)}"

# whether to split the training/testing set by date
SPLIT_BY_DATE = False
SPLIT_BY_DATE_STR = f"sbd-{int(SPLIT_BY_DATE)}"

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2

# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

# date now
DATE_NOW = time.strftime("%Y-%m-%d")
last_close_date = None

# LSTM cell
CELL = LSTM

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"

# Other constants
FRIDAY = 4
SATURDAY = 5
SUNDAY = 6

def load_config(config_file, config_id):
    """
    load hyperparameters from config file
    """

    file = open(config_file, "r")
    config = json.load(file)
    hyperparameters = config[str(config_id)]["hyperparameters"]
    n_steps = int(hyperparameters["window"])
    n_layers = int(hyperparameters["layers"])
    units = int(hyperparameters["neurons"])
    dropout = int(hyperparameters["dropout"])
    bidirectional = hyperparameters["bidirectional"]
    batch_size = int(hyperparameters["batch"])
    #ticker = hyperparameters["ticker"]

    config = Config()
    config.n_steps = n_steps
    config.n_layers = n_layers
    config.units = units
    config.dropout = dropout
    config.batch_size = bidirectional
    config.batch_size = batch_size

    return config



#################### TRAINING ####################
# Train a model with given hyperparameters
# using historical data of certain stock

# create these folders if they does not exist
def setup_environment():
    """
    create necessary directories if needed
    """
    if not os.path.isdir("models"):
        os.mkdir("models")

    if not os.path.isdir("logs"):
        os.mkdir("logs")

    if not os.path.isdir("data"):
        os.mkdir("data")

def train(config, model_info):
    """
    train a new model for the given ticker
    """

    # load the data
    data = load_data(config.ticker, n_steps=config.n_steps, scale=SCALE, crush=config.crush,
    split_by_date=SPLIT_BY_DATE,shuffle=SHUFFLE, lookup_step=config.lookup_step,
    test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)

    # save the dataframe
    data["df"].to_csv(model_info.ticker_data_filename)

    # construct the model
    model = create_model(config.n_steps, len(FEATURE_COLUMNS), loss=LOSS,
    units=config.units, cell=CELL, n_layers=config.n_layers, dropout=config.dropout,
    optimizer=OPTIMIZER, bidirectional=config.bidirectional)

    # some tensorflow callbacks
    checkpointer = ModelCheckpoint(os.path.join("models", model_info.model_name + ".h5"),
    save_weights_only=True, save_best_only=True, verbose=1)

    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_info.model_name))
    # train the model and save the weights whenever we see
    # a new optimal model using ModelCheckpoint

    #history =
    model.fit(data["x_train"], data["y_train"],batch_size=config.batch_size,
    epochs=config.epochs,validation_data=(data["x_test"], data["y_test"]),
    callbacks=[checkpointer, tensorboard],verbose=1)



#################### PREDICTION ####################
# Use given data and recent stock prices to predict future prices

def plot_graph(config, test_df):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f'true_adjclose_{config.lookup_step}'], c='b')
    plt.plot(test_df[f'adjclose_{config.lookup_step}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])

    graph_file = os.path.join(GRAPHS_DIR, f"{config.ticker}-{last_close_date}.png")
    plt.savefig(graph_file)


def get_final_df(config, model, data):
    """
    This function takes the `model` and `data` dict to
    construct a final dataframe that includes the features along
    with true and predicted prices of the testing dataset
    """
    # if predicted future price is higher than the current,
    # then calculate the true future price minus the current price, to get the buy profit
    buy_profit  = lambda current, pred_future, true_future: true_future - \
    current if pred_future > current else 0

    # if the predicted future price is lower than the current price,
    # then subtract the true future price from the current price
    sell_profit = lambda current, pred_future, true_future: current - \
    true_future if pred_future < current else 0

    x_test = data["x_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(x_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].
            inverse_transform(np.expand_dims(y_test, axis=0)))

        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))

    test_df = data["test_df"]

    # add predicted future prices to the dataframe
    test_df[f"adjclose_{config.lookup_step}"] = y_pred

    # add true future prices to the dataframe
    test_df[f"true_adjclose_{config.lookup_step}"] = y_test

    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df

    # add the buy profit column
    final_df["buy_profit"] = list(map(buy_profit,
        final_df["adjclose"],
        final_df[f"adjclose_{config.lookup_step}"],
        final_df[f"true_adjclose_{config.lookup_step}"])
        # since we don't have profit for last sequence, add 0's
        )

    # add the sell profit column
    final_df["sell_profit"] = list(map(sell_profit,
        final_df["adjclose"],
        final_df[f"adjclose_{config.lookup_step}"],
        final_df[f"true_adjclose_{config.lookup_step}"])
        # since we don't have profit for last sequence, add 0's
    )

    return final_df


def predict(config, model, data):
    """calculate future price prdictions"""
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-config.n_steps:]
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


def test(config, model_info):
    """predict future prices"""

    # load the data
    data = load_data(config.ticker, config.n_steps, scale=SCALE, split_by_date=SPLIT_BY_DATE,
        shuffle=SHUFFLE, lookup_step=config.lookup_step,
        test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)

    # construct the model
    model = create_model(config.n_steps, len(FEATURE_COLUMNS), loss=LOSS, units=config.units,
    cell=CELL, n_layers=config.n_layers, dropout=config.dropout, optimizer=OPTIMIZER,
    bidirectional=config.bidirectional)

    # load optimal model weights from results folder
    model_path = os.path.join("models", model_info.model_name) + ".h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError

    model.load_weights(model_path)

    # evaluate the model
    loss, mae = model.evaluate(data["x_test"], data["y_test"], verbose=0)
    # calculate the mean absolute error (inverse scaling)
    if SCALE:
        mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
    else:
        mean_absolute_error = mae

    # get the final dataframe for the testing set
    final_df = get_final_df(config, model, data)

    # predict the future price
    future_price = predict(config, model, data)

    if data["crushed"]:
        future_price *= data["crush_factor"]

    # we calculate the accuracy by counting the number of positive profits
    #accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) +\
    #    len(final_df[final_df['buy_profit'] > 0])) / len(final_df)

    # calculating total buy & sell profit
    #total_buy_profit  = final_df["buy_profit"].sum()
    #total_sell_profit = final_df["sell_profit"].sum()

    # total profit by adding sell & buy together
    #total_profit = total_buy_profit + total_sell_profit

    # dividing total profit by number of testing samples (number of trades)
    #profit_per_trade = total_profit / len(final_df)
    # printing metrics
    final_price = f"${future_price:.2f}"
    print(f"Future price of {config.ticker} after {config.lookup_step} days is {final_price}")
    print(f"{LOSS} loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    #print("Accuracy score:", accuracy_score)
    #print("Total buy profit:", total_buy_profit)
    #print("Total sell profit:", total_sell_profit)
    #print("Total profit:", total_profit)
    #print("Profit per trade:", profit_per_trade)
    # plot true/pred prices graph
    #plot_graph(config, final_df)
    #print(final_df.tail(10))
    # save the final dataframe to csv-results folder
    csv_results_folder = "csv-results"
    if not os.path.isdir(csv_results_folder):
        os.mkdir(csv_results_folder)
    csv_filename = os.path.join(csv_results_folder, model_info.model_name + ".csv")
    final_df.to_csv(csv_filename)

    prediction = Prediction()
    prediction.price = final_price
    prediction.date = calculate_prediction_date(config)
    prediction.error = mean_absolute_error
    return prediction



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


def load_data(ticker, n_steps=50, scale=True, crush=False, shuffle=True,
lookup_step=1, split_by_date=True, test_size=0.2,
feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
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

    global last_close_date

    data_frame = None
    crush_factor = -1
    crushed = False

    if ticker == "generic":
        # TODO special handling for generic model
        """
        Download information for market indicator tickers
        Crush all data
        Compute average to simulate overall market trends
        Use crushed data for training
        """
        print("generic model is not currently supported")
    else:
        # see if ticker is already a loaded stock from yahoo finance
        if isinstance(ticker, str):
            # load it from yahoo_fin library
            data_frame = si.get_data(ticker)
        elif isinstance(ticker, pd.DataFrame):
            # already loaded, use it directly
            data_frame = ticker
        else:
            raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    if crush:
        crushed_data = crush_data(data_frame)
        data_frame = crushed_data.data
        crush_factor = crushed_data.crush_factor
        crushed = True

    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = data_frame.copy()

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in data_frame.columns, f"'{col}' does not exist in the dataframe."

    # add date as a column
    if "date" not in data_frame.columns:
        data_frame["date"] = data_frame.index

    if SCALE:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            data_frame[column] = scaler.fit_transform(np.expand_dims(data_frame[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    data_frame['future'] = data_frame['adjclose'].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(data_frame[feature_columns].tail(lookup_step))

    # drop NaNs
    data_frame.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(data_frame[feature_columns + ["date"]].values, data_frame['future'].values):
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
    x, y = [], []
    for seq, target in sequence_data:
        x.append(seq)
        y.append(target)

    # convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(x))
        result["x_train"] = x[:train_samples]
        result["y_train"] = y[:train_samples]
        result["x_test"]  = x[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["x_train"], result["y_train"])
            shuffle_in_unison(result["x_test"], result["y_test"])
    else:
        # split the dataset randomly
        result["x_train"], result["x_test"], result["y_train"], result["y_test"] = \
        train_test_split(x, y, test_size=test_size, shuffle=shuffle)

    # get the list of test set dates
    dates = result["x_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["x_train"] = result["x_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["x_test"] = result["x_test"][:, :, :len(feature_columns)].astype(np.float32)

    result["crush_factor"] = crush_factor
    result["crushed"] = crushed
    return result

def crush_data(data_frame):
    """
    linearly crush given dataset's prices to be between 0 - 100
    returns a crushed data object containing the crush factor and the crushed data
    """
    crushed_data_frame = data_frame.copy(deep=True)

    # Calculate crush factor
    max_price = data_frame["high"].max()
    crush_factor = max_price / 100

    # Crush relevant columns
    crushed_data_frame["low"] = crushed_data_frame["low"].div(crush_factor)
    crushed_data_frame["open"] = crushed_data_frame["open"].div(crush_factor)
    crushed_data_frame["high"] = crushed_data_frame["high"].div(crush_factor)
    crushed_data_frame["close"] = crushed_data_frame["close"].div(crush_factor)
    crushed_data_frame["adjclose"] = crushed_data_frame["adjclose"].div(crush_factor)

    crushed_data = CrushedData()
    crushed_data.data = crushed_data_frame
    crushed_data.crush_factor = crush_factor

    return crushed_data

def expand_data(data_frame, crush_factor):
    """
    linearly expand crushed data back to original bounds using initial crush factor
    returns a dataframe with the expanded values
    """
    expanded_data_frame = data_frame.copy(deep=True)

    # Expand relevant columns
    expanded_data_frame["low"] = expanded_data_frame["low"].multiply(crush_factor)
    expanded_data_frame["open"] = expanded_data_frame["open"].multiply(crush_factor)
    expanded_data_frame["high"] = expanded_data_frame["high"].multiply(crush_factor)
    expanded_data_frame["close"] = expanded_data_frame["close"].multiply(crush_factor)
    expanded_data_frame["adjclose"] = expanded_data_frame["adjclose"].multiply(crush_factor)

    return expanded_data_frame

def test_crush(config):
    """test function to verify crush algorithm"""
    # load it from yahoo_fin library
    data_frame = si.get_data(config.ticker)
    crushed_data = crush_data(data_frame)
    crushed_data_frame = crushed_data.data
    crush_factor = crushed_data.crush_factor
    expanded_data_frame = expand_data(crushed_data_frame, crush_factor)
    print(crushed_data_frame)
    print(crush_factor)
    print(expanded_data_frame)
    print(data_frame)

def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True),
                    batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True,
                    batch_input_shape=(None, sequence_length, n_features)))
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
def run(config, model_info, mode, force):
    """helper function to train/predict and generate results"""
    global DATE_NOW, PRED_DIR

    results = Results()
    total_days = config.lookup_step + 1

    match mode:
        # Train a new model with the give parameters
        case "train":
            train(config, model_info)

        # Use a model to predict future prices
        # Train first if necessary
        case "predict":
            prediction = Prediction()
            if force:
                prediction = train_and_predict(config, model_info)
            else:
                try:
                    prediction = test(config, model_info)
                except FileNotFoundError:
                    prediction = train_and_predict(config, model_info)

            results.predictions.append(prediction)

            #print(f"future prices for {config.ticker} in {config.lookup_step} "\
                #f"day(s) is: {prediction.to_string()}")

        # Use a model to predict all values until given number of days into the future
        # Train as needed
        case "forecast":
            all_predictions = []
            for i in range(1, total_days):
                config.lookup_step = i
                prediction = Prediction()

                model_info.model_name = generate_model_name(config)

                if force:
                    prediction = train_and_predict(config, model_info)
                else:
                    try:
                        prediction = test(config, model_info)
                    except FileNotFoundError:
                        prediction = train_and_predict(config, model_info)

                all_predictions.append(prediction)

            results.predictions = all_predictions

            #print(f"future prices for {config.ticker} for the next {total_days} day(s) are:")
            #for i in range(0, len(all_predictions)):
            #    print(all_predictions[i].to_string())

        case _:
            # TODO switch to throwing an error
            print("supported modes are \"train\", \"predict\", and \"forecast\"")

    print(f"predictions for {config.ticker}:")
    results.print_result()
    csv = f"{config.ticker}-{DATE_NOW}-mode-{mode}-days-{total_days}-epochs-{config.epochs}"
    if config.bidirectional:
        csv += "-b"

    if config.crush:
        csv += "-c"

    csv = os.path.join(PRED_DIR, csv)
    results.write_to_csv(csv)

def train_and_predict(config, model_info):
    """train on historical data and then predict future prices"""
    train(config, model_info)
    return test(config, model_info)

def get_last_close_date():
    """get the last close date from yahoo finance"""

    global last_close_date

    df = si.get_data(SP500)

    # add date as a column
    if "date" not in df.columns:
        df["date"] = df.index

    last_close_date = df["date"].iloc[-1].to_pydatetime()


def calculate_prediction_date(config):
    """
    Get the date for the configured prediction
    Take into account weekends as days market is closed
    """
    global last_close_date

    prediction_date = last_close_date
    for _ in range(0, config.lookup_step):
        time_delta = datetime.timedelta(days = 1)
        prediction_date += time_delta

        prediction_date_weekday = prediction_date.weekday()
        additional_delta = datetime.timedelta(days = 0)
        if prediction_date_weekday == SATURDAY:
            additional_delta = datetime.timedelta(days = 2)
        elif prediction_date_weekday == SUNDAY:
            additional_delta = datetime.timedelta(days = 1)

        prediction_date += additional_delta

    final_prediction_date = prediction_date.strftime("%Y-%m-%d")
    return final_prediction_date

def calculate_lookup_step_from_date(date):
    """calculate the lookup step for the given date"""
    global last_close_date

    target_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    current_date = last_close_date

    if target_date < last_close_date:
        print(f"target date must be after last close_date ({last_close_date})")
        sys.exit()

    total_days = 0
    while current_date < target_date:
        time_delta = datetime.timedelta(days = 1)
        current_date += time_delta

        current_date_weekday = current_date.weekday()
        additional_delta = datetime.timedelta(days = 0)
        if current_date_weekday == FRIDAY:
            additional_delta = datetime.timedelta(days = 2)
        elif current_date_weekday == SATURDAY:
            additional_delta = datetime.timedelta(days = 1)

        current_date += additional_delta
        total_days += 1


    return total_days

def parse_ags(parser):
    """setup parser for program arguments"""
    date_group = parser.add_mutually_exclusive_group()

    parser.add_argument("-m", "--mode",
        required=True,
        dest="mode",
        help="""stonk engine mode (train, predict, forecast, analyze)
            train: only create a new model that can be used for future predictions
            predict: use an existing model for predictions, will train new model if one is not found
            forecast: perform prediction from 1-n days in the future where n = time
        """,
        type=str
        )

    parser.add_argument("--ticker",
        required=True,
        dest="ticker",
        help="ticker to train/predict for",
        type=str
        )

    parser.add_argument("--config",
        dest="config",
        default="",
        help="path to config file to load from",
        type=str
    )

    parser.add_argument("--epochs",
        dest="epochs",
        default=500,
        help="number of epochs to train for. Default: 500",
        type=int
    )

    date_group.add_argument("--days",
        dest="lookup_step",
        default=1,
        help="number of days in the future to predict for Default: 1.",
        type=int
    )

    date_group.add_argument("--date",
        dest="target_date",
        default="",
        help="future date to (yyyy-mm-dd) predict for.",
        type=str
    )

    parser.add_argument("--bidirectional",
        dest="bidirectional",
        action="store_true",
        help="use bidirectional RNN or not",
    )

    parser.add_argument("-f", "--force",
        dest="force",
        action="store_true",
        help="force retraining of model",
    )

    parser.add_argument("-c", "--crush",
        dest="crush",
        action="store_true",
        help="use crush algorithm for training and prediction"
    )

    parser.add_argument("--crush-test",
        dest="crush_test",
        action="store_true",
        help="test crush algorithm"
    )

    args = parser.parse_args()
    return args

def generate_model_name(config):
    """helper function to generate correct name for the model"""
    name = f"{config.ticker}-{SHUFFLE_STR}-{SCALE_STR}-{SPLIT_BY_DATE_STR}-"\
        f"{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{config.n_steps}-step-"\
        f"{config.lookup_step}-layers-{config.n_layers}-units-"\
        f"{config.units}-epochs-{config.epochs}"

    if config.bidirectional:
        name += "-b"

    if config.crush:
        name += "-c"

    return name

def main():
    """parse args and setup data structures for running in various modes"""
    global DATE_NOW

    parser = argparse.ArgumentParser(description="stonk engine for predicting stock prices")
    args = parse_ags(parser)

    get_last_close_date()

    config = Config()
    model_info = Model()

    initial_lookup_step = 0

    if args.lookup_step == -1:
        if args.target_date == "":
            parser.print_help()
            sys.exit()

        initial_lookup_step = calculate_lookup_step_from_date(args.target_date)
    else:
        initial_lookup_step = args.lookup_step

    config.lookup_step = initial_lookup_step
    config.ticker = args.ticker
    config.epochs = args.epochs
    config.bidirectional = args.bidirectional
    config.crush = args.crush

    if args.crush_test:
        test_crush(config)
        sys.exit()

    ticker_data_filename = os.path.join("data", f"{config.ticker}_{DATE_NOW}.csv")

    model_info.ticker_data_filename = ticker_data_filename
    model_info.model_name = generate_model_name(config)

    run(config, model_info, args.mode, args.force)

if __name__ == "__main__":
    main()
