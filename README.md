# bitcoin prediction

![bitcoin closing prices](notebooks/img/bitcoin.png)

Included in this repo are a few, simple models for bitcoin closing price prediction.
They use a neural network to predict the prices, specifically, an LSTM.

Included are a notebook with some background information and some slightly more polished code, that is pretty much ready to be used in less gimmicky environments than a notebook.

## Setup

### Install all the requirements

`pip install -r requirements.txt`

## What should I check out first

A good starting point would be to check out the 01-oo-mae-1-15 notebook.

It is hopefully well enough documented to get you going.

## Models

### MAE-01/15

This is the model used in the notebook. It uses mean absolute error as loss function and the last 15 datapoints as feature.

### MAE-01

This model is a simplified but more effective version of the one we build in the notebook. It can only predict one datapoint ahead, but does so with a pretty decent precision.

You can actually use this model for more serious cases than the one in the notebook. The path to the dataset is currently hardcoded, but that is easily changed.

Meaning, you can easily update and retrain the neural network to stay current!

#### visualization

`make mae-1.visualize`

#### predict

`make mae-1.predict`

This uses the last entry from the test set to predict the next closing price.

If you would like to provide the last closing price yourself and see what the LSTM will predict, use this:

`make predict`

This script is probably going to be the most useful!

#### train

`make mae-1.train`
