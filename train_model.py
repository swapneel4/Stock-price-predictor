import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def train_model():

    # read csv
    data = pd.read_csv("stock_data.csv")

    # keep only numeric close column
    data = data[['Close']]

    # convert to numeric (removes any string errors)
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

    # remove empty rows
    data = data.dropna()

    # create prediction column
    data['Prediction'] = data['Close'].shift(-30)

    # remove last rows with NaN
    data = data.dropna()

    X = np.array(data[['Close']])
    y = np.array(data['Prediction'])

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()

    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)

    return model, accuracy