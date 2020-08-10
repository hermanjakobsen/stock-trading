from alpha_vantage.timeseries import TimeSeries
import json
import pandas as pd
from sklearn import preprocessing
import numpy as np

HistoryPoints = 50

def fetchAndSaveCsvData(ticker):
    credentials = json.load(open('credentials.json','r'))
    apiKey = credentials['avApiKey']

    timeSeries = TimeSeries(key=apiKey, output_format='pandas')
    data, _, = timeSeries.get_daily(ticker, outputsize='full')

    data.to_csv(f'./{ticker}_daily.csv')


def reverseDataFrame(df):
    return df[::-1].reset_index(drop=True)


def filterDataFrame(df):
    df = reverseDataFrame(df)
    df = df.drop('date', axis=1)
    df = df.drop(0, axis=0)
    return df


def getTrainAndTestDataSplit(data, trainRows):
    train = data[:trainRows]
    test = data[trainRows:]
    return train, test


def normaliseData(data, isTrain=False):
    normaliser = preprocessing.MinMaxScaler()
    if (isTrain):
        dataShape = data.shape
        flattenedData = data.reshape(-1, data.shape[-1])
        normalisedFlattenedData = normaliser.fit_transform(flattenedData)
        return normalisedFlattenedData.reshape(dataShape[0], dataShape[1], dataShape[2])

    return normaliser.fit_transform(data)
    

def getOhlcvHistories(data):
    return np.array([data[i:i+HistoryPoints] for i in range(len(data) - HistoryPoints)])


def getNextDayOpenValues(data):
    return np.array([data[:,0][i+HistoryPoints] for i in range(len(data) - HistoryPoints)]) 


def csvToDataset(csvPath):
    df = filterDataFrame(pd.read_csv(csvPath))
    data = df.to_numpy()
    
    ohlcvHistories = getOhlcvHistories(data)
    nextDayOpenValues = getNextDayOpenValues(data).reshape(-1, 1)

    assert ohlcvHistories.shape[0] == nextDayOpenValues.shape[0]

    return ohlcvHistories, nextDayOpenValues


def getTrainAndTestData(csvPath, trainSplitPercentage):

    x, y = csvToDataset(csvPath)

    trainRows = int(x.shape[0] * trainSplitPercentage)
    
    xNormalised = normaliseData(x, True)
    yNormalised = normaliseData(y)

    xTrain = xNormalised[:trainRows]
    yTrain = yNormalised[:trainRows]

    xTest = xNormalised[trainRows:]
    yTest = yNormalised[trainRows:]

    yTestUnscaled = y[trainRows:]

    return xTrain, yTrain, xTest, yTest, yTestUnscaled

