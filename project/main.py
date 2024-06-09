import pandas as pd
from datetime import date
import yfinance as yf
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import statsmodels.api as sm
import subprocess

# News data
news = pd.read_csv('D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/news.csv')
news.head()

news.columns
news.shape[0]

news = news[['txt', 'date']].drop_duplicates()
news.shape[0]

news['txt'] = news['txt'].str.replace('\n', '').str.replace('\t', '')

def invert_date(date_string):
    day, month, year = date_string.split("-")
    return f"{year}-{month}-{day}"

news['date'] = pd.to_datetime(news['date'].str.split().apply(lambda x: x[0]).apply(invert_date))
news.dtypes

news = news[news['date'] >= '2020-01-01']
news.tail()

# Forex data
start = pd.to_datetime('2010-01-01')
end = date.today()

ticker = 'EURUSD=X'
forex = yf.download(ticker, start, end)['Close']
forex = forex.rename(ticker.lower().split('=', 1)[0] + '_close').to_frame().reset_index().rename(columns={'Date': 'date'})

forex = forex[forex['date'] >= '2020-01-01']
forex.head()

print("There are no missing values." if not forex.isnull().any().any()\
      else "There are missing values in the dataframe.")

# Forecast future values based on past values
forex_1 = forex.rename(columns={'date': 'ds', 'eurusd_close': 'y'})

train_ratio = 0.7 # Create training and test sets
 
total_rows = forex_1.shape[0]
train_end = int(total_rows*train_ratio)

forex_train = forex_1[:train_end]
forex_test = forex_1[train_end:]

m = Prophet()
m.fit(forex_train)

forex_test_1 = m.predict(forex_test[['ds']])[['ds', 'yhat_lower', 'yhat_upper', 'yhat']] # Compute predictions for the test set
forex_test_1 = forex_test_1.merge(forex_test, left_on='ds', right_on='ds')
forex_test_1.tail()

def percent_mae(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    return (mae / actual.mean()) * 100

def bias(actual, predicted):
    return (predicted - actual).mean()

actual = forex_test_1['y']
predicted = forex_test_1['yhat']

mae = mean_absolute_error(actual, predicted) # Evaluate the model
percent_mae_value = percent_mae(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
bias_value = bias(actual, predicted)

print(f"Model Evaluation Metrics:\n")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Percent Mean Absolute Error (%MAE): {percent_mae_value:.2f}%")
print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
print(f"Bias: {bias_value:.4f}")

# Forecast future values using rolling mean with a window size of 7
forex['yhat_bm'] = forex['eurusd_close'].rolling(7).mean()

forex_test_2 = forex[train_end:] # Create the test set again

actual_2 = forex_test_2['eurusd_close']
predicted_2 = forex_test_2['yhat_bm']

mae_2 = mean_absolute_error(actual_2, predicted_2) # Evaluate the model
percent_mae_value_2 = percent_mae(actual_2, predicted_2)
rmse_2 = np.sqrt(mean_squared_error(actual_2, predicted_2))
bias_value_2 = bias(actual_2, predicted_2)

print(f"Model Evaluation Metrics:\n")
print(f"Mean Absolute Error (MAE): {mae_2:.4f}")
print(f"Percent Mean Absolute Error (%MAE): {percent_mae_value_2:.2f}%")
print(f"Root Mean Square Error (RMSE): {rmse_2:.4f}")
print(f"Bias: {bias_value_2:.4f}")

# Train a model for sentiment prediction
#subprocess.run('git clone https://github.com/explosion/projects.git spacy-projects', shell=True)

text_file = open('D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/FinancialPhraseBank_AllAgree.txt','r') # Prepare the training set
news_train = text_file.read().split('\n')
news_train = pd.DataFrame(news_train, columns=['text'])
news_train.tail()

news_train[['text', 'sentiment']] = news_train['text'].str.split('@', expand=True)
news_train = news_train.iloc[:2264, :] # Remove the last row, since it's empty
news_train.head()

news_train['sentiment_id'] = news_train['sentiment'].apply(lambda x: 0 if x == 'negative' else (1 if x == 'neutral' else 2))
news_train['text_id'] = "T" + news_train.index.astype(str)
news_train = news_train[['text', 'sentiment_id', 'text_id']]
news_train.head()

news_train_1 = news_train.sample(frac=1) # Shuffle the data

train_ratio_1 = 0.7
 
total_rows_1 = news_train_1.shape[0]
train_end_1 = int(total_rows_1*train_ratio_1)

train_part = news_train_1[:train_end] # Training set
remaining_part = news_train_1[train_end:]

test_ratio = 2/3

remaining_rows = remaining_part.shape[0]
test_end = int(remaining_rows*test_ratio)

test_part = remaining_part[:test_end] # Test set
dev_part = remaining_part[test_end:] # Development set

path = 'D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/news/assets/'

train_part.to_csv(path + 'train.tsv', sep='\t', index=False, header=False)
test_part.to_csv(path + 'test.tsv', sep='\t', index=False, header=False)
dev_part.to_csv(path + 'dev.tsv', sep='\t', index=False, header=False)

subprocess.run('git clone https://github.com/explosion/projects.git spacy-projects', shell=True)