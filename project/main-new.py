import pandas as pd
from datetime import date
import yfinance as yf
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import subprocess
import os
import spacy

def load_news_data(file_path):
    news = pd.read_csv(file_path)
    news = news[['txt', 'date']].drop_duplicates()
    news['txt'] = news['txt'].str.replace('\n', '').str.replace('\t', '')
    news['date'] = pd.to_datetime(news['date'].str.split().apply(lambda x: x[0]).apply(invert_date))
    news = news[news['date'] >= '2020-01-01']
    return news

def load_forex_data(ticker, start_date):
    end_date = date.today()
    forex_data = yf.download(ticker, start_date, end_date)['Close']
    forex_data = forex_data.rename(ticker.lower().split('=', 1)[0] + '_close').to_frame().reset_index().rename(columns={'Date': 'date'})
    forex_data = forex_data[forex_data['date'] >= '2020-01-01']
    return forex_data

def compute_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    percent_mae_value = (mae / actual.mean()) * 100
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    bias_value = (predicted - actual).mean()
    return mae, percent_mae_value, rmse, bias_value

def preprocess_news(news_path, save_path):
    text_file = open(news_path, 'r')
    news_train = text_file.read().split('\n')
    news_train = pd.DataFrame(news_train, columns=['text'])
    news_train[['text', 'sentiment']] = news_train['text'].str.split('@', expand=True)
    news_train = news_train.iloc[:2264, :] # Remove the last row, since it's empty
    news_train['sentiment_id'] = news_train['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
    news_train['text_id'] = "T" + news_train.index.astype(str)
    news_train.to_csv(save_path, sep='\t', index=False, header=False)

def train_model(model_path):
    os.chdir(model_path)
    subprocess.run('spacy project run preprocess', shell=True)
    subprocess.run('spacy project run train', shell=True)
    subprocess.run('spacy project run evaluate', shell=True)

def invert_date(date_string):
    day, month, year = date_string.split("-")
    return f"{year}-{month}-{day}"

# Load news data
news_path = 'D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/news.csv'
news = load_news_data(news_path)

# Load forex data
ticker = 'EURUSD=X'
start_date = pd.to_datetime('2010-01-01')
forex_data = load_forex_data(ticker, start_date)

# Check for missing values in forex data
print("There are no missing values." if not forex_data.isnull().any().any() else "There are missing values in the dataframe.")

# Forecast future values based on past values
train_ratio = 0.7
total_rows = forex_data.shape[0]
train_end = int(total_rows * train_ratio)
forex_train = forex_data[:train_end]
forex_test = forex_data[train_end:]

m = Prophet()
m.fit(forex_train)

forex_test_1 = m.predict(forex_test[['ds']])[['ds', 'yhat_lower', 'yhat_upper', 'yhat']] # Compute predictions for the test set
forex_test_1 = forex_test_1.merge(forex_test, left_on='ds', right_on='ds')

actual = forex_test_1['y']
predicted = forex_test_1['yhat']
mae, percent_mae_value, rmse, bias_value = compute_metrics(actual, predicted)
print(f"Model Evaluation Metrics:\n")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Percent Mean Absolute Error (%MAE): {percent_mae_value:.2f}%")
print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
print(f"Bias: {bias_value:.4f}")

# Forecast future values using rolling mean with a window size of 7
forex_data['yhat_bm'] = forex_data['eurusd_close'].rolling(7).mean()
forex_test_2 = forex_data[train_end:]

actual_2 = forex_test_2['eurusd_close']
predicted_2 = forex_test_2['yhat_bm']
mae_2, percent_mae_value_2, rmse_2, bias_value_2 = compute_metrics(actual_2, predicted_2)
print(f"Model Evaluation Metrics:\n")
print(f"Mean Absolute Error (MAE): {mae_2:.4f}")
print(f"Percent Mean Absolute Error (%MAE): {percent_mae_value_2:.2f}%")
print(f"Root Mean Square Error (RMSE): {rmse_2:.4f}")
print(f"Bias: {bias_value_2:.4f}")

# Train a model for sentiment prediction
preprocess_news_path = 'D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/FinancialPhraseBank_AllAgree.txt'
save_path = 'D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/news/assets/'
preprocess_news(preprocess_news_path, save_path)
train_model_path = 'D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/news'
train_model(train_model_path)

# Predict the sentiment of scraped news
nlp = spacy.load("./training/cnn/model-best")
cats = []
for doc in nlp.pipe(news['txt']):
    cats.append(doc.cats)

news['sentiment'] = cats
news['sentiment'] = news['sentiment'].apply(get_max_key)

# Include the sentiment as an additional regressor in the first model
forex_train_1 = forex_train.merge(news, left_on='ds', right_on='date')
forex_train_1['sentiment'] = forex_train_1['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
forex_train_1 = forex_train_1[['ds', 'y', 'sentiment']]

forex_test_1 = forex_test.merge(news, left_on='ds', right_on='date')
forex_test_1['sentiment'] = forex_test_1['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
forex_test_1 = forex_test_1[['ds', 'y', 'sentiment']]

m_1 = Prophet()
m_1.add_regressor('sentiment')
m_1.fit(forex_test_1)

forex_test_2 = m_1.predict(forex_test_1[['ds', 'sentiment']])[['ds', 'yhat_lower', 'yhat_upper', 'yhat']] # Compute predictions for the test set
forex_test_2 = forex_test_2.merge(forex_test_1, left_on='ds', right_on='ds')

actual_3 = forex_test_2['y']
predicted_3 = forex_test_2['yhat']

mae_3, percent_mae_value_3, rmse_3, bias_value_3 = compute_metrics(actual_3, predicted_3)
print(f"Model Evaluation Metrics:\n")
print(f"Mean Absolute Error (MAE): {mae_3:.4f}")
print(f"Percent Mean Absolute Error (%MAE): {percent_mae_value_3:.2f}%")
print(f"Root Mean Square Error (RMSE): {rmse_3:.4f}")
print(f"Bias: {bias_value_3:.4f}")

# Train a model for binary classification: y increase/decrease 5 days from now
forex_data['y_week_future'] = forex_data['y'].shift(-5) # Calculate the difference of y
forex_data['y_difference_week'] = forex_data['y_week_future'] - forex_data['y']
forex_data = forex_data.dropna()
forex_data['y_difference'] = forex_data['y_difference_week'].apply(lambda x: 1 if x > 0 else 0).astype(int)

news_1 = news.merge(forex_data[['ds', 'y_difference']], left_on='date', right_on='ds')
news_1 = news_1[['txt', 'y_difference']]
news_1['text_id'] = "T" + news_1.index.astype(str)

news_2 = news_1.sample(frac=1) # Shuffle the data

train_ratio = 0.7
total_rows = news_2.shape[0]
train_end = int(total_rows * train_ratio)

train = news_2[
] # Training set
remaining = news_2[train_end:]

test_ratio = 2/3

remaining_rows = remaining.shape[0]
test_end = int(remaining_rows * test_ratio)

test = remaining[
] # Test set
dev = remaining[test_end:] # Development set

path = 'D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/news-1/assets/'

train.to_csv(path + 'train.tsv', sep='\t', index=False, header=False)
test.to_csv(path + 'test.tsv', sep='\t', index=False, header=False)
dev.to_csv(path + 'dev.tsv', sep='\t', index=False, header=False)

os.chdir('D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/news-1')
train_model(train_model_path)