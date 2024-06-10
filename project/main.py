import pandas as pd
from datetime import date
import yfinance as yf
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import subprocess
import os
import spacy
from prophet.diagnostics import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

def compute_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    percent_mae_value = (mae / actual.mean()) * 100
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    bias_value = (predicted - actual).mean()
    return mae, percent_mae_value, rmse, bias_value

mae, percent_mae_value, rmse, bias_value = compute_metrics(forex_test_1['y'], forex_test_1['yhat'])
print(f"Model Evaluation Metrics:\n")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Percent Mean Absolute Error (%MAE): {percent_mae_value:.2f}%")
print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
print(f"Bias: {bias_value:.4f}")

# Forecast future values using rolling mean with a window size of 7
forex['yhat_rm'] = forex['eurusd_close'].rolling(7).mean()

forex_test_rm = forex[train_end:] # Extract the test set

mae_rm, percent_mae_value_rm, rmse_rm, bias_value_rm = compute_metrics(forex_test_rm['eurusd_close'], forex_test_rm['yhat_rm'])
print(f"Model Evaluation Metrics:\n")
print(f"Mean Absolute Error (MAE): {mae_rm:.4f}")
print(f"Percent Mean Absolute Error (%MAE): {percent_mae_value_rm:.2f}%")
print(f"Root Mean Square Error (RMSE): {rmse_rm:.4f}")
print(f"Bias: {bias_value_rm:.4f}")

# Train a model for sentiment prediction
#subprocess.run('git clone https://github.com/explosion/projects.git spacy-projects', shell=True)

text_file = open('D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/FinancialPhraseBank_AllAgree.txt', 'r') # Prepare the training set
financial = text_file.read().split('\n')
financial = pd.DataFrame(financial, columns=['text'])
financial.tail()

financial[['text', 'sentiment']] = financial['text'].str.split('@', expand=True)
financial = financial.iloc[:2264, :] # Remove the last row, since it's empty
financial.head()

financial['sentiment_id'] = financial['sentiment'].apply(lambda x: 0 if x == 'negative' else (1 if x == 'neutral' else 2))
financial['text_id'] = "T" + financial.index.astype(str)
financial = financial[['text', 'sentiment_id', 'text_id']]
financial.head()

financial_1 = financial.sample(frac=1) # Shuffle the data

train_ratio_s = 0.7

total_rows_s = financial_1.shape[0]
train_end_s = int(total_rows_s*train_ratio_s)

financial_train = financial_1[:train_end_s] # Training set
remaining_s = financial_1[train_end_s:]

test_ratio_s = 2/3

remaining_rows_s = remaining_s.shape[0]
test_end_s = int(remaining_rows_s*test_ratio_s)

financial_test = remaining_s[:test_end_s] # Test set
financial_dev = remaining_s[test_end_s:] # Development set

path = 'D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/news/assets/'

financial_train.to_csv(path + 'train.tsv', sep='\t', index=False, header=False)
financial_test.to_csv(path + 'test.tsv', sep='\t', index=False, header=False)
financial_dev.to_csv(path + 'dev.tsv', sep='\t', index=False, header=False)

os.chdir('D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/news')
#subprocess.run('spacy project run preprocess', shell=True)
#subprocess.run('spacy project run train', shell=True)
#subprocess.run('spacy project run evaluate', shell=True)

# Predict the sentiment of scraped news
nlp_s = spacy.load("./training/cnn/model-best")
sentiments = []
for doc in nlp_s.pipe(news['txt']):
    sentiments.append(doc.cats)

news['sentiment'] = sentiments

def get_max_key(d):
    return max(d, key=d.get)

news['sentiment'] = news['sentiment'].apply(get_max_key)
news.head()

# Include the sentiment as an additional regressor in the first model
forex_train_s = forex_train.merge(news, left_on='ds', right_on='date')
forex_train_s['sentiment'] = forex_train_s['sentiment'].apply(lambda x: 0 if x == 'negative' else (1 if x == 'neutral' else 2))
forex_train_s = forex_train_s[['ds', 'y', 'sentiment']]
forex_train_s.head()

forex_test_s = forex_test.merge(news, left_on='ds', right_on='date')
forex_test_s['sentiment'] = forex_test_s['sentiment'].apply(lambda x: 0 if x == 'negative' else (1 if x == 'neutral' else 2))
forex_test_s = forex_test_s[['ds', 'y', 'sentiment']]
forex_test_s.head()

forex_s = pd.concat([forex_train_s, forex_test_s])
forex_s.shape[0]

m_s = Prophet()
m_s.add_regressor('sentiment')
m_s.fit(forex_test_s)

forex_test_s_1 = m_s.predict(forex_test_s[['ds', 'sentiment']])[['ds', 'yhat_lower', 'yhat_upper', 'yhat']] # Compute predictions for the test set
forex_test_s_1 = forex_test_s_1.merge(forex_test_s, left_on='ds', right_on='ds')
forex_test_s_1.tail()

mae_s, percent_mae_value_s, rmse_s, bias_value_s = compute_metrics(forex_test_s_1['y'], forex_test_s_1['yhat'])
print(f"Model Evaluation Metrics:\n")
print(f"Mean Absolute Error (MAE): {mae_s:.4f}")
print(f"Percent Mean Absolute Error (%MAE): {percent_mae_value_s:.2f}%")
print(f"Root Mean Square Error (RMSE): {rmse_s:.4f}")
print(f"Bias: {bias_value_s:.4f}")

# Perform cross-validation
cut_offs = pd.DataFrame(forex_s[(forex_s['ds'] >= '2024-01-01') & (forex_s['ds'] <= '2024-05-20')]['ds'])

date_strings = [timestamp.strftime('%Y-%m-%d') for timestamp in cut_offs['ds'].to_list()] # Fix dates here
date_strings

m_s_cv = cross_validation(m_s, cutoffs = pd.to_datetime(date_strings), period = '1 days', horizon = '14 days')
m_s_cv

# Is the model able to predict y increase/decrease 14 days from the cutoff date?
m_s_cv['days_difference'] = m_s_cv['ds'] - m_s_cv['cutoff']
m_s_cv['days_difference'] = m_s_cv['days_difference'].dt.days
m_s_cv.head()

m_s_cv_selected = m_s_cv[m_s_cv['days_difference'] == 14][['ds', 'yhat', 'y', 'cutoff']]
m_s_cv_selected = m_s_cv_selected.rename(columns={'y': 'y_ds'}).merge(forex_s[['ds', 'y']], how='left', left_on='cutoff', right_on='ds').drop("ds_y", axis= 1).rename(columns={'ds_x': 'ds', 'y': 'y_cutoff'})
m_s_cv_selected['delta_y_cutoff_to_ds'] = m_s_cv_selected['y_ds'] - m_s_cv_selected['y_cutoff']
m_s_cv_selected['delta_y_cutoff_to_yhat'] = m_s_cv_selected['yhat'] - m_s_cv_selected['y_cutoff']

m_s_cv_selected['delta_y_cutoff_to_ds'] = m_s_cv_selected['delta_y_cutoff_to_ds'].apply(lambda x: 1 if x>0 else 0)
m_s_cv_selected['delta_y_cutoff_to_yhat'] = m_s_cv_selected['delta_y_cutoff_to_yhat'].apply(lambda x: 1 if x>0 else 0)
m_s_cv_selected

acc_s = accuracy_score(m_s_cv_selected['delta_y_cutoff_to_ds'], m_s_cv_selected['delta_y_cutoff_to_yhat'])
print(f"Accuracy: {acc_s:.4f}")

conf_mat_s = confusion_matrix(m_s_cv_selected['delta_y_cutoff_to_ds'], m_s_cv_selected['delta_y_cutoff_to_yhat'])
ConfusionMatrixDisplay(conf_mat_s).plot()
plt.show()

# Include the COVID-19 vaccination rate as an additional regressor in the first model
vaccinations = pd.read_csv('D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/vaccinations.csv')
vaccinations.head()

vaccinations = vaccinations[['iso_code', 'date', 'total_vaccinations_per_hundred']]

european_countries_alpha3 = [
    "ALB", "AND", "ARM", "AUT", "AZE", "BLR", "BEL", "BIH", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN",
    "FRA", "GEO", "DEU", "GRC", "HUN", "ISL", "IRL", "ITA", "KAZ", "XKX", "LVA", "LIE", "LTU", "LUX", "MLT",
    "MDA", "MCO", "MNE", "NLD", "MKD", "NOR", "POL", "PRT", "ROU", "RUS", "SMR", "SRB", "SVK", "SVN", "ESP",
    "SWE", "CHE", "TUR", "UKR", "GBR", "VAT"
]

north_american_countries_alpha3 = [
    "ATG", "BHS", "BRB", "BLZ", "CAN", "CRI", "CUB", "DMA", "DOM", "SLV", "GRD", "GTM", "HTI", "HND", "JAM",
    "MEX", "NIC", "PAN", "KNA", "LCA", "VCT", "TTO", "USA"
]

countries = european_countries_alpha3 + north_american_countries_alpha3

vaccinations = vaccinations[vaccinations['iso_code'].isin(countries)] # Filter for European and North American countries
vaccinations = vaccinations.groupby('date')['total_vaccinations_per_hundred'].mean().reset_index().rename(columns={'total_vaccinations_per_hundred': 'vaccination_rate'}) # Group by date and compute mean vaccination rate
vaccinations['date'] = pd.to_datetime(vaccinations['date'])
vaccinations.head()

forex_train_v = forex_train_s.merge(vaccinations, left_on='ds', right_on='date')
forex_train_v = forex_train_v[['ds', 'y', 'sentiment', 'vaccination_rate']]
forex_train_v.head()

forex_test_v = forex_test_s.merge(vaccinations, left_on='ds', right_on='date')
forex_test_v = forex_test_v[['ds', 'y', 'sentiment', 'vaccination_rate']]
forex_test_v.head()

forex_v = pd.concat([forex_train_v, forex_test_v])
forex_v.shape[0]

m_v = Prophet()
m_v.add_regressor('sentiment')
m_v.add_regressor('vaccination_rate')
m_v.fit(forex_test_v)

forex_test_v_1 = m_v.predict(forex_test_v[['ds', 'sentiment', 'vaccination_rate']])[['ds', 'yhat_lower', 'yhat_upper', 'yhat']]
forex_test_v_1 = forex_test_v_1.merge(forex_test_v, left_on='ds', right_on='ds')
forex_test_v_1.tail()

mae_v, percent_mae_value_v, rmse_v, bias_value_v = compute_metrics(forex_test_v_1['y'], forex_test_v_1['yhat'])
print(f"Model Evaluation Metrics:\n")
print(f"Mean Absolute Error (MAE): {mae_v:.4f}")
print(f"Percent Mean Absolute Error (%MAE): {percent_mae_value_v:.2f}%")
print(f"Root Mean Square Error (RMSE): {rmse_v:.4f}")
print(f"Bias: {bias_value_v:.4f}")

# Perform cross-validation
m_v_cv = cross_validation(m_v, cutoffs = pd.to_datetime(date_strings), period = '1 days', horizon = '14 days')
m_v_cv

# Is the model able to predict y increase/decrease 14 days from the cutoff date?
m_v_cv['days_difference'] = m_v_cv['ds'] - m_v_cv['cutoff']
m_v_cv['days_difference'] = m_v_cv['days_difference'].dt.days
m_v_cv.head()

m_v_cv_selected = m_v_cv[m_v_cv['days_difference'] == 14][['ds', 'yhat', 'y', 'cutoff']]
m_v_cv_selected = m_v_cv_selected.rename(columns={'y': 'y_ds'}).merge(forex_s[['ds', 'y']], how='left', left_on='cutoff', right_on='ds').drop("ds_y", axis= 1).rename(columns={'ds_x': 'ds', 'y': 'y_cutoff'})
m_v_cv_selected['delta_y_cutoff_to_ds'] = m_v_cv_selected['y_ds'] - m_v_cv_selected['y_cutoff']
m_v_cv_selected['delta_y_cutoff_to_yhat'] = m_v_cv_selected['yhat'] - m_v_cv_selected['y_cutoff']

m_v_cv_selected['delta_y_cutoff_to_ds'] = m_v_cv_selected['delta_y_cutoff_to_ds'].apply(lambda x: 1 if x>0 else 0)
m_v_cv_selected['delta_y_cutoff_to_yhat'] = m_v_cv_selected['delta_y_cutoff_to_yhat'].apply(lambda x: 1 if x>0 else 0)
m_v_cv_selected

acc_v = accuracy_score(m_v_cv_selected['delta_y_cutoff_to_ds'], m_v_cv_selected['delta_y_cutoff_to_yhat'])
print(f"Accuracy: {acc_v:.4f}")

conf_mat_v = confusion_matrix(m_v_cv_selected['delta_y_cutoff_to_ds'], m_v_cv_selected['delta_y_cutoff_to_yhat'])
ConfusionMatrixDisplay(conf_mat_v).plot()
plt.show()

# Train a model for binary classification: y increase/decrease 5 days from now
forex_b = forex[['date', 'eurusd_close']].rename(columns={'date': 'ds', 'eurusd_close': 'y'})
forex_b.head()

forex_b['y_week_future'] = forex_b['y'].shift(-5)  # Calculate the difference of y
forex_b['y_difference_week'] = forex_b['y_week_future'] - forex_b['y']
forex_b = forex_b.dropna()
forex_b['y_difference'] = forex_b['y_difference_week'].apply(lambda x: 1 if x>0 else 0).astype(int)
forex_b.head()

news_b = news.merge(forex_b[['ds', 'y_difference']], left_on='date', right_on='ds')
news_b = news_b[['txt', 'y_difference']]
news_b['text_id'] = "T" + news_b.index.astype(str)
news_b.head()

news_b_1 = news_b.sample(frac=1) # Shuffle the data

train_ratio_b = 0.7
 
total_rows_b = news_b_1.shape[0]
train_end_b = int(total_rows_b*train_ratio_b)

train_b = news_b_1[:train_end_b] # Training set
remaining_b = news_b_1[train_end_b:]

test_ratio_b = 2/3

remaining_rows_b = remaining_b.shape[0]
test_end_b = int(remaining_rows_b*test_ratio_b)

test_b = remaining_b[:test_end_b] # Test set
dev_b = remaining_b[test_end_b:] # Development set

path = 'D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/news-1/assets/'

train_b.to_csv(path + 'train.tsv', sep='\t', index=False, header=False)
test_b.to_csv(path + 'test.tsv', sep='\t', index=False, header=False)
dev_b.to_csv(path + 'dev.tsv', sep='\t', index=False, header=False)

os.chdir('D:/Studies/Materials/Second-cycle/I year/III trimester/Ammagamma-Lab/ammagamma-lab/project/news-1')
subprocess.run('spacy project run preprocess', shell=True)
subprocess.run('spacy project run train', shell=True)
subprocess.run('spacy project run evaluate', shell=True)