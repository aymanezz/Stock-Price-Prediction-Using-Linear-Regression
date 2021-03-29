import quandl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

auth_tok = "jsqVCw-U9zAC-m_sWhj5"
df = quandl.get("WIKI/AAPL", trim_start = "2000-12-12", trim_end = "2014-12-30", authtoken=auth_tok)
df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Close']*100.0
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0
print(df.columns)

'''
Index(['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio',
       'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',
       'HL_PCT', 'PCT_change'],
      dtype='object')
'''

df2 = df[['Adj. Close','PCT_change','HL_PCT','Adj. Volume']]
df2['labels'] = df2['Adj. Close'].shift(periods=-1, fill_value=-99999)

X = df2[['Adj. Close','PCT_change','HL_PCT','Adj. Volume']]
y = df2['labels']
nans = df2.isnull().sum()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)

model = LinearRegression()

model.fit(X_scaled,y_train)

y_pred = model.predict(X_test)

r = mean_squared_error(y_test, y_pred)
print(r)
