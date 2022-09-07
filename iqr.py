import pandas as pd
import numpy as np
from scipy.stats import iqr

df=pd.read_csv('avocado.csv')

print(df.head())

#Q3 = np.quantile(df['AveragePrice'], 0.75)
#Q1 = np.quantile(df['AveragePrice'], 0.25)

#iqr_price=Q3 - Q1

iqr_price = iqr(df['AveragePrice'])

print(iqr_price)

iqr_vol = iqr(df['Total Volume'])
print(iqr_vol)