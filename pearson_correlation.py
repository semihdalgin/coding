import codecademylib3
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


df = pd.read_csv("avocado.csv")

plt.scatter(df['AveragePrice'], df['Total Volume'])
plt.xlabel('Average Daily Price')
plt.ylabel('Total Daily Sales')
plt.show()

# calculate corr_sqfeet_beds and print it out:
corr_price_vol, p = pearsonr(df['AveragePrice'], df['Total Volume'])
print(corr_price_vol)
