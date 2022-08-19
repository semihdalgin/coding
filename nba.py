import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns


np.set_printoptions(suppress=True, precision = 2)

nba = pd.read_csv('./nba_games.csv')

# Subset Data to 2010 Season, 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

print(nba_2010.head())
print(nba_2014.head())

knick_pts_2010=nba_2010.pts[nba_2010.fran_id =='Knicks']
nets_pts_2010= nba_2010.pts[nba_2010.fran_id == 'Nets']

diff_means_2010 = knick_pts_2010.mean() - nets_pts_2010.mean()
print(diff_means_2010)

plt.hist(x=knick_pts_2010, alpha=0.5, label='Knicks')
plt.hist(x=nets_pts_2010, alpha=0.5, label='Nets')
plt.legend()
plt.show()
plt.clf()

knick_pts_2014 = nba_2014.pts[nba_2014.fran_id =='Knicks']
nets_pts_2014 = nba_2014.pts[nba_2014.fran_id == 'Nets']

diff_means_2014 = knick_pts_2014.mean() - nets_pts_2014.mean()
print(diff_means_2014)
plt.hist(x=knick_pts_2014, alpha=0.5, label='Knicks')
plt.hist(x=nets_pts_2014, alpha=0.5, label='Nets')
plt.legend()
plt.show()
plt.clf()

sns.boxplot(data= nba_2010, y = nba_2010.pts, x = nba_2010.fran_id)
plt.show()
plt.clf()

location_result_freq = pd.crosstab(nba.game_result, nba.game_location)

location_result_proportions = location_result_freq / len(nba)
print(location_result_proportions)

chi2, pval, dof, expected = chi2_contingency(location_result_freq)

covn= np.cov(nba_2010.forecast, nba_2010.point_diff)
print(covn)

corrv= pearsonr(nba_2010.forecast, nba_2010.point_diff)

print(corrv)

plt.scatter(x= nba_2010.forecast, y= nba_2010.point_diff)
plt.show()
plt.clf()

