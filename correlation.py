import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

np.set_printoptions(suppress=True, precision = 1)

penguins = pd.read_csv('penguins.csv')

print(penguins.head())

plt.scatter(penguins.flipper_length_mm, penguins.body_mass_g)
plt.show()
covv, p= pearsonr(penguins.flipper_length_mm, penguins.body_mass_g)
print(covv)

corr = np.cov(penguins.flipper_length_mm, penguins.body_mass_g)

print(corr)




