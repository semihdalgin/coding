import requests
import csv
import pandas
import numpy

r = requests.get('https://api.census.gov/data/2020/acs/acs5?get=NAME,B08303_001E,B08303_013E&for=county:*&in=state:36')

r_json=r.json()

with open('commute_data.csv', mode='w', newline='') as file:
  writer = csv.writer(file)
  writer.writerows(r_json)

commute_df=pandas.read_csv('commute_data.csv')

print(commute_df.head())

commute_df.columns = ['name', 'total_commuters', 'under90', 'state','county']

print(commute_df.head(2))
