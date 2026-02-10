import pandas as pd

df = pd.read_csv('machine_data-1.csv')

if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
    df = df.iloc[:, 1:]

ranges = df.groupby('manufacturef').agg({'load': ['min', 'max'], 'time': ['min', 'max']})

print(ranges)
