import pandas as pd

df = pd.read_csv('machine_data-1.csv')

if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
    df = df.iloc[:, 1:]

most_expected_load = df['load'].mean()

# Expected load
print(most_expected_load)
