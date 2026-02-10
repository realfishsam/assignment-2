import pandas as pd

df = pd.read_csv('machine_data-1.csv')

if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
    df = df.iloc[:, 1:]

print(df['load'].corr(df['time'])) # -.58. As time increases, load decreases
