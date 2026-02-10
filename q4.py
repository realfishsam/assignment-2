import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import numpy as np

df = pd.read_csv('machine_data-1.csv')
load = df['load']

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.histplot(load, kde=True, stat="density", color="skyblue", label="Actual Data")

mu, std = load.mean(), load.std()
x = np.linspace(load.min(), load.max(), 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r', linewidth=2, label=f'Normal Fit (μ={mu:.2f}, σ={std:.2f})')

plt.title('Distribution of Machine Load', fontsize=15)
plt.xlabel('Load', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()

plt.tight_layout()
plt.savefig('a4.png')
