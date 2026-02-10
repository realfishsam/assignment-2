import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gamma, lognorm, skew, kurtosis, shapiro
import numpy as np

df = pd.read_csv('machine_data-1.csv')
time_data = df['time']

print(f"Mean: {time_data.mean():.2f}")
print(f"Median: {time_data.median():.2f}")
print(f"Skewness: {skew(time_data):.2f}")
print(f"Shapiro-Wilk p-value: {shapiro(time_data)[1]:.2e}")

# 1. Normal dist
mu_norm, std_norm = norm.fit(time_data)
# 2. Gammadistribution
a_gamma, loc_gamma, scale_gamma = gamma.fit(time_data)

# Qualitative comparison via Log-Likelihood
log_lik_norm = np.sum(norm.logpdf(time_data, mu_norm, std_norm))
log_lik_gamma = np.sum(gamma.logpdf(time_data, a_gamma, loc_gamma, scale_gamma))

print(f"\n--- Distribution Fitting ---")
print(f"Normal Log-Likelihood: {log_lik_norm:.2f}")
print(f"Gamma Log-Likelihood: {log_lik_gamma:.2f}")

best_dist = "Gamma" if log_lik_gamma > log_lik_norm else "Normal"
print(f"Best distribution based on Log-Likelihood: {best_dist}")

# Plotting
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Histogram
sns.histplot(time_data, kde=False, stat="density", color="salmon", alpha=0.6, label="Actual Data")

# X range for curves
x = np.linspace(time_data.min() - 5, time_data.max() + 5, 200)

# Plot Normal Fit
plt.plot(x, norm.pdf(x, mu_norm, std_norm), 'k--', linewidth=1.5, label='Normal Distribution')

# Plot Gamma Fit
plt.plot(x, gamma.pdf(x, a_gamma, loc_gamma, scale_gamma), 'r-', linewidth=2, 
         label=f'Gamma Distribution (Best Fit)')

plt.title(f'Time Distribution Analysis (Best Fit: {best_dist})', fontsize=15)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()

plt.tight_layout()
plt.savefig('a5.png')
print("\nPlot updated and saved as a5.png")
