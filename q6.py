import pandas as pd

df = pd.read_csv('machine_data-1.csv')
df = df.rename(columns={'manufacturef': 'manufacturer'})

stats = df.groupby('manufacturer').agg({
    'time': 'mean',
    'load': 'mean'
})
stats['efficiency'] = stats['load'] / stats['time']

print("Performance Statistics by Manufacturer:")
print(stats)

best_manufacturer = stats['time'].idxmin()

"""
Answer: Manufacturer '{best_manufacturer}' has the best performance."
Why: Manufacturer '{best_manufacturer}' has the lowest average processing time ({stats.loc[best_manufacturer, 'time']:.2f})
despite handling similar average loads ({stats.loc[best_manufacturer, 'load']:.2f}) as the other manufacturers.
This results in the highest efficiency (load/time) among all manufacturers.
"""
