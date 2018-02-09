import pandas as pd
import numpy as np

csv_perf = "DVFS-Performance.csv"
df = pd.read_csv(csv_perf, header = 0)

print df.head(3)
print df.dtypes