import pandas as pd

df = pd.read_csv("flightData.csv")
df.to_parquet("flightData.parquet", index=False)