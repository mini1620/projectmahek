import pandas as pd
dataset = pd.read_csv("d1.csv")
dataset
dataset.isnull()
dataset.isnull().head(10)
dataset.isnull().sum()
dataset.isnull().head().sum()
modifieddataset=dataset.fillna(" ")
modifieddataset.isnull().sum()
dataset=dataset.dropna()
