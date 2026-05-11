import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('train.csv')
out = pd.read_csv('sample_submission.csv')
datasets = pd.DataFrame(df)
predicted = pd.DataFrame(out)
print(df.head())
print(out.head())

# independent features and dependent features
X = datasets
Y = out

