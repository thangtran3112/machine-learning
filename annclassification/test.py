import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle

## Load the dataset
data=pd.read_csv("Churn_Modelling.csv")
data_head = data.head()

print(data_head)