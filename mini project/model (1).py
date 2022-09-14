
# **Importing libraries**

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree


data=pd.read_csv("Crop_recommendation.csv")
data.head(5)

data.tail(5)

data.info()

data.describe()

data.size

data.shape

data.columns

data["label"].unique()

data.dtypes

data["label"].value_counts()

data.hist()

sns.heatmap(data.corr(),annot=True)

f = data.iloc[:,:-1]
t= data.iloc[:,-1]
print(f)
print(t)

# **Splitting into train and test data**"""


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(f, t, test_size = 0.3, random_state = 0)



from sklearn.ensemble import RandomForestClassifier


# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
import pickle
pickle.dump(classifier, open("model.pkl", "wb"))