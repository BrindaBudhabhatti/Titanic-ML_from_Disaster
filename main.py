import os
from sklearn.linear_model import LogisticRegression
import pandas as pd


# load training and testing dataset
training_file = r'D:\brinda\Python\Titanic - Machine Learning from Disaster\train.csv'
training_df = pd.read_csv(training_file)

testing_file = r'D:\brinda\Python\Titanic - Machine Learning from Disaster\test.csv'
testing_df = pd.read_csv(testing_file)


# display first few lines of training and testing dataset
print(training_df.head())
print(testing_df.head())


# feature and target selection for training dataset
train_features = ["PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
train_target = ["Survived"]


# feature selection for testing dataset
test_features = ["PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]

# initialising model
model = LogisticRegression()
model.fit(train_features, train_target)