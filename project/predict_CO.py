import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


df = pd.read_csv("airQuality.csv")
#print(df.head())

#droping the redundant columns
df.drop(['date'], axis=1, inplace=True)
df.drop(['cityname'], axis=1, inplace=True)
df.drop(['longitude'], axis=1, inplace=True)
df.drop(['latitude'], axis=1, inplace=True)

#dropping all the rows which have null values
df = df.dropna()  #Drops all rows with at least one null value. 

#This is what we are predecting 
Y = df["CO"].values  #At this point Y is an object not of type int
#Convert Y to int
Y=Y.astype('int')
#this is what we are using to predict the Y
X = df.drop(labels = ["CO"], axis=1) 

#Setting the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=500)

#Creating the random forest
random_forest = RandomForestRegressor(n_estimators = 50)
# Train the model on training data
model_r = random_forest.fit(X_train, y_train)
random_forest_pred = model_r.predict(X_test)
#creating the decision tree
decisionTree = DecisionTreeRegressor()
decisionTree = decisionTree.fit(X_train, y_train)
decisionTree_pred = decisionTree.predict(X_test)
#creating the fradient boost classifier
GradientBoostingClassifier = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, validation_fraction=0.1,  n_iter_no_change=20, max_features='log2' )
GradientBoostingClassifier.fit(X_train, y_train)
GradientBoostingClassifier_pred = GradientBoostingClassifier.predict(X_test)




#Print the prediction accuracy for all the models
print ("Accuracy for random forest = ", explained_variance_score(y_test, random_forest_pred))
print ("Accuracy for Decision tree = ", explained_variance_score(y_test, decisionTree_pred))
print ("Accuracy for Gradient Boosting = ", explained_variance_score(y_test, GradientBoostingClassifier_pred))



#print MAE and Prediction results for all 
print("true values=", y_test)
print("Decision tree prediction results", decisionTree_pred)
print("Decision tree MAE = ", mean_absolute_error(y_test, decisionTree_pred))
print("random_forest prediction results", random_forest_pred)
print("random_forest MAE = ", mean_absolute_error(y_test, random_forest_pred))
print("GradientBoostingClassifier prediction results", GradientBoostingClassifier_pred)
print("GradientBoostingClassifier MAE = ", mean_absolute_error(y_test, GradientBoostingClassifier_pred))


#Importance of each columns
feature_list = list(X.columns)
feature_imp = pd.Series(model_r.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)