# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 14:23:49 2019

@author: Dyass Khalid
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#Reading the data from the csv file

df = pd.read_csv("car_evaluation.csv")
print("Data loaded successfully")

#seperating the prediction variable also known as y
df = df.apply(preprocessing.LabelEncoder().fit_transform)


y = df["Decision"]

#Deleting the decision from df and storing it in x
x = df.drop(df.columns[df.columns.str.contains('Decision',case = False)],axis = 1) #Drop the decision from the data set

#converting into numpy arrays

#splitting the data set into train and test 
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 2)


#Now training a naive bayes classifier

NaiveBayes = GaussianNB()
NaiveBayes.fit(xTrain,yTrain)

#checking score on xTrain and yTrain
print("Accuracy on training data:",NaiveBayes.score(xTrain,yTrain))
print("Accuracy on testing data:",NaiveBayes.score(xTest,yTest))
#predicting the testing data set
y_pred = NaiveBayes.predict(xTest)
#calculating the prediction probabilities
y_probas = NaiveBayes.predict_proba(xTest)
#computing the confusion matrix now
print("Confusion Matrix:")
print(confusion_matrix(yTest,y_pred))
#computing the F1 score 
print("F1 Score:")
print(f1_score(yTest,y_pred,average="macro"))
#Plotting the ROC curve based on testing data
print("Plottong ROC Curves")
#plotting the the roc curve
skplt.metrics.plot_roc(yTest, y_probas)
#saving the curve
plt.savefig('NaiveBayes ROC.png')
#showing the curve
plt.show()




#Decision tree starting from here
clf = tree.DecisionTreeClassifier()
#fitting the decision tree
clf.fit(xTrain,yTrain)
#Checking accuracy on the training data
print("Accuracy on training data:",clf.score(xTrain,yTrain))
#Checking the accuracy on the testing data
print("Accuracy on testing data:",clf.score(xTest,yTest))
#Predicting on the testing data
y_pred = clf.predict(xTest)
#Calculating the prediction probabilities
y_probas = clf.predict_proba(xTest)
#Plotting the confusion matrix
print(confusion_matrix(yTest,y_pred))
#Calculating the F1 score
print("F1 Score:")
print(f1_score(yTest,y_pred,average="macro"))
#Plotting the ROC curve
print("Plottong ROC Curves")
skplt.metrics.plot_roc(yTest, y_probas)
#Saving the ROC Curve
plt.savefig('Decision Trees ROC.png')
plt.show()



#K means starting here
#change n here for cluster center
kmeans = KMeans(n_clusters=5)
#Fitting the classifier
kmeans.fit(xTrain)
#Accuracy on the training data
print("Accuracy on training data:",kmeans.score(xTrain,yTrain))
#Accuracy on the testing data
print("Accuracy on testing data:",kmeans.score(xTest,yTest))
#Predicting  on testing data
y_pred = kmeans.predict(xTest)
#Calculating the confusion matrix
print(confusion_matrix(yTest,y_pred))
#Calculating the f1 score using the metric of macro
print("F1 Score:")
print(f1_score(yTest,y_pred,average="macro"))

#Now for plotting using pca and then getting reduced data to plot
reduced_data = PCA(n_components=2).fit_transform(xTrain)

kmeans.fit(reduced_data)


h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the car evaluation (PCA-reduced data)\n'
          'Centroids are marked with white crosses')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.savefig("Centroids alongwith decision boundaries")
plt.show()











