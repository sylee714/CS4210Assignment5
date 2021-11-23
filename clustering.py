#-------------------------------------------------------------------------
# AUTHOR: Seungyun Lee
# FILENAME: clustering.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = df.to_numpy()
k_values = []
silhouette_scores = []
max_k = 0
max_s_score = 0

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
for k in range(2,21):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
    s_score = silhouette_score(X_training, kmeans.labels_)
    print("k=" + str(k) + " | s_score=" + str(s_score))
    if s_score > max_s_score:
        max_s_score = s_score
        max_k = k
    k_values.append(k)
    silhouette_scores.append(s_score)

print("Max K = ", max_k)
print("Max Silhouette Coeff = ", max_s_score)

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
plt.plot(k_values, silhouette_scores)
plt.ylabel('Silhouette Coefficient')
plt.xlabel('K')
plt.show()

#reading the validation data (clusters) by using Pandas library
#--> add your Python code here
test_df = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(test_df.values).reshape(1, test_df.size)[0] # [0] becuz it returns it as an array of an array
print(labels)

#Calculate and print the Homogeneity of this kmeans clustering
# print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
kmeans = KMeans(n_clusters=max_k, random_state=0)
kmeans.fit(X_training)
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())

#run agglomerative clustering now by using the best value of k calculated before by kmeans
#Do it:
agg = AgglomerativeClustering(n_clusters=max_k, linkage='ward')
agg.fit(X_training)

#Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())
