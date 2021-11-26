#-------------------------------------------------------------------------
# AUTHOR: Seungyun Lee
# FILENAME: association_rule_mining.py
# SPECIFICATION: Practicing association rule mining
# FOR: CS 4200- Assignment #5
# TIME SPENT: 24 hr
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

def count_support_counts(itemset, all_items):
    support_counts = 0

    # Go thru every row of all the items
    for row in all_items:
        # Reset all_passed
        all_passed = True
        # Go thru every item in the itemset
        for item in itemset:
            # If current row is missing an item, break out and set all_passed as False
            if row[item] == 0:
                all_passed = False
                break
        # If all_passed means that every item appeared
        if all_passed:
            support_counts = support_counts + 1

    return support_counts


#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

#find the unique items all over the data an store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))


#remove nan (empty) values by using:
itemset.remove(np.nan)

# print(itemset)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:

#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)
#-->add your python code below

encoded_vals = []
for index, row in df.iterrows():
    labels = {}

    # Create and initialize dict with 0
    for item in itemset:
        labels[item] = 0

    # Update the values of items in the dict to 1
    for i in range(row.size):
        if row[i] in labels:
            labels[row[i]] = 1

    encoded_vals.append(labels)


#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

#iterate the rules data frame and print the apriori algorithm results by using the following format:

#Meat, Cheese -> Eggs
#Support: 0.21587301587301588
#Confidence: 0.6666666666666666
#Prior: 0.4380952380952381
#Gain in Confidence: 52.17391304347825
#-->add your python code below

#To calculate the prior and gain in confidence, find in how many transactions the consequent of the rule appears (the supporCount below). Then,
#use the gain formula provided right after.
#prior = suportCount/len(encoded_vals) -> encoded_vals is the number of transactions
#print("Gain in Confidence: " + str(100*(rule_confidence-prior)/prior))
#-->add your python code below

for index,row in rules.iterrows():
    antecedents = list(row["antecedents"])
    consequents = list(row["consequents"])

    for i in range(len(antecedents)):
        if i < len(antecedents)-1:
            print(antecedents[i] + ", ", end=" ")
        else:
            print(antecedents[i] + " -> ", end=" ")

    for i in range(len(consequents)):
        if i < len(consequents)-1:
            print(consequents[i] + ", ", end=" ")
        else:
            print(consequents[i])

    prior = count_support_counts(consequents, encoded_vals)/len(encoded_vals)
    print("Support: ", row["support"])
    print("Confidence: ", row["confidence"])
    print("Prior: ", prior)
    print("Gain in Confidence: ", 100*(row["confidence"]-prior)/prior)
    print("-------------------------------------------------")

#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()