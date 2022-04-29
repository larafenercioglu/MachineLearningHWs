import os
import math
import random
import numpy as np
import operator
import pandas as pd
import time
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

def euclidean_distance(sample1, sample2):
    assert len(sample1) == len(sample2), "Different number of features exist for the given samples"

    distance = 0
    for i in range(len(sample1)):
        distance += pow((sample1[i] - sample2[i]), 2)
    return math.sqrt(distance)

def get_neighbors(train_X, train_Y, test_sample, k):
    distances = []

    for train_sample, train_gt, in zip(train_X, train_Y):
        distance = euclidean_distance(train_sample, test_sample)
        distances.append((train_sample, train_gt, distance))
    distances.sort(key=operator.itemgetter(2))
    return distances[:k]

def classify(neighbors, classes):
    class_votes = {cls: 0 for cls in classes}
    for n in neighbors:
        c = n[1]
        class_votes[c[0]] += 1

    return max(class_votes, key=class_votes.get)

def calc_accuracy(gt_y, pred_y):
    correct = 0
    for g_y, p_y in zip(gt_y, pred_y):
        if g_y == p_y:
            correct += 1
    return (correct / float(len(gt_y)) * 100)

class KNN:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y
        self.classes = np.unique(y)

    def predict(self, X):
        predictions = []
        for x in X:
            neighbors = get_neighbors(self.train_X, self.train_y, x, self.n_neighbors)
            p_class = classify(neighbors, self.classes)
            predictions.append(p_class)
        return predictions

################################################# MAIN PROGRAM #################################################
random.seed(42)
np.random.seed(42)

to_remove_features = list()
achieved_accs = list()
results = dict()

"""
results = {[removed feature]: acc}
"""

# Change root to where the CSV files are located.
root = r'C:\Users\User\Desktop\hw3'

os.chdir(root)

print("Reading Data")
# If CSV's are named differently, change them here.
train_x_csv = root + '\diabetes_train_features.csv'
train_y_csv = root + '\diabetes_train_labels.csv'
test_x_csv = root + '\diabetes_test_features.csv'
test_y_csv = root + '\diabetes_test_labels.csv'

train_x = pd.read_csv(train_x_csv)
train_y = pd.read_csv(train_y_csv)
test_x = pd.read_csv(test_x_csv)
test_y = pd.read_csv(test_y_csv)

train_x.drop('Unnamed: 0', inplace=True, axis=1)
test_x.drop('Unnamed: 0', inplace=True, axis=1)
train_y.drop('Unnamed: 0', inplace=True, axis=1)
test_y.drop('Unnamed: 0', inplace=True, axis=1)

all_columns = list(train_x.columns)

train_y = train_y.to_numpy()
test_y = test_y.to_numpy()

train_x = train_x.to_numpy()
test_x = test_x.to_numpy()

knn = KNN(n_neighbors=9)
start = time.time()
knn.fit(train_x, train_y)
end = time.time()
print("train time:", end - start, "seconds")
start = time.time()
pred_y = knn.predict(test_x)
end = time.time()
print("test time:", end - start, "seconds")
max_acc = calc_accuracy(test_y, pred_y)
TP = 0
FP = 0
TN = 0
FN = 0
i = 0
confusion_matrix = np.array([[0, 0], [0, 0]])

for prediction in pred_y:
    if prediction == 0 and test_y[i][0] == 0:
        TN += 1
    elif prediction == 0 and test_y[i][0] == 1:
        FN += 1
    elif prediction == 1 and test_y[i][0] == 1:
        TP += 1
    elif prediction == 1 and test_y[i][0] == 0:
        FP += 1
    i += 1

confusion_matrix[0][0] = TP
confusion_matrix[0][1] = FN
confusion_matrix[1][0] = FP
confusion_matrix[1][1] = TN

print("Confusion Matrix:")
print(confusion_matrix)
print("No features subtracted")
print("accuracy:", max_acc)
print("\n")

columns = list(set(all_columns)-set(to_remove_features))

def backward_elimination():
    print("BACKWARD ELIMINATION STARTS")
    for column in columns:
        train_x_cp = pd.read_csv(train_x_csv)
        test_x_cp = pd.read_csv(test_x_csv)
        train_x_cp.drop('Unnamed: 0', inplace=True, axis=1)
        test_x_cp.drop('Unnamed: 0', inplace=True, axis=1)

        for r in to_remove_features:
            train_x_cp.drop(r, inplace=True, axis=1)
            test_x_cp.drop(r, inplace=True, axis=1)

        print("Subtracted feature:",column)
        train_x_cp.drop(column, inplace=True, axis=1)
        test_x_cp.drop(column, inplace=True, axis=1)

        train_x_cp = train_x_cp.to_numpy()
        test_x_cp = test_x_cp.to_numpy()

        knn = KNN(n_neighbors=9)
        start = time.time()
        knn.fit(train_x_cp, train_y)
        end = time.time()
        print("train time:", end - start, "seconds")
        start = time.time()
        pred_y = knn.predict(test_x_cp)
        end = time.time()
        print("test time:", end - start, "seconds")
        acc = calc_accuracy(test_y, pred_y)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        i = 0
        confusion_matrix = np.array([[0, 0], [0, 0]])

        for prediction in pred_y:
            if prediction == 0 and test_y[i][0] == 0:
                TN += 1
            elif prediction == 0 and test_y[i][0] == 1:
                FN += 1
            elif prediction == 1 and test_y[i][0] == 1:
                TP += 1
            elif prediction == 1 and test_y[i][0] == 0:
                FP += 1
            i += 1

        confusion_matrix[0][0] = TP
        confusion_matrix[0][1] = FN
        confusion_matrix[1][0] = FP
        confusion_matrix[1][1] = TN

        print("Confusion Matrix:")
        print(confusion_matrix)
        print("accuracy:", acc)
        print("\n")
        results[column] = acc

    print("\n")
    x = list(results.keys())
    y = list(results.values())

    plt.plot(x, y, 'bo')
    plt.xlabel('removed feature')
    plt.ylabel('accuracy')

    plt.title("Backward Elimination")
    plt.show()

############################ features are removed one by one ############################
backward_elimination()
max_key = max(results, key=results.get)
to_remove_features.append(max_key)
achieved_accs.append(results[max_key])
results.pop(max_key)
columns = list(set(all_columns)-set(to_remove_features))

while max_acc < achieved_accs[len(achieved_accs)-1]:
    max_acc = achieved_accs[len(achieved_accs) - 1]
    backward_elimination()
    max_key = max(results, key=results.get)
    to_remove_features.append(max_key)
    achieved_accs.append(results[max_key])
    results.pop(max_key)
    columns = list(set(all_columns)-set(to_remove_features))

to_remove_features.remove(to_remove_features[len(to_remove_features)-1])
achieved_accs.remove(achieved_accs[(len(achieved_accs)-1)])
print("Removed features are\n",to_remove_features)
print("Accuracy after removing the above features is:",max_acc)



