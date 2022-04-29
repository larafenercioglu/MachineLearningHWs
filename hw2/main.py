import statistics

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

ROOT = r'C:\Users\User\Desktop\cs\hw2'
os.chdir(ROOT)

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def PCA():
    '''
    QUESTION 1.1
    '''
    # Read images.csv into a numpy array
    images = np.array(np.genfromtxt('images.csv', delimiter=',')[1:])
    #Mean center the data
    mean = np.mean(images, axis=0)
    centered = images - mean

    #Compute covariance matrix
    covar = np.cov(centered.T)

    #Calculate eigenvalues and eigenvectors of covariance matrix
    eig_val, eig_vec = np.linalg.eigh(covar)

    '''
    Eigenvector with largest eigenvalue λ1 is 1st principal component (PC)
    Eigenvector with kth largest eigenvalue λk is kth PC
    '''
    inds = np.argsort(eig_val)[::-1]
    sorted_eig_val = eig_val[inds]
    sorted_eig_vec = eig_vec[:, inds]

    kVectors = sorted_eig_vec[:, :10]
    kValues = sorted_eig_val[:10]

    #Proportion of variance captured by kth PC = λk / Σi λI
    pve = kValues / sum(eig_val)
    print("PVE for", 10, "components:",pve)
    mng = plt.get_current_fig_manager()
    mng.set_window_title(f"PVE values for {10} components")
    plt.bar(range(1, 10 + 1), pve)
    plt.xlabel("kth principal component")
    plt.ylabel("PVE")
    plt.show()

    v = pd.DataFrame(kVectors)

    fig = plt.figure(figsize=(10, 5))
    col = 5
    row = 2
    for i in range(0, 10):
        df = pd.DataFrame(np.array(v[i]).reshape((48, 48)))
        fig.add_subplot(row, col, i+1)
        plt.imshow(df,cmap="magma")
        plt.title(f"Component {i+1}")
        mng = plt.get_current_fig_manager()
        mng.set_window_title("FIRST 10 PC'S AS 48X48 MATRICES")
    plt.show()

    '''
    QUESTION 1.2
    '''
    k = [1, 10, 50, 100, 500]
    PVE = sorted_eig_val / sum(eig_val)
    pve = [sum(PVE[:i]) for i in k]
    plt.figure('PVE vs k')
    plt.title('PVE for k Principal Components')
    plt.xlabel('k')
    plt.ylabel('PVE')
    plt.xticks(k)
    plt.plot(k, pve)
    plt.show()

    '''
    QUESTION 1.3
    '''
    imgs = []
    ks = [1, 10, 50, 100, 500]
    fig = plt.figure(figsize=(10, 8))
    col = 5
    row = 1
    x = 1

    for i in ks:
        fig.add_subplot(row, col, x)
        plt.title(f"With k = {i}")
        principal = sorted_eig_vec[:, :i]
        reduced = np.dot(centered, principal)
        reconstruction = np.dot(reduced, principal.T) + mean
        img = [reconstruction[0].reshape(48, 48), i]
        plt.imshow(img[0], cmap='magma')
        x = x + 1

    mng = plt.get_current_fig_manager()
    mng.set_window_title("Reconstructing the first digit with different k values")
    plt.show()

def linearRegression():
    features = pd.read_csv('question-2-features.csv')
    labels = pd.read_csv('question-2-labels.csv')

    featuresTranspose = features.transpose()
    XTX = np.dot(featuresTranspose, features)
    rankXTX = np.linalg.matrix_rank(XTX)
    rank_x = np.linalg.matrix_rank(features)
    print("rank xtx: ", rankXTX)
    print("rank x:",rank_x)

    #lstat feature only
    lstat = pd.DataFrame()
    lstat["lstat"] = features["LSTAT"]
    X = np.ones((len(lstat), 1))
    lstat.insert(0,"ones",X)

    transposed_lstat = lstat.T
    beta = np.linalg.inv(transposed_lstat.dot(lstat)).dot(transposed_lstat).dot(labels)

    print("coefficients", beta)

    prediction = lstat.dot(beta)
    prediction["labels"] = labels["Price"]
    prediction.rename(columns={0: "preds"}, inplace=True)
    MSE = np.square(np.subtract(prediction["labels"], prediction["preds"])).mean()
    print("MSE:", MSE)

    #plot
    plt.scatter(lstat["lstat"], labels)
    plt.plot(np.array(lstat["lstat"]), np.array(prediction["preds"]), color='red')
    plt.xlabel("LSTAT")
    plt.ylabel("Prices")
    mng = plt.get_current_fig_manager()
    mng.set_window_title("Linear Regression plot")
    plt.show()

def polynomialRegression():
    features = pd.read_csv('question-2-features.csv')
    labels = pd.read_csv('question-2-labels.csv')

    lstat = pd.DataFrame()
    lstat["lstat"] = features["LSTAT"]
    X = np.ones((len(lstat), 1))
    lstat.insert(0,"ones",X)
    lstat["x^2"] = lstat["lstat"] ** 2
    transposed_lstat = lstat.T
    beta = np.linalg.inv(transposed_lstat.dot(lstat)).dot(transposed_lstat).dot(labels) # calculate the coefficients
    print("coefficients", beta)

    prediction = lstat.dot(beta)
    prediction["labels"] = labels["Price"]
    prediction.rename(columns={0: "preds"}, inplace=True)
    MSE = np.square(np.subtract(prediction["labels"], prediction["preds"])).mean()
    print("MSE:", MSE)

    plt.scatter(np.array(lstat["lstat"]),labels)
    fix = np.argsort(lstat["lstat"])
    plt.plot(lstat["lstat"][fix], prediction["preds"][fix], color='red')
    plt.xlabel("lstat")
    plt.ylabel("Prices")
    mng = plt.get_current_fig_manager()
    mng.set_window_title("Polynomial Regression plot")
    plt.show()

def logisticRegressionFullBatch():
    learnig_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    train_features = pd.read_csv('question-3-features-train.csv')
    train_labels = pd.read_csv('question-3-labels-train.csv')
    test_features = pd.read_csv('question-3-features-test.csv')
    test_labels = pd.read_csv('question-3-labels-test.csv')
    iter = 1000

    # normalize fare
    meanFare = np.mean(train_features["Fare"])
    stdev = statistics.stdev(train_features["Fare"])
    train_features["Fare"] = train_features["Fare"].apply(lambda x: (x - meanFare) / stdev)

    meanFare2 = np.mean(test_features["Fare"])
    stdev2 = statistics.stdev(test_features["Fare"])
    test_features["Fare"] = test_features["Fare"].apply(lambda x: (x - meanFare2) / stdev2)

    # normalize Pclass
    meanPclass = np.mean(train_features["Pclass"])
    stdev = statistics.stdev(train_features["Pclass"])
    train_features["Pclass"] = train_features["Pclass"].apply(lambda x: (x - meanPclass) / stdev)

    meanPclass2 = np.mean(test_features["Pclass"])
    stdev2 = statistics.stdev(test_features["Pclass"])
    test_features["Pclass"] = test_features["Pclass"].apply(lambda x: (x - meanPclass2) / stdev2)

    # normalize Age
    meanAge = np.mean(train_features["Age"])
    stdev = statistics.stdev(train_features["Age"])
    train_features["Age"] = train_features["Age"].apply(lambda x: (x - meanAge) / stdev)

    meanAge2 = np.mean(test_features["Age"])
    stdev2 = statistics.stdev(test_features["Age"])
    test_features["Age"] = test_features["Age"].apply(lambda x: (x - meanAge2) / stdev2)

    Xte = np.ones((len(test_features), 1))
    test_features.insert(0, "ones", Xte)  # fill the first column with 1s
    Xtr = np.ones((len(train_features), 1))
    train_features.insert(0, "ones", Xtr)  # fill the first column with 1s

    weights = np.zeros(train_features.shape[1])
    for learningRate in learnig_rates:
        print("-----For learning rate", learningRate,"-----")
        for step in range(iter):
            scores = np.dot(train_features, weights)
            predictions = sigmoid(scores)
            error = train_labels - predictions.reshape(len(train_labels),1)
            gradient = np.dot(train_features.T, error)
            weights += learningRate * gradient.reshape(4)

        final_values = np.dot(test_features,weights)
        preds = np.round(sigmoid(final_values))
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        k = 0

        for i in test_labels["Survived"]:
            if int(preds[k]) == 0 and i == 0:  # true negative
                TN += 1
            elif preds[k] == 1 and i == 1:  # true positive
                TP += 1
            elif preds[k] == 0 and i == 1:  # false positive
                FP += 1
            elif preds[k] == 1 and i == 0:  # false negative
                FN += 1
            k += 1

        confusion_matrix = np.array([[0, 0], [0, 0]])

        confusion_matrix[0][0] = TP
        confusion_matrix[0][1] = FN
        confusion_matrix[1][0] = FP
        confusion_matrix[1][1] = TN

        acc = ((TN + TP) / len(preds)) * 100
        print("Full Batch Gradient Ascent accuracy=", acc)
        Precision= TP / (TP + FP)
        Recall= TP / (TP + FN)
        print("Precision=", Precision)
        print("Recall=", Recall)
        print("NPV=", TN / (TN + FN))
        print("FPR=", FP / (FP + TN))
        print("FDR=", FP / (FP + TP))
        print("F1=", 2*TP / (2*TP + FP + FN))
        print("F2=", 5*Precision*Recall / (4*Precision + Recall))
        print("Confusion Matrix:")
        print(confusion_matrix)

def batch(f, l, size):
    for i in np.arange(0, f.shape[0], size):
        yield f[i:i + size], l[i:i + size]

def logisticRegressionMiniBatch():
    learning_rate = 0.0001
    train_features = pd.read_csv('question-3-features-train.csv')
    train_labels = pd.read_csv('question-3-labels-train.csv')
    test_features = pd.read_csv('question-3-features-test.csv')
    test_labels = pd.read_csv('question-3-labels-test.csv')
    iter = 1000
    batch_size = 100
    w = np.random.uniform(0, 0.1, size=(train_features.shape[1]))

    for step in range(iter):
        for (batchX, batchY) in batch(train_features, train_labels, batch_size):
            p = sigmoid(batchX.dot(w))
            error = batchY["Survived"] - p
            gradient = np.dot(batchX.transpose(), error)
            w += learning_rate * gradient

    final_values = np.round(np.dot(test_features, w))
    preds = np.round(sigmoid(final_values))

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    k = 0

    for i in test_labels["Survived"]:
        if int(preds[k]) == 0 and i == 0:  # true negative
            TN += 1
        elif preds[k] == 1 and i == 1:  # true positive
            TP += 1
        elif preds[k] == 0 and i == 1:  # false positive
            FP += 1
        elif preds[k] == 1 and i == 0:  # false negative
            FN += 1
        k += 1

    confusion_matrix = np.array([[0, 0], [0, 0]])

    confusion_matrix[0][0] = TP
    confusion_matrix[0][1] = FN
    confusion_matrix[1][0] = FP
    confusion_matrix[1][1] = TN

    acc = ((TN + TP) / len(preds)) * 100
    print("Mini Batch Gradient Ascent accuracy=", acc)
    Precision= TP / (TP + FP)
    Recall= TP / (TP + FN)
    print("Precision=", Precision)
    print("Recall=", Recall)
    print("NPV=", TN / (TN + FN))
    print("FPR=", FP / (FP + TN))
    print("FDR=", FP / (FP + TP))
    print("F1=", 2*TP / (2*TP + FP + FN))
    print("F2=", 5*Precision*Recall / (4*Precision + Recall))
    print("Confusion Matrix:")
    print(confusion_matrix)

def logisticRegressionStochastic():
    learnig_rate = 0.0001
    train_features = pd.read_csv('question-3-features-train.csv')
    train_labels = pd.read_csv('question-3-labels-train.csv')
    test_features = pd.read_csv('question-3-features-test.csv')
    test_labels = pd.read_csv('question-3-labels-test.csv')
    iter = 1000

    w = np.random.uniform(0, 0.1, size=(train_features.shape[1]))
    ftrain = np.array(train_features)
    ltrain = np.array(train_labels)

    for step in range(0, iter):
        for index in range(0, len(ltrain)):
            X = ftrain[index,:]
            Y = ltrain[index]
            p = sigmoid(X.dot(w))
            error = Y - p
            gradient = np.dot(X.transpose().reshape(3,1), error)
            w += learnig_rate * gradient.reshape(3)

    final_values = np.round(np.dot(test_features, w))
    preds = np.round(sigmoid(final_values))
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    k = 0

    for i in test_labels["Survived"]:
        if int(preds[k]) == 0 and i == 0:  # true negative
            TN += 1
        elif preds[k] == 1 and i == 1:  # true positive
            TP += 1
        elif preds[k] == 0 and i == 1:  # false positive
            FP += 1
        elif preds[k] == 1 and i == 0:  # false negative
            FN += 1
        k += 1

    confusion_matrix = np.array([[0, 0], [0, 0]])

    confusion_matrix[0][0] = TP
    confusion_matrix[0][1] = FN
    confusion_matrix[1][0] = FP
    confusion_matrix[1][1] = TN

    acc = ((TN + TP) / len(preds)) * 100
    print("Stochastic Gradient Ascent accuracy=", acc)
    Precision= TP / (TP + FP)
    Recall= TP / (TP + FN)
    print("Precision=", Precision)
    print("Recall=", Recall)
    print("NPV=", TN / (TN + FN))
    print("FPR=", FP / (FP + TN))
    print("FDR=", FP / (FP + TP))
    print("F1=", 2*TP / (2*TP + FP + FN))
    print("F2=", 5*Precision*Recall / (4*Precision + Recall))
    print("Confusion Matrix:")
    print(confusion_matrix)

def main():
    #QUESTION 1
    print("----------QUESTION 1----------")
    PCA()

    #QUESTION 2
    print("----------QUESTION 2----------")
    linearRegression()
    polynomialRegression()

    #QUESTION 3
    print("----------QUESTION 3----------")
    logisticRegressionFullBatch()
    logisticRegressionMiniBatch()
    logisticRegressionStochastic()

if __name__ == '__main__':
    main()