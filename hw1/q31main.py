import numpy as np
import pandas as pd
import os
import math

## Change root to where the CSV files are located.
root = r'C:\Users\User\Desktop\hw3'
os.chdir(root)

# If CSV's are named differently, change them here.
train_x_csv = root + '\sms_train_features.csv'
train_y_csv = root + '\sms_train_labels.csv'
test_x_csv = root + '\sms_test_features.csv'
test_y_csv = root + '\sms_test_labels.csv'

############BEGINNING OF MULTINOMIAL############
def Multinomial():

    train_x = pd.read_csv(train_x_csv)
    train_y = pd.read_csv(train_y_csv)
    test_x = pd.read_csv(test_x_csv)
    test_y = pd.read_csv(test_y_csv)

    train_x.drop('Unnamed: 0', inplace=True, axis=1)
    train_y.drop('Unnamed: 0', inplace=True, axis=1)
    test_x.drop('Unnamed: 0', inplace=True, axis=1)
    test_y.drop('Unnamed: 0', inplace=True, axis=1)
    test_y = test_y.values.tolist()

    #merge labels to features
    train_x['class'] = train_y
    spam = train_x.loc[train_x['class'] == 1]
    ham = train_x.loc[train_x['class'] == 0]

#first part of the equation
    #count of each class and the number of all of them
    spamCount = len(spam.index)
    hamCount = len(ham.index)
    total_email = spamCount + hamCount

    #calculate pi and take log
    #log P (Y = yk)
    pi_spam = np.log2(spamCount / total_email)
    pi_ham = np.log2(hamCount / total_email)
#end of the first part of the equation

    #calculate theta
    theta_spam = pd.DataFrame()
    theta_ham = pd.DataFrame()
    #calculate the number of occurrences of the word j in spam messages in the training set including the
    #multiple occurrences of the word in a single message.
    #T j, spam
    theta_spam['total'] = spam.iloc[:, :-1].sum(axis = 0)

    #calculate the number of occurrences of the word j in ham message in the training set including
    #the multiple occurrences of the word in a single message.
    #T j, ham
    theta_ham['total'] = ham.iloc[:, :-1].sum(axis = 0)

    #ΣT j, spam
    total_theta_spam = theta_spam.sum(axis = 0)

    #ΣT j, ham
    total_theta_ham = theta_ham.sum(axis = 0)

    temp_spam = []
    for total in theta_spam['total']:
        temp_spam.append(np.log2(total / total_theta_spam['total']) if total > 0 else -math.inf)

    theta_spam['log'] = temp_spam

    temp_ham = []
    for total in theta_ham['total']:
        temp_ham.append(np.log2(total / total_theta_ham['total']) if total > 0 else -math.inf)

    theta_ham['log'] = temp_ham

    theta_spam_list = theta_spam['log'].tolist()
    theta_ham_list = theta_ham['log'].tolist()

    #test part
    confusion_matrix = np.array([[0, 0], [0, 0]])

    x_test_spam = test_x.copy()
    x_test_ham = test_x.copy()

    x_test_spam = x_test_spam * theta_spam_list
    x_test_ham = x_test_ham * theta_ham_list

    x_test_spam = x_test_spam.sum(axis=1)
    x_test_ham = x_test_ham.sum(axis=1)

    x_test_spam = x_test_spam.apply(lambda x: x + pi_spam)
    x_test_ham = x_test_ham.apply(lambda x: x + pi_ham)

    #predict
    predictions = np.where(x_test_ham >= x_test_spam, 0, 1)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    i = 0

    for prediction in predictions:
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

    acc = (TP + TN)/(TP+TN+FP+FN)
    print("Accuracy:", acc * 100)
    print("Confusion Matrix:")
    print(confusion_matrix)

############END OF MULTINOMIAL############

def main():

    Multinomial()

if __name__ == '__main__':
    main()