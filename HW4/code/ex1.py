'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 1: Logistic Regression

Authors: Anja Gumpinger, Dean Bodenham, Bastian Rieck
'''

#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def compute_metrics(y_true, y_pred):
    '''
    Computes several quality metrics of the predicted labels and prints
    them to `stdout`.

    :param y_true: true class labels
    :param y_pred: predicted class labels
    '''

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    
    print('TP: {0:d}'.format(tp))
    print('FP: {0:d}'.format(fp))
    print('TN: {0:d}'.format(tn))
    print('FN: {0:d}'.format(fn))
    print('Accuracy: {0:.3f}'.format(accuracy_score(y_true, y_pred)))


if __name__ == "__main__":

    ###################################################################
    
    import numpy as np 
    import pandas as pd

    train_file = '../data/diabetes_train.csv'
    test_file= '../data/diabetes_test.csv' 
    
    #read data from file using pandas
    df_train = pd.read_csv(train_file)
    df_test=pd.read_csv(test_file)
    

    # extract first 7 columns to data matrix X_train and X_test (actually, a numpy ndarray)
    train = df_train.iloc[:, 0:7].values
    test= df_test.iloc[:, 0:7].values
    
    #scaling the data
    scaler=StandardScaler()
    X_train=scaler.fit_transform(train)
    X_test=scaler.transform(test)


    # extract 8th column (labels) to numpy array Y_train and Y_test
    Y_train = df_train.iloc[:, 7].values
    Y_test = df_test.iloc[:, 7].values

    #fit model
    logistic_regression=LogisticRegression().fit(X_train, Y_train)
    Y_pred=logistic_regression.predict(X_test)

    print('Exercise 1.a')
    print('------------')
    compute_metrics(Y_test, Y_pred)
    print('\n')


    ###################################################################

    print('Exercise 1.b')
    print('------------')

    print('For the diabetes dataset I would choose LDA since it produces less false negatives (28 vs 43 for\n logistic regression) and has higher recall which is the most important aspect in this context.\n')
    
    

    print('Exercise 1.c')
    print('------------')

    print("For another dataset I would choose logistic regression since it is more robust than LDA. For \n LDA you have to consider the assumption of the multivariate Gaussian distribution \n as well as assuming that the covariance matrices are the same for both classes.\n")

    

    

    print('Exercise 1.d')
    print('------------')


    for i in range(7):
        print("Attribute ", i+1 , " has coefficient ",'{0:.2f}'.format(logistic_regression.coef_[0,i]))

    print("\nThe attributes which appear to contribute the most to the prediction \n are the second one (glu: plasma glucose concentration in an oral glucose tolerance test) \n and the sixth one (ped: diabetes pedigree function).\n")
    age_coeff=logistic_regression.coef_[0,6]
    print("The coefficient for age is ",'{0:.2f}'.format(age_coeff) ,". Calculating the exponential function \n results in ", '{0:.2f}'.format(np.exp(age_coeff)), ", which amounts to an increase \n in diabetes risk of", '{0:.1f}'.format((np.exp(age_coeff)-1)*100), " percent per additional year.\n " )


    #performing classification on the reduced data set 
    scaler_red=StandardScaler()
    
  
    train_red=df_train.drop(columns=['skin']).iloc[:, 0:6].values
    X_train_red=scaler_red.fit_transform(train_red)

    test_red=df_test.drop(columns=['skin']).iloc[:, 0:6].values
    X_test_red=scaler_red.transform(test_red)

    logistic_regression_red=LogisticRegression().fit(X_train_red, Y_train)
    Y_pred_red=logistic_regression_red.predict(X_test_red)

    print('Performance on the reduced dataset:')
    compute_metrics(Y_test, Y_pred_red)

    for i in range(6):
        print("Attribute ", i+1 , " has coefficient ",'{0:.2f}'.format(logistic_regression_red.coef_[0,i]))
    print("By comparing the performance and the coefficients obtained on the reduced \n dataset with the ones on the model including all the attributes, I observe \n that the same results are obtained for the reduced dataset. \n")
    print("This is to be expected as when analysing the coefficients for the original dataset \n it can be noted that the skin attribute has a very low value (0), signifying\n that it does not contribute to the classification. \n Therefore when that attribute is deleted no change can be observed.")