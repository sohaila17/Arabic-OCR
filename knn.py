import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



def k_nn(x,y):
    #print('start')
    x = x.reshape(-1, 1) 
    #sets aside 25% of the samples in the original dataset for testing.
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

    #looks for the 5 nearest neighbors. the classifier to use
    #Euclidean distance for determining the proximity between neighboring points.
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

    #‘fit’ method is used to train the model on training data
    knn.fit(X_train, y_train)

    #predict the y values for X-test
    y_pred = knn.predict(X_test)

    #compute the confusion matrix to evaluate the model
    m = confusion_matrix(y_test, y_pred)
    # get the correct predections from the confusion matrix by summing the diagonal
    correct = sum(m[i][i] for i in range(len(m)))

    #get accurecy by dividing the correct predictions by the total prediction
    total = np.sum(m)
    accuracy = correct / total
    #print('finish')
    return accuracy
