import pandas as pd
import numpy as np
from nn import MultiLayerPerceptron
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report

if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    X = np.array(df.iloc[:, 1:])
    y = np.array(df.iloc[:,0])
#    digits = load_digits()
#    X = digits.data
#    y = digits.target
    X /= X.max()

    mlp = MultiLayerPerceptron(784, 300, 10, act1="sigmoid", act2="sigmoid")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    labels_train = LabelBinarizer().fit_transform(y_train)
    labels_test = LabelBinarizer().fit_transform(y_test)
    

    mlp.fit(X_train, labels_train, epochs=50000)
    
    predictions = []
    for i in range(X_test.shape[0]):
        o = mlp.predict(X_test[i])
        predictions.append(np.argmax(o))
    
    print confusion_matrix(y_test, predictions)
    print classification_report(y_test, predictions)

