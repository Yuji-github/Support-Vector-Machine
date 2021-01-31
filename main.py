import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# splitting dataset
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# classifier
from sklearn.svm import SVC

# confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

# plotting
from matplotlib.colors import ListedColormap # for plot


def SVM():
    # import data
    dataset = pd.read_csv('Social_Network_Ads.csv')

    # independent variables
    independent = dataset.iloc[:, :-1].values

    # dependent variable
    dependent = dataset.iloc[:, -1].values

    # print(independent, dependent)

    # splitting dataset into 4 parts 300 customers go to train
    x_train, x_test, y_train, y_test = train_test_split(independent, dependent, train_size=0.75, random_state=0)

    # feature scaling
    # this is not necessary to do, but for better prediction we do
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    # classifiler
    # this time, kerel is linear because need a linearly separation
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(x_train, y_train)

    # prediction
    print('The prediction value is %.2f' % classifier.predict(sc.transform([[30, 87000]])))

    # confusion matrix
    cm = confusion_matrix(y_true=y_test, y_pred=classifier.predict(x_test))
    print(cm)

    print('\nThe correctness is %.2f percent' % accuracy_score(y_true=y_test, y_pred=classifier.predict(x_test)))

    # taking several min to plot
    # because a lot of calculations behind the code

    # Train set visualization
    x_set, y_set = sc.inverse_transform(x_train), y_train
    X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=1),
                         np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=1))
    plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('SVM (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    SVM()

