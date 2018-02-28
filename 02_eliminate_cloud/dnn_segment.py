# Copyright 2018 Jeffrey Hoa. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# How to run it?
# $ python main.py
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def func_dnn_segment(X, y, resultFile, rst_h, rst_w):
    # config
    alphas = np.logspace(-4, 1, 1)
    nnShape = (50,60)

    #-----------------------------------

    h = .02  # step size in the mesh
    names = []
    for i in alphas:
        names.append('alpha ' + str(i))

    classifiers = []
    for i in alphas:
        classifiers.append(MLPClassifier(alpha=i, random_state=1, hidden_layer_sizes=nnShape, max_iter=500000))

    i = 1
    #figure = plt.figure(figsize=(17, 9))
    figure = plt.figure(figsize=(9, 9))

    datasets = [(X, y)]


    # iterate over datasets
    for X, y in datasets:

        # preprocess dataset, split into training and test part
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.05)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        #cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        # Plot the training points
        #ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        #ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        #ax.set_xlim(xx.min(), xx.max())
        #ax.set_ylim(yy.min(), yy.max())
        #ax.set_xticks(())
        #ax.set_yticks(())
        #i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            ax = plt.subplot(len(datasets), len(classifiers), i)

            ############################
            # training...
            ############################
            clf.fit(X_train, y_train)
            #score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                #print("----------Jeff decision_function----------")
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                #print("----------Jeff predict_proba----------")
                #print("xx",xx)
                #print("xx.ravel()",xx.ravel())
                #print("yy",yy)
                #print("yy.ravel()",yy.ravel())
                #print("np.c_",np.c_[xx.ravel(), yy.ravel()])
                #print("-->", clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]))
                #print("Z:", Z)


            # Put the result into a color plot
            #print("xx.shape", xx.shape)
            #print("Z.shape", Z.shape)
            Z = Z.reshape(xx.shape)

            #print("################################")

            #print("xx.shape", xx.shape)
            #print("xx",xx)
            #print("yy.shape", yy.shape)
            #print("yy",yy)
            #print("Z.shape", Z.shape)
            #print("Z:", Z)
            ax.contourf(xx, yy, Z, 7, cmap=cm, alpha=.8)

            # Plot also the training points
            #ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='black', s=25)
            # and testing points
            #ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='black', s=25)

            #print("xx limit", xx.min(), xx.max())
            #print("yy limit", yy.min(), yy.max())
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            #ax.set_title(name)
            #ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
            i += 1

    figure.subplots_adjust(left=.00, right=1, top=1, bottom=.00)
    figure.set_size_inches(rst_h/100, rst_w/100)
    plt.savefig(resultFile, dpi=100)
    plt.show()


