# from sklearn.datasets import make_multilabel_classification

# X, y = make_multilabel_classification(sparse = True, n_labels = 20,
# return_indicator = 'sparse', allow_unlabeled = False)

# #print X
# print y

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                      allow_unlabeled=True,
                                      random_state=1)
print X
print Y
# print type(Y)
classif = RandomForestClassifier()
classif.fit(X, Y)

# plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca")
# plot_subfigure(X, Y, 4, "Without unlabeled samples + PCA", "pca")

# plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
# plt.show()