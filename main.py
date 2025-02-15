import os
import pickle

from skimage.io import imread
from skimage.transform import resize

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#prepare the dataset

input_dir = 'data'
categories = ['empty', 'not_empty']

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):

        # get image from path
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)

        #resize image
        img = resize(img, (15, 15))

        #save it in matrix
        data.append(img.flatten())

        #save the labels
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)


# train test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, shuffle=True)

#classify
classifier = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C':[1,10,100,1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(X_train, y_train)

#test performance

best_estimator = grid_search.best_estimator_
y_pred = best_estimator.predict(X_test)

score = accuracy_score(y_pred,y_test)
print(score)

pickle.dump(best_estimator, open('./model.p','wb'))