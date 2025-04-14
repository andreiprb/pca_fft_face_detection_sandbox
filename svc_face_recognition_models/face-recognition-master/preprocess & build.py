import os
import numpy as np
import cv2
import face_recognition

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_lfw_people

import pickle

if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('train_data'):
    os.makedirs('train_data')

"""-------------------------------------------------------------------------"""

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=1.0)
print("Data loaded from LFW dataset")
print("Number of people:", len(lfw_people.target_names))
print("Number of samples:", len(lfw_people.data))

labels = lfw_people.target_names
print("label count: ", len(labels))

def preprocess(img):
    # top, right, bottom, left
    (t, r, b, l) = face_recognition.face_locations(img)[0]
    # crop image
    face_img = img[t:b, l:r]
    # resize
    face_img = cv2.resize(face_img, (224, 224))
    # encode
    encode = face_recognition.face_encodings(face_img)[0]

    return encode

X = []
y = []

for i, face_img in enumerate(lfw_people.images):
    try:
        # Convert grayscale to RGB for face_recognition
        # Convert the float array to uint8 with correct range
        face_img_uint8 = (face_img * 255).astype(np.uint8)
        # Stack the same image across 3 channels to create RGB
        rgb_img = np.stack([face_img_uint8] * 3, axis=2)

        encode = preprocess(rgb_img)
        X.append(encode)
        y.append(lfw_people.target[i])

        if i % 50 == 0:
            print(f"Processed {i} images successfully")
    except Exception as e:
        print(e, ":", labels[lfw_people.target[i]])
        continue

X = np.asarray(X)
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# Save test data for later evaluation
np.save('train_data/X_test.npy', X_test)
np.save('train_data/y_test.npy', y_test)

"""-------------------------------------------------------------------------"""

# Train SVM model
print("Training SVM model...")
svc_model = svm.SVC(gamma='scale')
svc_model.fit(X_train, y_train)

## Train Accuracy
pred = svc_model.predict(X_train)
train_acc = accuracy_score(y_train, pred)
print("SVM Training Accuracy: ", train_acc)

## Test Accuracy
pred = svc_model.predict(X_test)
test_acc = accuracy_score(y_test, pred)
print("SVM Test Accuracy: ", test_acc)

model_name = 'models/svm-{}.model'.format(str(int(test_acc*100)))
pickle.dump(svc_model, open(model_name, 'wb'))

"""-------------------------------------------------------------------------"""

# Train KNN model
print("Training KNN model...")
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

## Train Accuracy
pred = knn_model.predict(X_train)
train_acc = accuracy_score(y_train, pred)
print("KNN Training Accuracy: ", train_acc)

## Test Accuracy
pred = knn_model.predict(X_test)
test_acc = accuracy_score(y_test, pred)
print("KNN Test Accuracy: ", test_acc)

model_name = 'models/knn-{}.model'.format(str(int(test_acc*100)))
pickle.dump(knn_model, open(model_name, 'wb'))

# Save labels
with open('models/labels.pickle', 'wb') as f:
    pickle.dump(labels, f)

print("All models and data saved successfully.")
