import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

current_dir = os.getcwd()

DATADIR = current_dir + '/dataset'
CATEGORIES = ['orange','violet','red','blue','green','black','brown','white']
IMG_SIZE=100

training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR, category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()

lenofimage = len(training_data)

X=[]
y=[]

for categories, label in training_data:
    X.append(categories)
    y.append(label)
    
X= np.array(X).reshape(lenofimage, -1)
X = X/255.0

y=np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

svc = SVC(kernel='linear',gamma='auto')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print("Accuracy on unknown data is", accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))