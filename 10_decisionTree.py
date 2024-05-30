# play tennis 의사결정 트리
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pydotplus    
import os
tennis_data = pd.read_csv('tennis.csv')
print(tennis_data)



tennis_data.outlook = tennis_data.outlook.replace('sunny',0)
tennis_data.outlook = tennis_data.outlook.replace('overcast', 1)
tennis_data.outlook = tennis_data.outlook.replace('rainy',2)

tennis_data.temp = tennis_data.temp.replace('hot',3)
tennis_data.temp = tennis_data.temp.replace('mild',4)
tennis_data.temp = tennis_data.temp.replace('cool',5)

tennis_data.humidity = tennis_data.humidity.replace('high',6)
tennis_data.humidity = tennis_data.humidity.replace('normal',7)

tennis_data.windy = tennis_data.windy.replace(False,8)
tennis_data.windy = tennis_data.windy.replace(True,9)

tennis_data.play = tennis_data.play.replace('no',10)
tennis_data.play = tennis_data.play.replace('yes',11)

print(tennis_data)

X = np.array(pd.DataFrame(tennis_data, columns=['outlook','temp', 'humidity', 'windy']))
y = np.array(pd.DataFrame(tennis_data, columns=['play']))
X_train, X_test, y_train, y_test = train_test_split(X,y)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

dt_clf = DecisionTreeClassifier()
dt_clf = dt_clf.fit(X_train, y_train)
dt_prediction = dt_clf.predict(X_test)


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
feature_names = tennis_data.columns.tolist()
feature_names = feature_names[0:4]
target_name = np.array(['Play No', 'Play Yes'])


dt_dot_data = tree.export_graphviz(dt_clf, out_file = None,
                                  feature_names = feature_names,
                                  class_names = target_name,
                                  filled = True, rounded = True,
                                  special_characters = True)

dt_graph = pydotplus.graph_from_dot_data(dt_dot_data)

# 그래프를 PNG 이미지로 저장
dt_graph.write_png("decision_tree.png")

# matplotlib를 사용하여 이미지를 표시
img = mpimg.imread("decision_tree.png")
plt.figure(figsize=(20, 20))
plt.imshow(img)
plt.axis('off')
plt.show()
