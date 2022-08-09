import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.model_selection
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style

# Reading the data
data = pd.read_csv("student-mat.csv", sep=";")

# seperating the necessary attributes from the data set.
data = data[["age", 'Medu', 'studytime', 'G1', 'G2', "G3", 'absences']]

# entering the variable to predict
predict = "G3"

x_est = np.array(data.drop([predict], 1))
y_est = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_est, y_est, test_size=0.10)
# spliting the data into training set and testing set
"""acc_score = 0

for i in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_set, y_set, test_size=0.10)

    model = linear_model.LinearRegression()
    model.fit(x_train,y_train)
    acc = model.score(x_test,y_test)
    #print(acc)
    if acc > acc_score:
        acc_score = acc

        with open("studentmodel.pickle","wb") as f:
            pickle.dump(model,f)
print(acc_score)"""

pickle_file =  open("studentmodel.pickle","rb")


model = pickle.load(pickle_file)
predicitons = model.predict(x_test)

for x in range(len(predicitons)):
    print("predicting data : ",x_test[x],"Original : ",y_test[x],"||","Predicted number ", round(predicitons[x]))

# plotting the points in the graph

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("FINAL GRADE")
pyplot.show()