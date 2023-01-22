from hyperopt import fmin,tpe,hp
import hyperopt
from hyperopt import hp,STATUS_OK,fmin,tpe,Trials
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd




x_train = pd.read_excel("数据.xlsx").values[1:3133,0:8]
y_train = pd.read_excel("数据.xlsx").values[1:3133,8]
x_test = pd.read_excel("数据.xlsx").values[3133:,0:8]
y_test = pd.read_excel("数据.xlsx").values[3133:,8]

epoch = 0
accu = []
prec = []
reca = []
f1_sc = []

space = {"C":hp.loguniform("C",np.log(1),np.log(100)),
         "kernel":hp.choice("kernel",["rbf","poly","sigmoid"]),
         "gamma":hp.loguniform("gamma",np.log(0.0001),np.log(0.1))
         }

def hyperopt_param(args):
    clf = svm.SVC(**args)
    clf.fit(x_train,y_train)
    predict = clf.predict(x_test)
    global epoch
    epoch = epoch+1
    accuracy = accuracy_score(y_test,predict)

    print("*"*20)
    print(args)
    print("第{}次测试。".format(epoch))
    print("准确率={}".format(accuracy))
    print("*"*20)
    global accu
    accu.append(accuracy)


    return -accuracy

result = fmin(fn=hyperopt_param,space=space,algo=tpe.suggest,max_evals=500)
print(result)

plt.plot(accu)
plt.show()
