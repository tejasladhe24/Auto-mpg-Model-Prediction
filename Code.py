import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
from sklearn.metrics import confusion_matrix,roc_curve,auc 
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.model_selection import cross_val_score


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


data = pd.read_csv('dataset.csv', delimiter=",")

data.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                   'acceleration', 'model year', 'origin', 'car name']

#print('Shape of the dataset: ' + str(data.shape))
#print(data.head())

#define classification

vehicle_fuel_economy_class = []
for i in range(len(data['mpg'])):
    if data['mpg'][i]<12: 
        vehicle_fuel_economy_class.append('Extremely low fuel efficient')
    elif data['mpg'][i]>=12  and data['mpg'][i]<25:
         vehicle_fuel_economy_class.append('Low fuel efficient')
    elif data['mpg'][i]>=25  and data['mpg'][i]<34:
         vehicle_fuel_economy_class.append('Intermediate fuel efficient')
    elif data['mpg'][i]>=34  and data['mpg'][i]<42:
         vehicle_fuel_economy_class.append('High fuel efficient')
    else : vehicle_fuel_economy_class.append('Extremely high fuel efficient')


def ROC():
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(y_predicted))[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    lw=2
    plt.figure(figsize=(8,5))
    plt.plot(fpr["macro"], tpr["macro"],
             label='AUC (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='green', linestyle=':', linewidth=4)
    
    '''
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    '''
    plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

data['Ecomnoy Class']=vehicle_fuel_economy_class

X = data.iloc[:,1:6].values
y = data.iloc[:,9].values

#print(y)


# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 2) 

logR= LogisticRegression()
logR.fit(X_train, y_train)

print('''____________________________________________________________________
                                 Logistic Regression''')
print('Logistic Regression Score = ', logR.score(X_test, y_test),'\n For K=3, K-Fold Cross Validation Score = ', cross_val_score(LogisticRegression(),X ,y).mean(), '\n\n Confusion_Matrix')

y_predicted=logR.predict(X_test)

cm= confusion_matrix(y_test, y_predicted)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('CM Logistic Regression.png')
plt.show()


n_classes = 2
ROC()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Naive Bayes')
plt.legend(loc="lower right")
plt.savefig('ROC Logistic Regression.png')
plt.show()

###############################################################################
# training a linear SVM classifier 

svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
y_predicted = svm_model_linear.predict(X_test) 

# model accuracy for X_test 

print('''____________________________________________________________________
                                 SVM classifier''')
print('SVM classifier Score = ', svm_model_linear.score(X_test, y_test),'\n For K=3, K-Fold Cross Validation Score = ', cross_val_score(SVC(),X ,y).mean(), '\n\n Confusion_Matrix')

# creating a confusion matrix 
cm = confusion_matrix(y_test, y_predicted) 
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('CM SVM.png')
plt.show()

n_classes = 5
ROC()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Naive Bayes')
plt.legend(loc="lower right")
plt.savefig('ROC SVM.png')
plt.show()

###############################################################################
# training a Gaussian Niave Bayes Classifier

GauNB= GaussianNB()
GauNB.fit(X_train, y_train)

# model accuracy for X_test 
y_predicted= GauNB.predict(X_test)

print('''____________________________________________________________________
                          Gaussian Niave Bayes Classifier''')
print('Bernoulli Niave Bayes Classifier Score = ', GauNB.score(X_test, y_test),'\n For K=3, K-Fold Cross Validation Score = ', cross_val_score(GaussianNB(),X ,y).mean(), '\n\n Confusion_Matrix')

# creating a confusion matrix 
cm = confusion_matrix(y_test, y_predicted) 
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('CM Gaussian Niave Bayes.png')
plt.show()

n_classes = 5
ROC()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('ROC Gaussian Niave Bayes.png')
plt.show()
###############################################################################
# training a K Nearest Neighbbors Classifier

KNN= neighbors.KNeighborsClassifier()
KNN.fit(X_train, y_train)

# model accuracy for X_test 
y_predicted= KNN.predict(X_test)

print('''____________________________________________________________________
                          K Nearest Neighbbors Classifier''')
print(' K Nearest Neighbbors Classifier Score = ', KNN.score(X_test, y_test), '\n For K=3, K-Fold Cross Validation Score = ', cross_val_score(neighbors.KNeighborsClassifier(),X ,y).mean(), '\n\n Confusion_Matrix')

# creating a confusion matrix 
cm = confusion_matrix(y_test, y_predicted) 
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('CM K Nearest Neighbbors.png')
plt.show()

n_classes = 3
ROC()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for K Nearest Neighbours')
plt.legend(loc="lower right")
plt.savefig('ROC K Nearest Neighbbors.png')
plt.show()

###############################################################################
# training a Random Forrest Classifier

ranF= RandomForestClassifier(n_estimators=15)
ranF.fit(X_train, y_train)

# model accuracy for X_test 
y_predicted= ranF.predict(X_test)

print('''____________________________________________________________________
                          Random Forrest Classifier''')

print('Random Forrest Classifier Score= ',  ranF.score(X_test, y_test),'\n For K=3, K-Fold Cross Validation Score = ', cross_val_score(RandomForestClassifier(n_estimators=15),X ,y).mean(), '\n\n Confusion_Matrix')

# creating a confusion matrix 
cm = confusion_matrix(y_test, y_predicted) 
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('CM Random Forrest.png')
plt.show()

n_classes = 4
ROC()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Random Forest')
plt.legend(loc="lower right")
plt.savefig('ROC Random Forrest.png')
plt.show()
###############################################################################