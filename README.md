# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
```
step 1: start  
step 2:import the required libraries.
step 3:Upload and read the dataset.
step 4:Check for any null values using the isnull() function.
step 5:From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
step 6:Find the accuracy of the model and predict the required values by importing the required module from sklearn. 
step 7: end
```
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MONISH S
RegisterNumber:  212223040115
*/
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
head:

![Screenshot 2024-09-23 105346](https://github.com/user-attachments/assets/f82e006d-6279-4e6a-9572-c6b388bbc208)

accuracy:

![Screenshot 2024-09-23 105358](https://github.com/user-attachments/assets/fea9fea5-6521-4425-90ca-6448121ddb51)
![Screenshot 2024-09-23 105415](https://github.com/user-attachments/assets/371be8a7-904c-4323-95af-2b674565a299)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
