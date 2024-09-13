# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively. 
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: MAHALAKSHMI R
RegisterNumber:  212223230116
*/
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) #removes the specified row or column.
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1['status']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size =0.2,random_sta

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)


lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
PLACEMENT DATA:
![Screenshot 2024-09-13 142806](https://github.com/user-attachments/assets/b11efb6f-e450-405e-ac1f-da56f103110f)


SALARY DATA:
![Screenshot 2024-09-13 143119](https://github.com/user-attachments/assets/c493e531-ea8e-4358-8764-b562bcfcdafb)


CHECKING THE NULL() FUNCTION:
![Screenshot 2024-09-13 143134](https://github.com/user-attachments/assets/00592eb5-c564-47e3-9596-01d8b7268026)


DATA DUPLICATE:
![Screenshot 2024-09-13 143141](https://github.com/user-attachments/assets/423d4d97-f630-4efa-84f3-5884c3306221)


PRINT DATA:
![Screenshot 2024-09-13 143236](https://github.com/user-attachments/assets/12abc52b-a6b2-4663-992b-2451935d4dba)


DATA STATUS:
![Screenshot 2024-09-13 143246](https://github.com/user-attachments/assets/041463e6-5168-490b-bbb3-a40828f67610)


DATA STATUS:
![Screenshot 2024-09-13 143246](https://github.com/user-attachments/assets/0a22d0be-6050-4981-9b59-a5a615614992)


Y PREDICTION ARRAY:
![Screenshot 2024-09-13 143303](https://github.com/user-attachments/assets/758558fa-33d3-471e-8777-138ece93d804)


ACCURACY VALUE:
![Screenshot 2024-09-13 143356](https://github.com/user-attachments/assets/0f617052-241c-471a-b8dc-2bda50f8d076)


CONFUSION ARRAY:
![Screenshot 2024-09-13 143400](https://github.com/user-attachments/assets/bd3ea8a9-ab25-4414-b211-75d7ce3c3f61)


CLASSIFICATION REPORT:
![Screenshot 2024-09-13 143410](https://github.com/user-attachments/assets/fe94576f-d034-4835-b5bb-9fb6a7913dc3)


PREDICTION OF LR:
![Screenshot 2024-09-13 143425](https://github.com/user-attachments/assets/89d5d680-4d08-49e4-95e3-39c269f1d3c6)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
