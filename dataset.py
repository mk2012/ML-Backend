import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib


df = pd.read_csv("titanic.csv")
df.isna().sum()
df['Age']= df['Age'].fillna(df['Age'].mean())
df.dropna(subset=['Embarked'],inplace=True)
df.isna().sum()
print(df.describe())
# sys.stdout.flush()

print(df.shape)
# sys.stdout.flush()
print("The number of columns present is as follows",df.columns.value_counts().sum())
# sys.stdout.flush()
print("The columns present in the actual dataset is as follows", df.columns.tolist())
# sys.stdout.flush()
cols = df.columns.tolist()
print("Visualising the dtypes",df.dtypes)
# sys.stdout.flush()
num_cols = df.select_dtypes([np.int64,np.float64]).columns.tolist()
num_cols.remove('PassengerId')
print(num_cols)
# sys.stdout.flush()
# #Generating Histograms for numeric columns
for col in num_cols:
    df.hist(column=col)
#     #Studying the correlation of the columns using scatter plots

scatter_matrix(df[num_cols],figsize=(50,50))
obj_cols = df.select_dtypes([object]).columns.tolist()
obj_cols.remove('Name')
obj_cols.remove('Cabin')
obj_cols.remove('Ticket')
print(obj_cols)
# sys.stdout.flush()
# #Plotting categorical data against frequency
for col in obj_cols:
    df[col].value_counts().plot(kind='bar')
    y = pd.Series(df['Survived'])
drop_list = ['Survived','Name','Ticket','Cabin']
X = df.drop(drop_list,axis=1)
encoder=ce.OneHotEncoder(handle_unknown='return_nan',return_df=True,use_cat_names=True)
X = encoder.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
model = RandomForestClassifier()
model.fit(X_train,y_train)

train_preds = model.predict(X_train)
print("Training scores are as follows")
# sys.stdout.flush()
print("Accuracy Score",accuracy_score(train_preds,y_train))
# sys.stdout.flush()
print("F1 Score",f1_score(train_preds,y_train))
# sys.stdout.flush()
print("ROC AUC Score",roc_auc_score(train_preds,y_train))
# sys.stdout.flush()


test_preds = model.predict(X_test)
print("Testing scores are as follows")
# sys.stdout.flush()
print("Accuracy Score",accuracy_score(test_preds,y_test))
# sys.stdout.flush()
print("F1 Score",f1_score(test_preds,y_test))
# sys.stdout.flush()
print("ROC AUC Score",roc_auc_score(test_preds,y_test))
# sys.stdout.flush()
joblib.dump(model,"model_joblib")
#Testing
loaded_model = joblib.load("model_joblib")
array = [5,3,1.0,0.0,35.0,0,0,8.0500,1.0,0.0,0.0] 
#each value represents a feature present in the training set Hint: the users should be able to enter their own values/(or) select from a drop down list of values to make custom predictions
a = np.asarray(array).reshape(1,-1)
predicted_value= loaded_model.predict(a)
actual_value = y[4]
print("Actual Value",actual_value)
print("Predicted Value",predicted_value)
sys.stdout.flush()