import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import precision_recall_curve
import joblib
#Load Dataset
df=pd.read_csv("Churn_Modelling.csv")
print(df.head(n=7))
print(df.shape)
print(df.dtypes)
#Identify nulls
print(df.isna().sum())
#Feature Engineering
df['Gender']=df['Gender'].map(
    {
        "Female":0,
        "Male":1
    }
).astype(int)
print(df.dtypes)
print(df.head(n=7))
#Balance Flag
df['Balance_flag']=(df['Balance']==0).astype(int)
print(df[['Balance','Balance_flag']].head())
#Balance to Salary ratio
df['Balance_salary_ratio']=df['Balance']/df['EstimatedSalary']
print(df[['Balance','EstimatedSalary','Balance_salary_ratio']].head(n=10))
df['Balance_salary_ratio']=np.log1p(df['Balance_salary_ratio'])
#Age Group
df['Age_group'] = pd.cut(
    df['Age'],
    bins=[0, 24, 59, 120],
    labels=['Young Adults', 'Middle aged', 'Senior']
)
print(df[['Age','Age_group']].head(n=20))
#Credit-age score ratio
df["CreditScore_Age_ratio"] = df["CreditScore"] / df["Age"]
#Tenure-age ratio
df["Tenure_Age_ratio"] = df["Tenure"] / df["Age"]
df["IsActive_NumOfProducts"] = df["IsActiveMember"] * df["NumOfProducts"]
#Initialize features and target
x=df.drop(columns=['CustomerId','RowNumber','Exited','Surname'])
y=df['Exited']
categorical_cols=['Age_group','Geography']
#Train Test split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.20,random_state=42,stratify=y
)

#Build and Train model
params={
   "n_estimators":[200,300,400],
   "learning_rate":[0.1,0.2,0.3],
   "max_depth":[2,4,6],
   "auto_class_weights":["Balanced"],
   "cat_features":[categorical_cols]
}
model=CatBoostClassifier(verbose=0)
grid=GridSearchCV(estimator=model,param_grid=params,scoring='f1',cv=5)  
grid.fit(x_train,y_train)
#Choose the best threshold
x_tr,x_val,y_tr,y_val=train_test_split(x_train,y_train,test_size=0.25,random_state=42,stratify=y_train)
val_model=CatBoostClassifier(**grid.best_params_,verbose=0)
val_model.fit(x_tr,y_tr)
proba_val=val_model.predict_proba(x_val)[:,1]
prec,rec,thr=precision_recall_curve(y_val,proba_val)
f1=2*prec*rec/(prec+rec+1e-9)
best_thr=thr[np.argmax(f1[:-1])]
print("Best threshold:",best_thr)
best_model=grid.best_estimator_
proba=best_model.predict_proba(x_test)[:,1]
pred=(proba>=best_thr).astype(int)
print("Classification Report:",classification_report(y_test,pred))
print("Confusion matrix:",confusion_matrix(y_test,pred))
print("ROC-AUC:",roc_auc_score(y_test,proba))

#  Save the best trained CatBoost model
best_model = grid.best_estimator_
best_model.save_model("customer_churn_model.pkl")

#  Export cleaned dataset 
df.to_csv("Churn_cleaned.csv", index=False)