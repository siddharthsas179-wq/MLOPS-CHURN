import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score,f1_score, classification_report
import numpy as np

df=pd.read_csv(r"C:\Users\ssrsi\OneDrive\Desktop\MLOPS\python_project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce").fillna(0)

df=df.drop(columns=["customerID"])

df["Churn"] = (df["Churn"]== "Yes").astype(int)
df=pd.get_dummies(df,drop_first=True)

X=df.drop(columns=["Churn"])
Y=df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=43, stratify=Y)

model=LogisticRegression(max_iter=2000)
model.fit(X_train,y_train)

preds=model.predict(X_test)

print("Accuracy", round(accuracy_score(y_test,preds),3))
print("F1:",round(f1_score(y_test,preds),3))
print("hello",classification_report(y_test,preds))

coefs=pd.Series(model.coef_[0], index=X.columns).sort_values()
print("pishes towards staying\n",coefs.head(5))
print("puishes towards leaving\n",coefs.tail(5))