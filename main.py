from sklearn.model_selection import train_test_split
from src.data import load_data,preprocess
from src.model import train_model,evalute_model

DATA_PATH=r"C:\Users\ssrsi\OneDrive\Desktop\MLOPS\python_project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df=load_data(DATA_PATH)
X,y=preprocess(df)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model=train_model(X_train,y_train)
metrics=evalute_model(model,X_test,y_test)
print(metrics)