import pandas as pd

def load_data(path:str) ->pd.DataFrame: 
     df=pd.read_csv(path)
     df["TotalCharges"]=pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
     return df
def preprocess(df:pd.DataFrame):
     df=df.drop(columns=["customerID"])
     df["Churn"]=(df["Churn"]=="Yes").astype(int)
     df=pd.get_dummies(df, drop_first=True)
     X=df.drop(columns=["Churn"])
     y=df["Churn"]
     return X,y
