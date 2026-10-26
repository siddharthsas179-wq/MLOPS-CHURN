from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split

def train_model(X_train,y_train,max_iter:int=2000):
    model=LogisticRegression(max_iter=max_iter)
    model.fit(X_train,y_train)
    return model

def evalute_model(model,X_test,y_test) -> dict:
    preds = model.predict(X_test)
    return{
        "accuracy":round(accuracy_score(y_test,preds),3), "f1":round(f1_score(y_test,preds),3)
    }