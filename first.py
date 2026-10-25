from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

name="Siddharth"
expirence=13
def greet(person,years):
    return f"Hello {person}, {years} years in CMS, now learning MLOPS"
print(greet(name,expirence))
name1="ravi"
experince1=10
def add(name1,experince1):
    return name1 + experince1
print(add(5,7))

x, y = load_iris(return_X_y=True)
##print(x[:3])
##print(y[:3])
##print(x.shape)
##print(y.shape)
X_train, X_test, Y_train , Y_test=train_test_split(x,y,test_size=0.5,random_state=42)
model=LogisticRegression(max_iter=200)
model.fit(X_train,Y_train)
preds=model.predict(X_test)
print("Accuracy", accuracy_score(Y_test,preds))
