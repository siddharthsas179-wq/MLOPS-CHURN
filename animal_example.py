from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
x = [
    [25, 4, 5, 20],       # Cat(Height,weight,ear lenth(CM), leg length(cm))
    [23, 3, 5, 18],       # Cat
    [27, 5, 6, 21],       # Cat
    [60, 25, 12, 35],     # Dog
    [55, 20, 10, 30],     # Dog
    [65, 30, 13, 38],    # Dog
    [20, 2, 8, 12],       # Rabbit
    [18, 1.5, 7, 10],     # Rabbit
    [22, 2.5, 9, 13],     # Rabbit
]

y = [
    "Cat", "Cat", "Cat",
    "Dog", "Dog", "Dog",
    "Rabbit", "Rabbit", "Rabbit"
]

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.44, random_state=10, stratify=y
)
##print("Xtrain... " , X_train)
##print("Ytrain..... " , Y_train)
##print("Xtest..... " , X_test)
##print("Ytest..... " , Y_test)
model = DecisionTreeClassifier(random_state=10)
model.fit(X_train,Y_train)
print("model..", model)
preds=model.predict(X_test)
print("Preds...", preds)
accuracy_score(Y_test,preds)

new_animal=[[60,4,225,19]]

prediction= model.predict(new_animal)
print("New animal prediction : ", prediction[0])