from sklearn.model_selection import train_test_split
from src.data import load_data,preprocess
from src.model import train_model,evalute_model
from src.config import load_config
from src.logger import get_logger

logger=get_logger("main")
def main():
    cfg=load_config()
    logger.info("loadinf data from %s",cfg["data"]["path"])
    df=load_data(cfg["data"]["path"])
    X,y=preprocess(df)
    logger.info("data ready: %d rows,%d features",X.shape[0],X.shape[1])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    logger.info("traning model")
    model=train_model(X_train,y_train,max_iter=cfg["model"]["max_iter"])
    metrics=evalute_model(model,X_test,y_test)
    logger.info("metrics %s",metrics)
if __name__=="__main__":
    main()
