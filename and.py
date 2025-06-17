from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model,save_plot
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s :  %(levelname)s : %(module)s] : %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
logging.basicConfig(filename=os.path.join(log_dir, 'running_log.log'),
    filemode='a',  # 'w' to overwrite the log file, 'a' to append
    format=logging_str,
    level=logging.INFO,
    
)


def main(data, eta, epochs, filename, plotFileName):


    df = pd.DataFrame(data)

    logging.info(f"This is and Dataframe {df}")

    X,y = prepare_data(df)


    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()
    save_model(model, filename= filename)
    save_plot(df, plotFileName, model)

if __name__ == "__main__":
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    try:
        logging.info(">>>>>>Starting AND gate model training...>>>>>>>>")
        main(data =AND, eta=0.3, epochs=10, filename="and.model", plotFileName="and.png")
        logging.info("<<<<<<<<<<AND gate model training completed successfully.<<<<<<< \n")
    except Exception as e :
        logging.exception(e)
        raise e
