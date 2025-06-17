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
                    level=logging.INFO)

def main(data, eta, epochs, filename, plotFileName):


    df = pd.DataFrame(data)

    logging.info(f" NAND data frame{df}")

    X,y = prepare_data(df)


    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()
    save_model(model, filename= filename)
    save_plot(df, plotFileName, model)

if __name__ == "__main__":
    NAND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [1,1,1,0],
    }
    
    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    try:
        logging.info(">>>>>>Starting NAND gate model training...>>>>>>>>")
        main(data =NAND, eta=0.3, epochs=10, filename="nand.model", plotFileName="nand.png")
        logging.info("<<<<<<<<<<NAND gate model training completed successfully.<<<<<<< \n")
    except Exception as e :
        logging.exception(e)
        raise e
    

