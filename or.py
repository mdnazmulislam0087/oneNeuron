from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np

Or = {
    'x1': [0, 0, 1, 1],
    'x2': [0, 1, 0, 1],
    'y': [0, 1, 1, 1],
}
df = pd.DataFrame(Or)
print(df)

X, y = prepare_data(df)
ETA = 0.3  # 0 and 1
EPOCHS = 10

model_Or = Perceptron(eta=ETA, epochs=EPOCHS)
model_Or.fit(X, y)
_ = model_Or.total_loss()

save_model(model_Or, filename="or.model")
save_plot(df, "or.png", model_Or)

