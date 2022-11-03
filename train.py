import sys
import warnings
import os
import numpy as np
import pandas as pd

from data import iRealDataProcessor
from model import get_gru

warnings.filterwarnings("ignore")

def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save(os.path.join(os.path.dirname(__file__),'model',f'{name}.h5'))
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(os.path.join(os.path.dirname(__file__),'model',f'{name} loss.csv'), encoding='utf-8', index=False)

def main(argv):
    config = {"batch": 128, "epochs": 200}

    file = os.path.join(os.path.dirname(__file__),'jazz1400.txt')
    dp = iRealDataProcessor(file,4,'C',0.3).generate_training_data()

    dp.X_train = np.reshape(dp.X_train, (dp.X_train.shape[0], dp.X_train.shape[1], 1))
    m = get_gru([(dp.lag * 2) + 2, 400, 400, 400, dp.output_size])
    train_model(m, dp.X_train, dp.y_train, 'gru', config)

if __name__ == '__main__':
    main(sys.argv)
