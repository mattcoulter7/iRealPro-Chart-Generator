
import math
import warnings
import os
import numpy as np
import sklearn.metrics as metrics
from data import iRealDataProcessor
from keras.models import load_model
warnings.filterwarnings("ignore")

def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = metrics.mean_absolute_percentage_error(y_true,y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)

def main():
    file = os.path.join(os.path.dirname(__file__),'jazz1400.txt')

    dp = iRealDataProcessor(file,4,'C',0.3).generate_training_data()

    dp.X_test = np.reshape(dp.X_test, (dp.X_test.shape[0], dp.X_test.shape[1], 1))

    model = load_model(os.path.join(os.path.dirname(__file__),'model',f'gru.h5'))
    y_pred = model.predict(dp.X_test)

    eva_regress(dp.y_test,y_pred)

if __name__ == '__main__':
    main()
