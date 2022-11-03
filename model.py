from keras.layers import Dense, Dropout, Activation,InputLayer
from keras.layers import LSTM, GRU, SimpleRNN
from keras.models import Sequential

def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """
    model = Sequential()
    for i in range(1,len(units)):
        if i == 1:
            model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True, reset_after=True))
        elif i == len(units) - 1:
            model.add(Dropout(0.2))
            model.add(Dense(units[i], activation='sigmoid'))
        elif i < len(units) - 2:
            model.add(GRU(units[i], return_sequences=True))
        else:
            model.add(GRU(units[i]))

    return model