from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LSTM, GRU
from sklearn.model_selection import train_test_split
import pickle
import argparse
import numpy as np

def load_pkl(path):
    pickle_in = open(path, "rb")
    print('Loaded ', path)
    return pickle.load(pickle_in)


def get_compiled_model(input_dim):
        #model using embeddings
        model = Sequential([
                Dense(512, activation='relu', input_dim=input_dim),
                Dense(256, activation='relu'),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
        model.compile(optimizer='rmsprop',
                loss='mean_squared_error',
                #lr=0.01,
                metrics=['mse'])

        return model


def filter_duration(X, y, df_filtered):
    b = y < 300
    return X[b][:], y[b], df_filtered.iloc[b]

def main(args):
    pkl_path = args.pkl_path


    #every ith element in X correspond to ith element in df_filtered
    df_filtered, X, y = load_pkl(pkl_path)
    
    X, y, df_filtered = filter_duration(X, y, df_filtered)
    max_duration = np.max(y)

    #normalize duration
    y =  y / max_duration


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    input_dim = np.size(X, axis=1)

    #CREATE MODEL
    model = get_compiled_model(input_dim)
    print(model.summary())

    #TRAIN MODEL
    model.fit(x=X_train, y=y_train, validation_split=0.2, epochs=15, batch_size=128)

    #EVAL MODEL
    model.evaluate(X_test, y_test, verbose=1)


    #create model files to load an predict from
    preds = model.predict(y_test)
    print(preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pkl_path', action='store', dest='pkl_path', help='path to pkl file')
    args = parser.parse_args()

    main(args)