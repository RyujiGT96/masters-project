import numpy as np
import pandas as pd
from tqdm import tqdm

# first column is a timestamp
# last column is a label (normal - 0 or fault - 1)
# output_width = 1

def model_in_out_generator(df, input_width=2, input_features=None, label_features=None, autoencoder=False):  
    
    for i in tqdm(range(len(df) - input_width)):
        if i == 0:
            # X
            if input_features is None:
                X = df.iloc[i:i+input_width, 1:-1].to_numpy()[np.newaxis, ...]
                    
            else:
                X = df[input_features].iloc[i:i+input_width].to_numpy()[np.newaxis, ...]
            
            # Y
            if label_features is None:
                Y = df.iloc[i+input_width, 1:-1].to_numpy()[np.newaxis, ...]
            else:
                Y = df[label_features].iloc[i+input_width].to_numpy()[np.newaxis, ...]
            
        else:
            # X
            if input_features is None:
                x_i = df.iloc[i:i+input_width, 1:-1].to_numpy()[np.newaxis, ...]
                X = np.concatenate((X, x_i), axis=0)
               
            else:
                x_i = df[input_features].iloc[i:i+input_width].to_numpy()[np.newaxis, ...]
                X = np.concatenate((X, x_i), axis=0)
               
            if label_features is None:

                y_i = df.iloc[i+input_width, 1:-1].to_numpy()[np.newaxis, ...]
                Y = np.concatenate((Y, y_i), axis=0)
            else:
                
                y_i = df[label_features].iloc[i+input_width].to_numpy()[np.newaxis, ...]
                Y = np.concatenate((Y, y_i), axis=0)
                
    X_timestamps = df.iloc[input_width-1:-1, 0]
    Y_timestamps = df.iloc[input_width:, 0]
    Y_label = df.iloc[input_width:, -1]
    
    if autoencoder:
        Y = X
        Y_timestamps = X_timestamps
        Y_label = df.iloc[input_width-1:-1, -1]
    
    print(f'X shape: {X.shape}')
    print(f'X_timestamps shape: {X_timestamps.shape}')
    print(f'Y shape: {Y.shape}')
    print(f'Y_timestamps shape: {Y_timestamps.shape}')
    print(f'Y_label shape: {Y_label.shape}')

    return X, X_timestamps, Y, Y_timestamps, Y_label
