import numpy as np
from sklearn import metrics as metrics


def cnn_prediction(model, X_train, X_test, y_train, y_test):
    y_preds = model.predict(X_test)
    r2 = metrics.r2_score(y_test, y_preds)
    mse = metrics.mean_squared_error(y_test, y_preds)
    rmsep = np.sqrt(mse)

    [_, mse] = model.evaluate(X_train, y_train, verbose=0)
    rmsec = np.sqrt(mse)
    print(f'RMSEP: {rmsep:.3f} - R2: {r2:.3f} - Ratio: {rmsep/rmsec:.3f}')
    print(f'RMSEC: {rmsec:.3f}')

    mse = metrics.mean_squared_error(y_test, y_preds)
    print(f'Scaled RMSEP {np.sqrt(mse):.3f}')

    return rmsep, r2, y_preds
