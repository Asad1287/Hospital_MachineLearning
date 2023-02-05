import numpy as np

def rmse(y_true, y_pred):
    """
    rmse calcuate the root mean squared error for a given true and predicted values
    
    """
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_true, y_pred))

def split_data(X, y, test_size=0.2, random_state=42):
    """
    split data into train and test sets with a given test size and random state
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def print_scores(model,X_train, X_test, y_train, y_test):
    """
    print_scores prints the rmse and r2 scores for a given model and train and test sets
    """
    res = [rmse(model.predict(X_train), y_train),rmse(model.predict(X_test), y_test),
              model.score(X_train, y_train), model.score(X_test, y_test)]
    if hasattr(model, 'oob_score_'): res.append(model.oob_score_)
    print(res)


