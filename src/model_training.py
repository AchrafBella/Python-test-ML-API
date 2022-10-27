import pickle
import optuna
import pandas as pd
import numpy as np
import category_encoders as ce

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from optuna.samplers import TPESampler # Samplers class that defines the hyper-parameter space
from datetime import datetime


# Define an objective function to be minimized/maximized.
def objective(trial, x_train, y_train, x_test, y_test):
    # Number of trees in random forest
    et_n_estimators = trial.suggest_int('n_estimators', 10, 1000)
    # Number of features to consider at every split
    et_max_features = trial.suggest_int('max_features', 1, 13)
    # Maximum number of levels in tree
    et_max_depth = trial.suggest_int('max_depth', 2, 50)
    # Minimum number of samples required at each leaf node
    et_min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 50)
    # Minimum number of samples required to split a node
    et_min_samples_split = trial.suggest_int('min_samples_split', 2, 50)

    classifier_obj = ExtraTreesRegressor(max_depth = et_max_depth,
                                        max_features = et_max_features,
                                        min_samples_split = et_min_samples_split,
                                        min_samples_leaf = et_min_samples_leaf,
                                        n_estimators = et_n_estimators,
                                        n_jobs=-1)

    # training 
    classifier_obj.fit(x_train, y_train)
    predictions = classifier_obj.predict(x_test)
        
    return mean_squared_error(y_test, predictions)  # An objective value linked with the Trial object.


def clean_trainData(df: pd.DataFrame(), save_encoder: bool, test_size: float) -> pd.DataFrame():
    """Cleaning, splitting the training data and dummies the cateogial variables"""
    
    # convert the boolean features 
    df.replace('YES', 1, inplace=True)
    df.replace('NO', 0, inplace=True)

    # features selection (reduction of dimension)
    scores = ['score1', 'score2', 'score3', 'score4',
       'score5', 'score6', 'score7', 'score8', 'score9', 'score10', 'score11',
       'score12', 'score13', 'score14']

    df['min_scores'] = df[scores].min(axis=1)
    df['max_scores'] = df[scores].max(axis=1)
    df['avrg_scores'] = df[scores].mean(axis=1)
    
    df.drop(scores, axis=1, inplace=True)
    
    # extract target & features
    X = df.drop(['id', 'target'], axis=1)
    Y = df['target']
    
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=test_size,
                                                    random_state=0)


    # convert the categorical variables
    encoder = ce.cat_boost.CatBoostEncoder(random_state=0)
    x_train = encoder.fit_transform(X_train, y_train)
    x_test = encoder.transform(X_test)
    # save the encoder for later use
    if save_encoder:
        pickle.dump(encoder, open('encoder.sav', 'wb'))

    return x_train, x_test, y_train, y_test


def train_model(df, x_train, y_train, x_test, y_test, epochs):
    """Apply the model to the train data and the train target by turning multiple parameters"""
    # Define an objective function to be minimized/maximized.
    func = lambda trial: objective(trial, x_train, y_train, x_test, y_test)
    study = optuna.create_study(directions=['minimize'], sampler=TPESampler())  # Create a new study.
    study.optimize(func, n_trials=epochs)  # Invoke optimization of the objective function.

    params = dict(study.best_params)

    model = ExtraTreesRegressor(**{
        'max_depth': params['max_depth'],
        'max_features': params['max_features'], 
        'min_samples_leaf': params['min_samples_leaf'],
        'min_samples_split': params['min_samples_split'],
        'n_estimators': params['n_estimators']}, 
        n_jobs=-1).fit(x_train, y_train)

    return model

    

def get_parameters(model, x_test, y_test):
    """Calculate statistical parameters of the model (EX : RMSE)"""
    predictions = model.predict(x_test)
    # scores
    model_MAE = mean_absolute_error(y_test, predictions)
    model_MSE = mean_squared_error(y_test,predictions)
    model_RMSE = np.sqrt(mean_squared_error(y_test, predictions))
    model_R2 = explained_variance_score(y_test, predictions)

    scores = {'MAE': model_MAE,
               'MSE': model_MSE,
               'RMSE': model_RMSE,
               'R2': model_R2}
    return scores


def save_model(model):
    """Save the statistical model in a file"""
    date = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    filename = f'trainned_models/model_{date}.sav'
    pickle.dump(model, open(filename, 'wb'))
