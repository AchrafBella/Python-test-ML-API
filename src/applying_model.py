import pandas as pd
import pickle
import os


def prepare_testData(df: pd.DataFrame()):
    """Cleaning, splitting the testing data and apply the same training dumification to this data."""  
    # convert the boolean features to numericals
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

    # extract features 
    X = df.drop(['id'], axis=1)

    # encoder categorical varaibles
    loaded_encoder = pickle.load(open('encoder.sav', 'rb'))
    X = loaded_encoder.transform(X)
    return X



def apply_latestStatModel(X: pd.DataFrame()):
    """Application of the last statistical model saved to the test base"""
    models_dir = list(os.listdir('trainned_models'))
    # to retrieve the latest ML model
    filename = os.path.join('trainned_models', models_dir[-1])
    loaded_model = pickle.load(open(filename, 'rb'))
    predictions = loaded_model.predict(X.values)
    return predictions
