import pandas as pd
import jsonschema
from pydantic import BaseModel
from jsonschema import validate
from fastapi.encoders import jsonable_encoder



class request_body(BaseModel):
    id: int    
    asset_scores: dict
    asset_infos: dict


class request_body_full(BaseModel):
    data_train_assets: list[request_body]




json_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "number"},

        "asset_infos": {"properties":

                            {'var1': {"type": "number"},
                             'var10': {"type": "number"},
                             'var2': {"type": "number"},
                             'var3': {"type": "number"},
                             'var4': {"type": "string"},
                             'var5': {"type": "number"},
                             'var6': {"type": "string"},
                             'var7': {"type": "number"},
                             'var8': {"type": "string"},
                             'var9': {"type": "string"}}},

        "asset_scores": {"properties":

                             {'score1': {"type": "number"},
                              'score2': {"type": "number"},
                              'score3': {"type": "number"},
                              'score4': {"type": "number"},
                              'score5': {"type": "number"},
                              'score6': {"type": "number"},
                              'score7': {"type": "number"},
                              'score8': {"type": "number"},
                              'score9': {"type": "number"},
                              'score10': {"type": "number"},
                              'score11': {"type": "number"},
                              'score12': {"type": "number"},
                              'score13': {"type": "number"},
                              'score14': {"type": "number"}}},

    }
}


def validate_features(json_data: str) -> bool:
    """ Check if the sent json has the same features
    :param json_data: json file 
    :return: bool 
    """
    asset_info_keys = {key for key in dict(json_data['asset_infos']).keys()}
    asset_scores_keys = {key for key in dict(json_data['asset_scores']).keys()}
    info_variables = {'var7', 'var3', 'var10', 'var1', 'var2', 'var5', 'var9', 'var6', 'var8', 'var4'}
    scores_variables = {'score13', 'score10', 'score8', 'score5', 'score7', 'score3', 'score6', 'score1', 'score11',
                        'score12', 'score9', 'score4', 'score2', 'score14'}
    try:
        assert json_data['id'] is not None and isinstance(json_data['id'], int)
        assert info_variables == asset_info_keys
        assert scores_variables == asset_scores_keys
    except Exception as err:
        print('Please ensure that we have the same features')
        return False
    return True


def validate_type(json_data: str) -> bool:
    """ Check if the sent json has the same type of the features
    :param json_data: json file 
    :return: bool 
    """
    try:
        validate(instance=json_data, schema=json_schema)
    except jsonschema.exceptions.ValidationError as err:
        print(f'We could accept the json because of the following erreor: {err}')
        return False
    return True



def check_json(json_data):
    """ Check if the json file is acceptable
    :param json_data: 
    :return: 
    """
    if validate_type(json_data) and validate_features(json_data):
        return True
    return False


def deserialize_json(json_data: request_body_full):
    """Deserialize the input json to build the dataFrame to put in the statistical model
    :param json_data: 
    :return: 
    """
    training_set = []

    data = jsonable_encoder(json_data)
    list_data = data['data_train_assets']

    for elm in list_data:
        id_ = {'id':elm['id']}
        asset_infos = elm['asset_infos']
        asset_scores = elm['asset_scores']
        features = id_ | asset_infos | asset_scores
        training_set.append(features)   
    
    return pd.DataFrame(training_set)
