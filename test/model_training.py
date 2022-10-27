# QUESTIONS : 
    # Développement d'un service capable d'entrainer un modèle statistique de prédiction sur la base de train (les informations nécessaires doivent être comme des paramètres des fonnctions).
    # Transformation de ce service en API (port 6000)

# NB : 
    # Vous pouvez ajouter d'autres fonctions si vous juger cela nécessaire. 
    # NE PAS METTRE DES FONCTIONS HORS LE CONTEXTE DE LA PREDICTION.

# ATTENTION : Les 4 fonctions que j'ai listé ici doivent être présentes dans votre code sous les même noms
    
def clean_trainData():
    """Cleaning, splitting the training data and dummies the cateogial variables"""
    pass

def train_model():
    """Apply the model to the train data and the train target by turning multiple parameters"""
    pass

def get_parameters():
    """Calculate statistical parameters of the model (EX : RMSE)"""
    pass

def save_model():
    """Save the statistical model in a file"""
    pass