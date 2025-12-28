"""
Training script for heart disease model
"""
import pickle
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, hyperparams=None):
    """Train RandomForest model"""
    if hyperparams is None:
        hyperparams = {
            'n_estimators': 100,
            'max_depth': 6,
            'random_state': 42
        }
    
    model = RandomForestClassifier(**hyperparams)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    print("Training module loaded successfully")
