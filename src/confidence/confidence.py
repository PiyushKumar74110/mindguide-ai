import numpy as np

def get_classification_confidence(proba):
    return np.ax(proba, axis=1)

def get_uncertainity_flag(confidence, threshold=0.6):
    return (confidence<threshold).astype(int)

def get_regression_confidence(predictions):
    conf  = 1 - (np.abs(predictions-3)/2)

    return np.clip(conf,0,1)

def combine_confidence(class_conf, reg_conf, alpha=0.7):
    return alpha*class_conf + (1-alpha)*reg_conf
