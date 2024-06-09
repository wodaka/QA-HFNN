import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def cal_ms(list_s):
    mean_ = np.mean(np.array(list_s))
    stderr = np.std(np.array(list_s))
    return mean_, stderr

def cal_merics(y_true,y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return precision,recall,f1

