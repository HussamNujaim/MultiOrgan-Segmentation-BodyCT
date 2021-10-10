import tensorflow.python.keras as K

def evaluate(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)

    zeros_list_true, one_list_true, zeros_list_pred, one_list_pred = [], [], [], []
    
    for i in len(y_true_pos):
        ypred = y_pred_pos[i]
        ytrue  = y_true[i]
        if ypred == 0:
            zeros_list_pred.append(i)
        else:
            one_list_pred.append(i)
        if ytrue == 0:
            zeros_list_true.append(i)
        else:
            one_list_true.append(i)
    TP = len(set(one_list_true).intersection(set(one_list_pred)))
    TN = len(set(zeros_list_true).intersection(set(zeros_list_pred)))
    FP = len(set(one_list_true).intersection(set(zeros_list_pred)))
    FN = len(set(zeros_list_true).intersection(set(one_list_pred)))

    DSC = 2*(TP/(FP + 2*TP + FN))

    return DSC

