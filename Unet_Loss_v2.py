import tensorflow as tf
import numpy as np

def dice_loss_multi_weighted(num_classes,weights):
    def _dice_loss_multi_class_weighted(y_true, y_pred):
        smooth=1e-5
        include_background=False
        # Flatten/Format y_true, y_pred for dsc calculation
        probs = y_pred
        #print('probs',probs)
        probs_f_clasess = tf.reshape(probs,shape=(-1,probs.shape[-1]))
        #print('probs_f_clasess',probs_f_clasess)
        #print(tf.shape( probs )[0])
        y_true = tf.cast(y_true,tf.int32)
        onehot_labels = tf.one_hot(
            indices=y_true,
            depth=num_classes,
            dtype=tf.float32,
            name='onehot_labels')
        onehot_labels_f_classes = tf.reshape(onehot_labels,shape=(-1,onehot_labels.shape[-1]))
#         print('y_true',y_true)
#         print('onehot_labels',onehot_labels)
#         print('onehot_labels_f_classes',onehot_labels_f_classes )

                # Compute the Dice similarity coefficient
        label_sum = tf.reduce_sum(input_tensor=onehot_labels_f_classes, axis=0)
        #print('label_sum',label_sum)
        pred_sum = tf.reduce_sum(input_tensor=probs_f_clasess, axis=0)
        #print('pred_sum',pred_sum)
        intersection = tf.reduce_sum(input_tensor=onehot_labels_f_classes * probs_f_clasess, axis=0)
        #print('intersection',intersection)
        # ref: https://github.com/NifTK/NiftyNet/issues/22

        dice_classes = (2.*intersection+ smooth)/(label_sum + pred_sum+ smooth)
        #dice_classes = tf.boolean_mask(dice_classes,tf.logical_not(tf.equal(label_sum , 0)))
        dice_classes = tf.boolean_mask(tensor=dice_classes,mask=tf.logical_not(tf.math.is_nan(dice_classes)))
        #weights_mask = tf.boolean_mask(weights,tf.logical_not(tf.equal(label_sum , 0)))
        weights_mask = tf.boolean_mask(tensor=weights,mask=tf.logical_not(tf.math.is_nan(dice_classes)))
        weights_mask = tf.divide(weights_mask+ smooth,tf.reduce_sum(input_tensor=weights_mask)+ smooth)
        dice_classes_weighted = tf.multiply(dice_classes,weights_mask)
        loss = 1. - tf.reduce_sum(input_tensor=dice_classes_weighted)
        return loss
    return _dice_loss_multi_class_weighted


def ce_loss_multi(num_classes,weights):
    def _ce_loss_multi_class_weighted(y_true, y_pred):

        smooth=1e-5
        probs = y_pred
        probs_f_clasess = tf.reshape(probs,shape=(-1,probs.shape[-1]))
        #print('probs_f_clasess',probs_f_clasess)
        #print(tf.shape( probs )[0])
        y_true = tf.cast(y_true,tf.uint8)
        onehot_labels = tf.one_hot(
            indices=y_true,
            depth=num_classes,
            dtype=tf.float32,
            name='onehot_labels')
        onehot_labels_f_classes = tf.reshape(onehot_labels,shape=(-1,onehot_labels.shape[-1]))
#         print('y_true',y_true)
#         print('onehot_labels',onehot_labels)
#         print('onehot_labels_f_classes',onehot_labels_f_classes )

#         _weights = tf.divide(weights+ smooth,tf.reduce_sum(weights)+ smooth)
#         #_weights = tf.gather( _weights , tf.reshape(y_true,shape=(-1,onehot_labels.shape[-1])))
#         loss_ce =tf.losses.softmax_cross_entropy(onehot_labels_f_classes,
#                                                  tf.multiply(probs_f_clasess,_weights),
#                                                  #weights = _weights,
#                                                  label_smoothing=1)
        #_weights = tf.divide(weights+ smooth,tf.reduce_sum(weights)+ smooth)
        _weights = weights # to make the value more comparable to dice
        probs_f_clasess = tf.clip_by_value(probs_f_clasess, 10e-8, 1.-10e-8)
        loss_ce = -tf.reduce_sum(input_tensor=tf.multiply(tf.multiply(onehot_labels_f_classes,_weights) , tf.math.log(probs_f_clasess)))
        loss_ce = tf.divide(loss_ce,tf.cast(tf.shape(input=onehot_labels_f_classes)[0],tf.float32) )
        return loss_ce
    return _ce_loss_multi_class_weighted

def ce_dice_loss_multi_weighted(num_classes,weights):
    def ce_dice_loss_multi_class_weighted(y_true, y_pred):
        smooth=1e-5
        include_background=False
        # Flatten/Format y_true, y_pred for dsc calculation


        probs = y_pred
        #print('probs',probs)
        probs_f_clasess = tf.reshape(probs,shape=(-1,probs.shape[-1]))
        #print('probs_f_clasess',probs_f_clasess)
        #print(tf.shape( probs )[0])
        y_true = tf.cast(y_true,tf.int32)
        onehot_labels = tf.one_hot(
            indices=y_true,
            depth=num_classes,
            dtype=tf.float32,
            name='onehot_labels')
        onehot_labels_f_classes = tf.reshape(onehot_labels,shape=(-1,onehot_labels.shape[-1]))
#         print('y_true',y_true)
#         print('onehot_labels',onehot_labels)
#         print('onehot_labels_f_classes',onehot_labels_f_classes )


#         # weighted cross entropy
#         output /= tf.reduce_sum(probs_f_clasess, axis=-1, True)
#         _epsilon = 10e-8
#         output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
#         print('output',output)
#         loss_ce_class = - tf.reduce_sum(onehot_labels_f_classes * tf.log(output)*weights, axis=-1)
#         print('loss_ce_class',loss_ce_class )
#         print()

                # Compute the Dice similarity coefficient
        label_sum = tf.reduce_sum(input_tensor=onehot_labels_f_classes, axis=0)
        #print('label_sum',label_sum)
        pred_sum = tf.reduce_sum(input_tensor=probs_f_clasess, axis=0)
        #print('pred_sum',pred_sum)
        intersection = tf.reduce_sum(input_tensor=onehot_labels_f_classes * probs_f_clasess, axis=0)
        #print('intersection',intersection)
        # ref: https://github.com/NifTK/NiftyNet/issues/22

        dice_classes = (2.*intersection+ smooth)/(label_sum + pred_sum+ smooth)
        #dice_classes = tf.boolean_mask(dice_classes,tf.logical_not(tf.equal(label_sum , 0)))
        dice_classes = tf.boolean_mask(tensor=dice_classes,mask=tf.logical_not(tf.math.is_nan(dice_classes)))
        #weights_mask = tf.boolean_mask(weights,tf.logical_not(tf.equal(label_sum , 0)))
        weights_mask = tf.boolean_mask(tensor=weights,mask=tf.logical_not(tf.math.is_nan(dice_classes)))
        weights_mask = tf.divide(weights_mask+ smooth,tf.reduce_sum(input_tensor=weights_mask)+ smooth)
        dice_classes_weighted = tf.multiply(dice_classes,weights_mask)


        #_weights = tf.divide(weights+ smooth,tf.reduce_sum(weights)+ smooth)
        _weights = weights
        probs_f_clasess = tf.clip_by_value(probs_f_clasess, 10e-8, 1.-10e-8)
        loss_ce = -tf.reduce_sum(input_tensor=tf.multiply(tf.multiply(onehot_labels_f_classes,_weights) , tf.math.log(probs_f_clasess)))# * weights, axis=-1))
        loss_ce = tf.divide(loss_ce,tf.cast(tf.shape(input=onehot_labels_f_classes)[0],tf.float32) )
        print('loss_ce',loss_ce)
        print('_weights',_weights)
        #print('dice',dice)
        loss = 1. - tf.reduce_sum(input_tensor=dice_classes_weighted)+loss_ce
        #print('loss',loss)
        return loss
    return ce_dice_loss_multi_class_weighted

def dice_coeff_np(y_true, y_pred):
    smooth = 1.
    y_true_f = np.reshape(y_true, [-1])
    y_pred_f = np.reshape(y_pred, [-1])
    intersection = np.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return score

def dice_loss_multi_np(y_true, y_pred):

    # cal average dsc only for existing labels
    valid_lbls = np.unique(y_true)
    #print(y_true)
    #print(valid_lbls)
    scores = np.zeros_like(valid_lbls)
    for idx in range(len(valid_lbls)): # including background
        ib = valid_lbls[idx].astype(np.uint8)
        #print(ib)
        y_pred_lbl = y_pred[:,:,:,:,ib]
        y_true_lbl = np.zeros_like(y_true)
        y_true_lbl[y_true ==ib] = 1
        scores[idx] = dice_coeff_np(y_true_lbl, y_pred_lbl)
        print(ib,scores[idx])
    loss = np.mean(scores)
    return loss