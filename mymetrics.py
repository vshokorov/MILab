import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
from tqdm import tqdm_notebook, tqdm

from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
from keras.losses import binary_crossentropy
import tensorflow as tf
import keras as keras

from keras import backend as K

from tqdm import tqdm_notebook
from keras.utils.generic_utils import get_custom_objects

def dice_coef(y_true, y_pred):
    """
    Parameters
    ----------
    y_true : real map for human on picture
    y_pred : predicted map for human on picture
    
    Returns
    -------
    Function counts Dice score metrics for two sets.
    See also https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient
    """
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), TRUSTVAL), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    """
    Parameters
    ----------
    y_true : real map for human on picture
    y_pred : predicted map for human on picture
    
    Returns
    -------
    
    """
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    """
    Parameters
    ----------
    y_true : real mask for human on picture
    y_pred : predicted mask for human on picture
    
    Returns
    -------
    binary cross entropy with Dise loss score.
    """
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def get_iou_vector(A, B):
    """
    Parameters
    ----------
    A : true mask
    B : predicted mask
    
    Returns
    -------
    Intersection over Union metric for .
    """
    # Numpy version
    
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)
        
        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue
        
        # non empty mask case.  Union is never empty 
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        
        # iou metric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45)*20)) / 10
        
        metric += iou
        
    # take the average over all images in batch
    metric /= batch_size
    return metric

def my_iou_metric(label, pred):
    
    return tf.py_func(get_iou_vector, [label, pred > TRUSTVAL], tf.float64)

