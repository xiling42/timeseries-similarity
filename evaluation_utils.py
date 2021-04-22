import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import py_ts_data

import tensorflow as tf
# from tensorflow.signal import fft, ifft
# from tensorflow.math import conj
from tensorflow import norm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans
import argparse
import evaluation
from auto_encoder import _sbd_tf



def cal_evaluation(dataset, ENCODER_in, DECODER_in, X_TRAIN, X_TEST, Y_TRAIN, sbd_dis = False):
    N_NEIGHBORS = 10
    N_CLUSTERS = len(set(Y_TRAIN))

    global ENCODER
    global DECODER
    global CLUSTERING


    ENCODER = ENCODER_in
    DECODER = DECODER_in
    
    X_TRAIN = np.squeeze(X_TRAIN, axis=2)
    X_TEST = np.squeeze(X_TEST, axis=2)

    CLUSTERING = KMeans(N_CLUSTERS).fit(X_TRAIN)
    
    recon = evaluation.evaluate_reconstruction(X_TEST, encoder, decoder)
    dist = evaluation.evaluate_distance(X_TEST, encoder, distance_collection)

    common_sbd = evaluation.evaluate_common_nn(X_TEST, X_TEST, encoder, distance_sbd, N_NEIGHBORS)

    common_ed = evaluation.evaluate_common_nn(X_TEST, X_TEST, encoder, distance_timeseries, N_NEIGHBORS)
    ri = evaluation.evaluate_clustering_ri(X_TRAIN, X_TEST, encoder, clustering, N_CLUSTERS)
    
    print("{}, reconstruction: {:.3f}, distance mse: {:.3f}, distance mae: {:.3f}, common nn: {:.3f}, common nn sbd: {:.3f}, rand index: {:.3f}".format(dataset, recon, dist[0], dist[1], common_ed, common_sbd, ri))
    return recon, dist[0], dist[1], common_sbd, common_ed, ri



def encoder(x):
    assert len(x.shape) == 2
    x = x[..., np.newaxis]
    codes = ENCODER(x)
    return codes.numpy()

def decoder(x):
    assert len(x.shape) == 2
    result = DECODER(x).numpy()
    assert len(result.shape) == 3
    assert result.shape[2] == 1
    result = np.squeeze(result, axis=2)
    return result

def distance_collection(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert len(x) == len(y)
    return np.linalg.norm(x-y, axis=1)

def distance_timeseries(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert len(x) == len(y)
    return np.linalg.norm(x-y)

def distance_sbd(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert len(x) == len(y)
    # return np.linalg.norm(x-y)
    return _sbd_tf(x, y)

def clustering(x):
    assert len(x.shape) == 2
    return CLUSTERING.predict(x)



if __name__ == "__main__":

    global ENCODER
    global DECODER

    
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-d', '--dataset', default="GunPoint", required=False, help="dataset to run")
    PARSER.add_argument('-m', '--models', default="QZ", required=False, help="dataset to run")
    ARGS = PARSER.parse_args()
    
    DATA = ARGS.dataset
    MODELS_PATH = ARGS.models
    
    ENCODER = tf.keras.models.load_model(os.path.join(MODELS_PATH, DATA, "encoder"))
    DECODER = tf.keras.models.load_model(os.path.join(MODELS_PATH, DATA, "decoder"))
    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, _ = py_ts_data.load_data(DATA, variables_as_channels=True)
    # all are read in with 3 dims, last is num of variables in the TS
    # assert len(X_TRAIN.shape) == 3
    # # we care only about univariate TS
    # assert X_TRAIN.shape[2] == 1
    #
    # N_NEIGHBORS = 10
    #
    # N_CLUSTERS  = len(set(Y_TRAIN))
    # CLUSTERING = KMeans(N_CLUSTERS).fit(X_TRAIN)
    
    
    cal_evaluation(ARGS.dataset, ENCODER, DECODER, X_TRAIN, X_TEST, Y_TRAIN)
    


