# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 21:25:22 2021

@author: xilin
"""

import sys

sys.path.append("/Users/fsolleza/Documents/Projects/timeseries-data")  # path to this repository
import py_ts_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from evaluation_utils import cal_evaluation
import json
import os
import csv

import datetime
from auto_encoder import AutoEncoder, train_step, train_step_v2, Encoder, train_step_v3
from csv import DictWriter



def augmentation(x, y, lower_bond=-0.01, upper_bond=0.01, limits=1600):
    size = x.shape

    if size[0] > limits:  # limits is data augmentation limits
        return x, y

    new_x = [x]
    new_y = [y]
    for i in range(limits // size[0]):
        new_x.append(x + np.random.uniform(lower_bond, upper_bond, size))
        new_y.append(y)

    x = np.concatenate(new_x, axis=0)
    y = np.concatenate(new_y, axis=0)
    return x, y


# %%

def min_max(data, feature_range=(0, 1)):
    """
    implements min-max scaler
    """
    min_v = feature_range[0]
    max_v = feature_range[1]
    max_vals = data.max(axis=1)[:, None, :]
    min_vals = data.min(axis=1)[:, None, :]
    X_std = (data - min_vals) / (max_vals - min_vals)
    return X_std * (max_v - min_v) + min_v


def normalize(data):
    """
    Z-normalize data with shape (x, y, z)
    x = # of timeseries
    y = len of each timeseries
    z = vars in each timeseres

    s.t. each array in [., :, .] (i.e. each timeseries variable)
    is zero-mean and unit stddev
    """
    sz, l, d = data.shape
    means = np.broadcast_to(np.mean(data, axis=1)[:, None, :], (sz, l, d))
    stddev = np.broadcast_to(np.std(data, axis=1)[:, None, :], (sz, l, d))
    return (data - means) / stddev


def evaluate_similarity(X_test, code_test):
    def nn_dist(x, y):
        """
        Sample distance metric, here, using only Euclidean distance
        """
        x = x.reshape((150,1))
        y = y.reshape((150,1))
        return np.linalg.norm(x - y)

    nn_x_test = X_test.reshape((-1, 150))
    baseline_nn = NearestNeighbors(n_neighbors=10, metric=nn_dist).fit(nn_x_test)
    code_nn = NearestNeighbors(n_neighbors=10).fit(code_test)

    # For each item in the test data, find its 11 nearest neighbors in that dataset (the nn is itself)
    baseline_11nn = baseline_nn.kneighbors(nn_x_test, 11, return_distance=False)
    code_11nn = code_nn.kneighbors(code_test, 11, return_distance=False)

    # On average, how many common items are in the 10nn?
    result = []
    for b, c in zip(baseline_11nn, code_11nn):
        # remove the first nn (itself)
        b = set(b[1:])
        c = set(c[1:])
        result.append(len(b.intersection(c)))
    print('common items: ', np.array(result).mean())
    return np.array(result).mean()


def run_experiment_for_dataset(params, evaluate_when_running=False):
    dataset_name = params['dataset']
    X_train, y_train, X_test, y_test, info = py_ts_data.load_data(dataset_name, variables_as_channels=True)
    print("Dataset shape: Train: {}, Test: {}".format(X_train.shape, X_test.shape))

    kwargs = {
        "input_shape": (X_train.shape[1], X_train.shape[2]),
        "filters": params['filters'],
        "kernel_sizes": params["kernel_sizes"],
        "code_size": 16,
        "reverse": params['decoder_filter_reverse']
    }
    input_shape = kwargs["input_shape"]
    code_size = kwargs["code_size"]
    filters = kwargs["filters"]
    kernel_sizes = kwargs["kernel_sizes"]

    ae = AutoEncoder(**kwargs)

    similarity_encoder = Encoder(input_shape, code_size, filters, kernel_sizes)
    EPOCHS = params['epoch']
    BATCH = params['batch']
    SHUFFLE_BUFFER = 100
    similarity_loss_percentage = params['similarity_rate']
    K = len(set(y_train))

    X_train_aug, y_train_aug = augmentation(X_train, y_train)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_aug, y_train_aug))
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH)

    loss_history = []
    similarity_history, reconstruction_history = [], []

    def write_to_excel():
        print("Epoch {}: {}".format(epoch, total_loss))
        # common_nn = evaluate_similarity(X_test, ae.encode(X_test))
        common_nn = 0
        recon, mse, mae, common_sbd, common_ed, ri = cal_evaluation(params['dataset'], ae.encode, ae.decode, X_train,
                                                                    X_test,
                                                                    y_train)
        loss_history_list = [ch.numpy() for ch in loss_history]
        similarity_history_list = [ch.numpy() for ch in similarity_history]
        reconstruction_history_list = [ch.numpy() for ch in reconstruction_history]
        new_item = {
            "dataset": params['dataset'],
            "epoch": epoch,
            "batch": params['batch'],
            "similarity_rate": params['similarity_rate'],
            "filters": params['filters'],
            'decoder_filter_reverse': params['decoder_filter_reverse'],
            "kernel_sizes": params['kernel_sizes'],
            "ed_only": params['ed_only'],
            'loss_history': loss_history_list,
            'similarity_history': similarity_history_list,
            'reconstruction_history': reconstruction_history_list,
            "model": params['model'],
            'recon': recon,
            'mse': mse,
            'mae': mae,
            'common_nn': common_sbd,
            'common_nn_kmeans': common_ed,
            'rand_index': ri
        }
        cols = ["dataset",
                "epoch",
                "batch",
                "similarity_rate",
                "filters",
                'decoder_filter_reverse',
                "kernel_sizes",
                "ed_only",
                'loss_history',
                'similarity_history',
                'reconstruction_history',
                "model",
                'recon',
                'mse',
                'mae',
                'common_nn',
                'common_nn_kmeans',
                'rand_index']
        filename = 'other_dataset.csv'
        b_create = False
        if not os.path.exists(filename):
            b_create = True
        with open(filename, 'a') as f_object:

            # Pass the file object and a list
            # of column names to DictWriter()
            # You will get a object of DictWriter

            dictwriter_object = DictWriter(f_object, fieldnames=cols)
            if b_create:
                dictwriter_object.writeheader()
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(new_item)

            # Close the file object
            f_object.close()

    for epoch in range(EPOCHS):
        total_loss = 0
        total_similarity, total_reconstruction = 0, 0
        if epoch == 0:
            write_to_excel()
        if evaluate_when_running:
            if epoch % 10 == 0:
                # every 50 epoch
                evaluate_similarity(X_test, ae.encode(X_test))

        for i, (input, _) in enumerate(train_dataset):
            loss, reconstruction_loss, similarity_loss = train_step_v3(input, ae, similarity_encoder,
                                                                       ld=similarity_loss_percentage)  # 0 not use similarity
            total_loss += loss
            total_similarity += similarity_loss
            total_reconstruction += reconstruction_loss

        loss_history.append(total_loss)
        similarity_history.append(total_similarity)
        reconstruction_history.append(total_reconstruction)
        # print("Epoch {}: {}".format(epoch, total_loss), end="\r")

    write_to_excel()


def main():
    params = {
        "epoch": 100,
        "batch": 50,
        "similarity_rate": 0.001,
        "filters": [64, 32, 16],
        'decoder_filter_reverse': True,
        "kernel_sizes": [5, 5, 5],
        "ed_only": False,
        "dataset": "GunPoint",
        "model": "two_encoder",
    }

    params['dataset'] = 'GunPoint'
    ran_history = [] # batch size, similarity rate, filter size
    params['epoch'] = 150

    # for b in [50]:
    #     params['batch'] = b
    #     for s in [0.001, 0.01, 0.1, 0.5]: # 2 encoder, similarity rate cannot be 0
    #         params['similarity_rate'] = s
    #         params['filters'] = [64, 32, 16]
    #         # params['decoder_filter_reverse'] = True
    #         # run_experiment_for_dataset(params)
    #         ran_history.append((b, s, params['filters']))
    #         params['decoder_filter_reverse'] = False
    #         run_experiment_for_dataset(params)
    #         print('batch {} similarity_rate {}'.format(b, s))


    # params['epoch'] = 150
    #
    # params['batch'] = 50
    # for s in [0.1, 0.2, 0.5, 0.01, 0.001]: # 2 encoder, similarity rate cannot be 0
    #     params['similarity_rate'] = s
    #     for fs in [[32,32,32], [64, 32, 16], [16, 32, 64]]:
    #         params['filters'] = fs
    #         if [params['batch'], s, fs] in ran_history:
    #             continue
    #         # params['decoder_filter_reverse'] = True
    #         # run_experiment_for_dataset(params)
    #         ran_history.append((params['batch'], s, params['filters']))
    #         params['decoder_filter_reverse'] = False
    #         run_experiment_for_dataset(params)
    #     print('batch {} similarity_rate {}'.format(params['batch'], s))

    # params['epoch'] = 101
    #
    # params['batch'] = 50
    # for i in range(3):
    #     for s in [0.01]: # 2 encoder, similarity rate cannot be 0
    #         params['similarity_rate'] = s
    #         for fs in [[64,32,16]]:
    #             params['filters'] = fs
    #             if [params['batch'], s, fs] in ran_history:
    #                 continue
    #             # params['decoder_filter_reverse'] = True
    #             # run_experiment_for_dataset(params)
    #             ran_history.append((params['batch'], s, params['filters']))
    #             params['decoder_filter_reverse'] = False
    #             run_experiment_for_dataset(params)
    #         print('batch {} similarity_rate {}'.format(params['batch'], s))

    params['batch'] = 50
    params['epoch'] = 60
    for dset in [ 'Worms', 'UMD', 'Symbols']:
        params['dataset'] = dset
        for s in [0.01]:  # 2 encoder, similarity rate cannot be 0
            params['similarity_rate'] = s
            for fs in [[64, 32, 16]]:
                params['filters'] = fs
                if [params['batch'], s, fs] in ran_history:
                    continue
                # params['decoder_filter_reverse'] = True
                # run_experiment_for_dataset(params)
                ran_history.append((params['batch'], s, params['filters']))
                params['decoder_filter_reverse'] = False
                run_experiment_for_dataset(params)
            print('batch {} similarity_rate {}'.format(params['batch'], s))


    # params['batch'] = 50
    # params['epoch'] = 101

    # params['dataset'] = dset
    # for s in [0.01]:  # 2 encoder, similarity rate cannot be 0
    #     params['similarity_rate'] = s
    #     for fs in [[64, 32, 16]]:
    #         params['filters'] = fs
    #         if [params['batch'], s, fs] in ran_history:
    #             continue
    #         # params['decoder_filter_reverse'] = True
    #         # run_experiment_for_dataset(params)
    #         ran_history.append((params['batch'], s, params['filters']))
    #         params['decoder_filter_reverse'] = False
    #         run_experiment_for_dataset(params)
    #     print('batch {} similarity_rate {}'.format(params['batch'], s))
if __name__ == "__main__":
    # import sys
    # import doctest
    # sys.exit(doctest.testmod()[0])
    main()
