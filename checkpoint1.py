
import sys
sys.path.append("/Users/fsolleza/Documents/Projects/timeseries-data") # path to this repository
import py_ts_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


import datetime
from auto_encoder import AutoEncoder, train_step, train_step_v2, Encoder, train_step_v3

dataset_name = 'GunPoint'
X_train, y_train, X_test, y_test, info = py_ts_data.load_data(dataset_name, variables_as_channels=True)
print("Dataset shape: Train: {}, Test: {}".format(X_train.shape, X_test.shape))


def augmentation(x, y, lower_bond = -0.01, upper_bond = 0.01, limits = 1600):
    size = x.shape

    if size[0] > limits: # limits is data augmentation limits
        return x, y

    new_x = [x]
    new_y = [y]
    for i in range(limits // size[0]):
        new_x.append(x + np.random.uniform(lower_bond, upper_bond, size))
        new_y.append(y)

    x = np.concatenate(new_x, axis=0)
    y = np.concatenate(new_y, axis =0)
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
    return (data - means ) /stddev


def evaluate_similarity(X_test, code_test):
    def nn_dist(x, y):
        """
        Sample distance metric, here, using only Euclidean distance
        """
        x = x.reshape((45, 2))
        y = y.reshape((45, 2))
        return np.linalg.norm(x - y)

    nn_x_test = X_test.reshape((-1, 90))
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


# %%

# fig, axs = plt.subplots(1, 2, figsize=(10, 3))
# axs[0].plot(X_train[0])
# X_train = min_max(X_train, feature_range=(-1, 1))
# axs[1].plot(X_train[0])
# X_test = min_max(X_test, feature_range=(-1, 1))
# plt.show()

# %% md

# Encode and Decode

# %%

kwargs = {
    "input_shape": (X_train.shape[1], X_train.shape[2]),
    "filters": [64, 32, 16],
    "kernel_sizes": [5, 5, 5],
    "code_size": 16,
}
input_shape = kwargs["input_shape"]
code_size = kwargs["code_size"]
filters = kwargs["filters"]
kernel_sizes = kwargs["kernel_sizes"]

ae = AutoEncoder(**kwargs)
similarity_encoder = Encoder(input_shape, code_size, filters, kernel_sizes)
# %% md

# Training

# %%

EPOCHS = 100
BATCH = 50
SHUFFLE_BUFFER = 100
similarity_loss_percentage = 0.01
K = len(set(y_train))


X_train, y_train = augmentation(X_train, y_train)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH)

loss_history = []
similarity_history, reconstruction_history = [], []


for epoch in range(EPOCHS):
    total_loss = 0
    total_similarity, total_reconstruction = 0, 0
    if epoch % 10 == 0:
        # every 50 epoch
        evaluate_similarity(X_test, ae.encode(X_test))

    for i, (input, _) in enumerate(train_dataset):
        loss, reconstruction_loss, similarity_loss = train_step_v3(input, ae, similarity_encoder, ld =similarity_loss_percentage) # 0 not use similarity
        total_loss += loss
        total_similarity += similarity_loss
        total_reconstruction += reconstruction_loss

    loss_history.append(total_loss)
    similarity_history.append(total_similarity)
    reconstruction_history.append(total_reconstruction)
    # print("Epoch {}: {}".format(epoch, total_loss), end="\r")
    print("Epoch {}: {}".format(epoch, total_loss))


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(loss_history, 'b')
axs[0, 0].set_title('total loss')
axs[1, 0].plot(similarity_history, 'g')
axs[1, 0].set_title('similarity loss')
axs[1, 1].plot(reconstruction_history, 'r')
axs[1, 1].set_title('reconstruction loss')

print('dataset: ', dataset_name)
print('epochs: ', EPOCHS)
print('batch: ', BATCH)
print('similarity loss percentage: ', similarity_loss_percentage)
# plt.subplot(1,2,1)
# t = [loss_history, similarity_history]
# plt.plot(t)

# plt.plot(loss_history, 'b')

# plt.subplot(2,2,1)
# plt.plot(similarity_history, 'g')
# plt.subplot(2,2,2)
# plt.plot(reconstruction_history, 'r')


# %%

# ae.encode.save()
# ae.decode.save()
# %%

X_train.shape

# %%

y_train[0]

# %%

X_train[0].shape

# %%

# for ch in train_dataset:
#     print(ch)
#     break

# %% md

# Test

# %% md

## Evaluate reconstruction

# %%

code_test = ae.encode(X_test)
decoded_test = ae.decode(code_test)

# plt.subplot(1,2,2)
axs[0, 1].plot(X_test[0])
axs[0, 1].plot(decoded_test[0])
axs[1, 1].set_title('reconstruction example')
# plt.plot(X_test[0])
# plt.plot(decoded_test[0])
plt.show()

losses = []
for ground, predict in zip(X_test, decoded_test):
    losses.append(np.linalg.norm(ground - predict))
print("Mean L2 distance: {}".format(np.array(losses).mean()))

# %% md

## Evaluate Similarity

# %%






# %%
evaluate_similarity(X_test, code_test)

# tf.saved_model.save(ae.decode, '/tmp/adder')

ae.encode.save(r'C:\Users\Ling\OneDrive\Documents\Brown-DESKTOP-8B9G99R\Timeseries-database\\timeseries-similarity\QZ\GunPoint\encoder')
ae.decode.save(r'C:\Users\Ling\OneDrive\Documents\Brown-DESKTOP-8B9G99R\Timeseries-database\\timeseries-similarity\QZ\GunPoint\decoder')

from sample_evaluation import evaluate
evaluate()