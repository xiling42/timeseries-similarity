import tensorflow as tf

import math
import numpy as np

from numpy.random import randint
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft

import itertools as it
from sklearn.metrics import adjusted_rand_score, mean_squared_error, mean_absolute_error

import tensorflow as tf


class Encoder(tf.keras.Model):

    def __init__(self, input_shape, code_size, filters, kernel_sizes):
        super(Encoder, self).__init__()
        assert len(filters) == len(kernel_sizes)
        assert len(input_shape) == 2  # (x, y), x = # of samples, y = # of vars
        # self.input_shape = input_shape
        self.code_size = code_size

        self.convs = []
        self.norms = []
        output_len = input_shape[0]
        output_channels = input_shape[1]

        for f, k in zip(filters, kernel_sizes):
            l = tf.keras.layers.Conv1D(f, k, activation="tanh")
            b = tf.keras.layers.BatchNormalization()
            self.convs.append(l)
            self.norms.append(b)
            output_len = output_len - (k - 1)
            output_channels = f

        self.last_kernel_shape = (output_len, output_channels)
        self.flatten = tf.keras.layers.Flatten()
        self.out = tf.keras.layers.Dense(code_size)

    def call(self, inputs, training=False):

        x = self.convs[0](inputs)
        x = self.norms[0](x)
        for conv, norm in zip(self.convs[1:], self.norms[1:]):
            x = conv(x)
            x = norm(x, training=training)
        assert x.shape[1:] == self.last_kernel_shape
        # print(x.shape)
        x = self.flatten(x)

        x = self.out(x)
        return x




class Decoder(tf.keras.Model):

    def __init__(self, code_size, last_kernel_shape, output_shape, filters, kernel_sizes):
        super(Decoder, self).__init__()

        assert len(last_kernel_shape) == 2
        assert len(output_shape) == 2  # (x, y) x = # of samples, y = samples n variables

        self.code_size = code_size
        self.last_kernel_shape = last_kernel_shape
        self.expected_output_shape = output_shape

        flat_len = last_kernel_shape[0] * last_kernel_shape[1]

        self.expand = tf.keras.layers.Dense(flat_len)
        self.reshape = tf.keras.layers.Reshape(last_kernel_shape)

        self.convs = []
        self.norms = []

        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            l = tf.keras.layers.Conv1DTranspose(f, k)
            b = tf.keras.layers.BatchNormalization()
            self.convs.append(l)
            self.norms.append(b)

    def call(self, inputs, training=False):
        x = self.expand(inputs)
        x = self.reshape(x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x, training=training)
            x = conv(x)
        assert self.expected_output_shape == x.shape[1:]
        return x


_optimizer = tf.keras.optimizers.Nadam(learning_rate=0.00015)
_mse_loss = tf.keras.losses.MeanSquaredError()



def _ncc_c_tf(x, y):
    """
    >>> _ncc_c([1,2,3,4], [1,2,3,4])
    array([ 0.13333333,  0.36666667,  0.66666667,  1.        ,  0.66666667,
            0.36666667,  0.13333333])
    >>> _ncc_c([1,1,1], [1,1,1])
    array([ 0.33333333,  0.66666667,  1.        ,  0.66666667,  0.33333333])
    >>> _ncc_c([1,2,3], [-1,-1,-1])
    array([-0.15430335, -0.46291005, -0.9258201 , -0.77151675, -0.46291005])
    """
    # x, y = x.numpy(), y.numpy()
    # print(x)
    den = np.array(norm(x) * norm(y))
    # den = tf.convert_to_tensor(den)
    # den = tf.Variable(tf.linalg.normalize(x), tf.linalg.normalize(y))
    # print("norm x y ",norm(x), norm(y), "den: ", den)
    den[den == 0] = np.Inf
    # print('den: ', den)
    # print('x: ', x)
    # den = tf.convert_to_tensor(den, dtype = 'float32')
    # print("norm x y ",norm(x), norm(y), "den: ", den)

    x_len = len(x)
    y_len = len(y)
    fft_size = 1 << (2*x_len-1).bit_length()
    # print("fft size", fft_size)
    # cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    # print(x)
    # new_x = tf.zeros(fft_size)
    # print('vs: ',x.numpy())
    # new_x = tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
    #     x.numpy(), maxlen=fft_size, dtype='float32', padding='post',
    #     truncating='post', value=0.0
    # ))
    # print("0000000")
    #

    # if x_len < fft_size:
    #     for i in range(fft_size-x_len):
    #         x.append(0)
    # if x_len > fft_size:
    #     x = x[ :fft_size]
    # # print(x)
    # if y_len < fft_size:
    #     for i in range(fft_size-y_len):
    #         y.append(0)
    # if y_len > fft_size:
    #     y = y[ :fft_size]
    # print(y)
    x = tf.cast(x, dtype =tf.complex64)
    y = tf.cast(y, dtype=tf.complex64)
    cc = tf.signal.ifft(tf.signal.fft(x) * tf.math.conj(tf.signal.fft(y)))

    # cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
    # print("x ", tf.signal.fft(x, fft_size), "   len", len(tf.signal.fft(x, fft_size)), " ", tf.signal.fft(x, fft_size).dtype)
    # print("y ", tf.signal.fft(y, fft_size), "   len", len(tf.signal.fft(x, fft_size)))
    # print("conj y ", tf.math.conj(tf.signal.fft(y, fft_size)), "   len", len(tf.math.conj(tf.signal.fft(y, fft_size))), " ", tf.math.conj(tf.signal.fft(y, fft_size)).dtype)

    # print(cc)
    # print(cc.dtype)
    cc = tf.concat((cc[-(x_len-1):], cc[:x_len]), axis = 0)
    # print(cc)
    cc = tf.cast(cc, dtype = 'float32')
    # cc = tf.dtypes.cast(cc, dtype=tf.complex128)

    return tf.math.real(cc) / den

# def _ncc_c(x, y):
#     """
#     >>> _ncc_c([1,2,3,4], [1,2,3,4])
#     array([ 0.13333333,  0.36666667,  0.66666667,  1.        ,  0.66666667,
#             0.36666667,  0.13333333])
#     >>> _ncc_c([1,1,1], [1,1,1])
#     array([ 0.33333333,  0.66666667,  1.        ,  0.66666667,  0.33333333])
#     >>> _ncc_c([1,2,3], [-1,-1,-1])
#     array([-0.15430335, -0.46291005, -0.9258201 , -0.77151675, -0.46291005])
#     """
#     den = np.array(norm(x) * norm(y))
#     den[den == 0] = np.Inf
#     den = tf.convert_to_tensor(den)
#
#     x_len = len(x)
#     fft_size = 1 << (2*x_len-1).bit_length()
#     # print(fft_size)
#     # # print('fft', tf.signal.fft(x, fft_size))
#     # print(tf.signal.fft(x))
#     # print(tf.conj(tf.signal.fft(y)))
#     # print('i: ', tf.signal.fft(x) * tf.conj(tf.signal.fft(y)))
#     cc = tf.signal.ifft(tf.signal.fft(x) * tf.conj(tf.signal.fft(y)))
#     cc = tf.concatenate((cc[-(x_len-1):], cc[:x_len]))
#     return tf.math.real(cc) / den

def _sbd_tf(x, y):
    """
    >>> _sbd([1,1,1], [1,1,1])
    (-2.2204460492503131e-16, array([1, 1, 1]))
    >>> _sbd([0,1,2], [1,2,3])
    (0.043817112532485103, array([1, 2, 3]))
    >>> _sbd([1,2,3], [0,1,2])
    (0.043817112532485103, array([0, 1, 2]))
    """
    ncc = _ncc_c_tf(x, y)
    # print("ncc_tf  ",ncc)
    # idx = ncc.argmax()
    # print(ncc, "   ", idx)
    ncc_max = tf.reduce_max(ncc)
    # print(ncc_max)
    dist = tf.subtract(1, ncc_max)
    # yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    # yshift_tf = roll_zeropad_tf(y, (idx + 1) - max(len(x), len(y)))

    return dist

class AutoEncoder:
    def __init__(self, **kwargs):

        input_shape = kwargs["input_shape"]
        code_size = kwargs["code_size"]
        filters = kwargs["filters"]
        kernel_sizes = kwargs["kernel_sizes"]

        if "loss" in kwargs:
            loss = kwargs["loss"]
        else:
            loss = _mse_loss

        if "optimizer" in kwargs:
            optimizer = kwargs["optimizer"]
        else:
            optimizer = _optimizer

        self.encode = Encoder(input_shape, code_size, filters, kernel_sizes)

        decoder_filters = list(filters[:len(filters) - 1])
        decoder_filters.append(input_shape[1])
        last_kernel_shape = self.encode.last_kernel_shape

        self.decode = Decoder(code_size, last_kernel_shape, input_shape, decoder_filters,
                              kernel_sizes)

        self.loss = loss
        self.optimizer = optimizer

    def similarity_loss(self, inputs, codes):
        # batchs size * timestamp size * variable size  ?? flatten?
        # batch [10,x,x] -> [C10 2, x, x]

        idx_combination = list(it.combinations([i for i in range(len(inputs))], 2))
        # print('l idx: ', len(idx_combination))
        idx_list_1, idx_list_2 = [list(c) for c in zip(*idx_combination)]
        codes_dist = tf.convert_to_tensor(0.0)
        true_dist = tf.convert_to_tensor(0.0)

        diff = tf.convert_to_tensor(0.0)

        for i in range(len(idx_combination)):
            idx1, idx2 = idx_combination[i]
            inputs_sbd = _sbd_tf(tf.reshape(codes[idx1], [-1]), tf.reshape(codes[idx2], [-1]))
            codes_sbd = _sbd_tf(tf.reshape(inputs[idx1], [-1]), tf.reshape(inputs[idx2], [-1]))
            # print(inputs_sbd, codes_sbd)
            diff += tf.math.square(tf.subtract(inputs_sbd, codes_sbd))
            # codes_dist = tf.add(_sbd_tf(tf.reshape(codes[idx1], [-1]), tf.reshape(codes[idx2], [-1])), codes_dist)
            # codes_dist.append(_sbd_tf(codes[idx1], codes[idx2]))
            # true_dist = tf.add( _sbd_tf(tf.reshape(inputs[idx1], [-1]), tf.reshape(inputs[idx2], [-1])), true_dist)
            # true_dist.append(_sbd_tf(tf.reshape(inputs[idx1], [-1]), tf.reshape(inputs[idx2], [-1])))

        # dist_mae = mean_absolute_error(codes_dist, true_dist)
        # dist_mse = mean_squared_error(codes_dist, true_dist)
        # return tf.math.square(codes_dist-true_dist)

        return diff





# @tf.function
def train_step(inputs, auto_encoder, optimizer=_optimizer, loss=_mse_loss, ld = 0.5):
    # print('---')


    with tf.GradientTape() as tape:

        codes = auto_encoder.encode(inputs, training=True)
        decodes = auto_encoder.decode(codes, training=True)
        loss = loss(inputs, decodes)
        if ld == 1:
            similarity_loss = 0
        else:
            similarity_loss = auto_encoder.similarity_loss(inputs, codes)

        # print('loss')
        # print(loss)
        # print(similarity_loss)
        total_loss = ld * loss + (1 - ld) * similarity_loss
        # total_loss = similarity_loss # use this line to check if similarity loss correctly implemented
        trainables = auto_encoder.encode.trainable_variables + auto_encoder.decode.trainable_variables
        # total_loss = tf.convert_to_tensor(0)
    gradients = tape.gradient(total_loss, trainables)
    optimizer.apply_gradients(zip(gradients, trainables))
    return loss
