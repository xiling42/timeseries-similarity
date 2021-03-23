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
        # print(inputs.shape)
        x = self.convs[0](inputs)
        x = self.norms[0](x)
        for conv, norm in zip(self.convs[1:], self.norms[1:]):
            x = conv(x)
            x = norm(x, training=training)
        assert x.shape[1:] == self.last_kernel_shape
        # print(x.shape)
        x = self.flatten(x)

        x = self.out(x)
        # print(x.shape)
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
        # print(inputs.shape)
        x = self.expand(inputs)
        x = self.reshape(x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x, training=training)
            x = conv(x)
        assert self.expected_output_shape == x.shape[1:]
        # print(x.shape)
        return x


_optimizer = tf.keras.optimizers.Nadam(learning_rate=0.00015)
_mse_loss = tf.keras.losses.MeanSquaredError()
_similarity_loss = tf.keras.losses.MeanSquaredError()

def euclideanDistances(A, B):
    """
    >>> euclideanDistances(np.array([[0,1],[2,3],[4,5]]), np.array([[8,9]]))
    array([ 128, 72, 32])

    """
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A, BT)
    # print(vecProd)
    SqA = A ** 2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


def euclidean(a, b, sqrt=False):
    """
    >>> euclidean(tf.reshape(tf.range(6), [3,2]), tf.convert_to_tensor([[8, 9]]))
    array([ 128, 72, 32])

    """
    aTa = tf.linalg.diag_part(tf.matmul(a, tf.transpose(a)))
    bTb = tf.linalg.diag_part(tf.matmul(b, tf.transpose(b)))
    aTb = tf.matmul(a, tf.transpose(b))
    ta = tf.reshape(aTa, [-1, 1])
    tb = tf.reshape(bTb, [1, -1])

    D = ta - 2.0 * aTb + tb
    if sqrt:
        D = tf.sqrt(D)
    return D


def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res

def _sbd_tf_2d(x, y):
    """
    >>> _sbd_tf_2d([1,1,1], [1,1,1])
    (-2.2204460492503131e-16, array([1, 1, 1]))
    >>> _sbd_tf_2d([0,1,2], [1,2,3])
    (0.043817112532485103, array([1, 2, 3]))
    >>> _sbd_tf_2d([1,2,3], [0,1,2])
    (0.043817112532485103, array([0, 1, 2]))
    """
    # print(x.dtype, y.dtype)
    ncc = _ncc_c_3dim_tf(x, y)
    ncc = tf.reshape(ncc, (-1, ncc.shape[2]))
    # print("ncc_tf  ",ncc)
    # idx = ncc.argmax()
    # print(ncc)
    ncc_max = tf.reduce_max(ncc, axis=1)
    # print(ncc_max)
    dist = 1 - ncc_max
    # dist = 1 - ncc_max + 1.0e-12 # ????

    return dist
def _ncc_c_3dim_tf(x, y):
    """
    Variant of NCCc that operates with 2 dimensional X arrays and 2 dimensional
    y vector
    Returns a 3 dimensional array of normalized fourier transforms
    """
    # print("-----ncc3_tf-----")
    den = den = np.array(norm(x, axis=1)[:, None] * norm(y, axis=1))
    den[den == 0] = np.Inf
    den = tf.convert_to_tensor(den)
    # print("den ", den)
    # print("den.T ", tf.transpose(den)[:, :, None])
    x_len = x.shape[-1]
    fft_size = 1 << (2 * x_len - 1).bit_length()
    # print("x_len ", x_len, "  fft size: ", fft_size)

    # new_x = tf.zeros(fft_size)
    # print('vs: ', new_x)
    # temp = [[1], [1,2]]
    new_x = tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
        x, maxlen=fft_size, dtype='float32', padding='post',
        truncating='post', value=0.0
    ))
    # print("new x ", new_x)
    # print("x: ", x)

    new_y = tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
        y, maxlen=fft_size, dtype='float32', padding='post',
        truncating='post', value=0.0
    ))
    # print("new y ", new_y)
    # print("y: ", y)

    new_x = tf.cast(new_x, dtype=tf.complex128)
    new_y = tf.cast(new_y, dtype=tf.complex128)

    # cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size))[:, None])
    cc = tf.signal.ifft(tf.signal.fft(new_x) * tf.math.conj(tf.signal.fft(new_y))[:, None])
    # print("fft x: ", tf.math.conj(tf.signal.fft(new_y)))

    # print("cc: ", cc)
    # cc = np.concatenate((cc[:,:,-(x_len-1):], cc[:,:,:x_len]), axis=2)
    cc = tf.concat((cc[:, :, -(x_len - 1):], cc[:, :, :x_len]), axis=2)
    # print("cc concatenate: ", cc)
    return tf.cast(tf.math.real(cc), dtype = tf.float32) / tf.transpose(den)[:, :, None]


def _ncc_c_3dim(x, y):
    """
    Variant of NCCc that operates with 2 dimensional X arrays and 2 dimensional
    y vector
    Returns a 3 dimensional array of normalized fourier transforms
    """
    x, y = np.squeeze(x), np.squeeze(y)
    den = norm(x, axis=1)[:, None] * norm(y, axis=1)
    den[den == 0] = np.Inf
    x_len = x.shape[-1]
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size))[:, None])
    cc = np.concatenate((cc[:,:,-(x_len-1):], cc[:,:,:x_len]), axis=2)
    return np.real(cc) / den.T[:, :, None]


def _sbd_ling(x, y):
    """
    >>> _sbd([1,1,1], [1,1,1])
    (-2.2204460492503131e-16, array([1, 1, 1]))
    >>> _sbd([0,1,2], [1,2,3])
    (0.043817112532485103, array([1, 2, 3]))
    >>> _sbd([1,2,3], [0,1,2])
    (0.043817112532485103, array([0, 1, 2]))
    """
    ncc = _ncc_c_3dim(x, y)
    distances = (1 - _ncc_c_3dim(x, y).max(axis=2))

    return distances



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
    fft_size = 1 << (2 * x_len - 1).bit_length()
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
    x = tf.cast(x, dtype=tf.complex64)
    y = tf.cast(y, dtype=tf.complex64)
    cc = tf.signal.ifft(tf.signal.fft(x) * tf.math.conj(tf.signal.fft(y)))

    # cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
    # print("x ", tf.signal.fft(x, fft_size), "   len", len(tf.signal.fft(x, fft_size)), " ", tf.signal.fft(x, fft_size).dtype)
    # print("y ", tf.signal.fft(y, fft_size), "   len", len(tf.signal.fft(x, fft_size)))
    # print("conj y ", tf.math.conj(tf.signal.fft(y, fft_size)), "   len", len(tf.math.conj(tf.signal.fft(y, fft_size))), " ", tf.math.conj(tf.signal.fft(y, fft_size)).dtype)

    # print(cc)
    # print(cc.dtype)
    cc = tf.concat((cc[-(x_len - 1):], cc[:x_len]), axis=0)
    # print(cc)
    cc = tf.cast(cc, dtype=tf.complex128)
    # cc = tf.dtypes.cast(cc, dtype=tf.complex128)

    return tf.math.real(cc) / den


def _ncc_c_tf_1(x, y):
    """
    >>> _ncc_c([1,2,3,4], [1,2,3,4])
    array([ 0.13333333,  0.36666667,  0.66666667,  1.        ,  0.66666667,
            0.36666667,  0.13333333])
    >>> _ncc_c([1,1,1], [1,1,1])
    array([ 0.33333333,  0.66666667,  1.        ,  0.66666667,  0.33333333])
    >>> _ncc_c([1,2,3], [-1,-1,-1])
    array([-0.15430335, -0.46291005, -0.9258201 , -0.77151675, -0.46291005])
    """
    den = np.array(norm(x) * norm(y))
    # den = tf.convert_to_tensor(den)
    # den = tf.Variable(tf.linalg.normalize(x), tf.linalg.normalize(y))
    # print("norm x y ",norm(x), norm(y), "den: ", den)
    den[den == 0] = np.Inf
    den = tf.convert_to_tensor(den)
    # print("norm x y ",norm(x), norm(y), "den: ", den)

    x_len = len(x)
    y_len = len(y)
    fft_size = 1 << (2 * x_len - 1).bit_length()
    # print("fft size", fft_size)
    # cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    # print(x)
    if x_len < fft_size:
        for i in range(fft_size - x_len):
            # x.append(0)
            x = np.append(x, np.array([0]))
    if x_len > fft_size:
        x = x[:fft_size]
    # print(x)
    if y_len < fft_size:
        for i in range(fft_size - y_len):
            # y.append(0)
            y = np.append(y, np.array([0]))
    if y_len > fft_size:
        y = y[:fft_size]
    # print(y)
    x = tf.cast(x, dtype=tf.complex64)
    y = tf.cast(y, dtype=tf.complex64)
    cc = tf.signal.ifft(tf.signal.fft(x) * tf.math.conj(tf.signal.fft(y)))
    # cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
    # print("x ", tf.signal.fft(x, fft_size), "   len", len(tf.signal.fft(x, fft_size)), " ", tf.signal.fft(x, fft_size).dtype)
    # print("y ", tf.signal.fft(y, fft_size), "   len", len(tf.signal.fft(x, fft_size)))
    # print("conj y ", tf.math.conj(tf.signal.fft(y, fft_size)), "   len", len(tf.math.conj(tf.signal.fft(y, fft_size))), " ", tf.math.conj(tf.signal.fft(y, fft_size)).dtype)

    # print(cc)
    # print(cc.dtype)
    cc = tf.concat((cc[-(x_len - 1):], cc[:x_len]), axis=0)
    # print(cc)
    cc = tf.dtypes.cast(cc, dtype=tf.complex128)  # 64?
    den = tf.dtypes.cast(den, dtype='float64')  # 32?
    # print(cc.dtype)
    # return np.real(cc) / den
    # print(tf.math.real(cc).dtype)
    # print(den.dtype)
    return tf.math.real(cc) / den


def _sbd_tf(x, y):
    """
    >>> _sbd([1,1,1], [1,1,1])
    (-2.2204460492503131e-16, array([1, 1, 1]))
    >>> _sbd([0,1,2], [1,2,3])
    (0.043817112532485103, array([1, 2, 3]))
    >>> _sbd([1,2,3], [0,1,2])
    (0.043817112532485103, array([0, 1, 2]))
    """
    ncc = _ncc_c_tf_1(x, y)
    # print("ncc_tf  ",ncc)
    # idx = ncc.argmax()
    # print(ncc, "   ", idx)
    ncc_max = tf.reduce_max(ncc)
    # print(ncc_max)
    dist = tf.subtract(1, ncc_max)
    # yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    # yshift_tf = roll_zeropad_tf(y, (idx + 1) - max(len(x), len(y)))

    return dist


def _ncc_c(x, y):
    """
    >>> _ncc_c([1,2,3,4], [1,2,3,4])
    array([ 0.13333333,  0.36666667,  0.66666667,  1.        ,  0.66666667,
            0.36666667,  0.13333333])
    >>> _ncc_c([1,1,1], [1,1,1])
    array([ 0.33333333,  0.66666667,  1.        ,  0.66666667,  0.33333333])
    >>> _ncc_c([1,2,3], [-1,-1,-1])
    array([-0.15430335, -0.46291005, -0.9258201 , -0.77151675, -0.46291005])
    """
    den = np.array(norm(x) * norm(y))
    # print("norm x y ",norm(x), norm(y), "den: ", den)
    den[den == 0] = np.Inf
    # print("norm x y ",norm(x), norm(y), "den: ", den)

    x_len = len(x)
    fft_size = 1 << (2 * x_len - 1).bit_length()
    # print("fft size", fft_size)
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    # print("x ", fft(x, fft_size), "   len", len(fft(x, fft_size)), " ", fft(x, fft_size).dtype)
    # print("y ", fft(y, fft_size), "   len", len(fft(x, fft_size)))
    # print("conj y ", np.conj(fft(y, fft_size)), "   len", len(np.conj(fft(y, fft_size))))
    # print(cc)
    # print(cc.dtype)
    cc = np.concatenate((cc[-(x_len - 1):], cc[:x_len]))
    # print(cc)
    # print(cc.dtype)
    # print((np.real(cc) / den).dtype)
    return np.real(cc) / den


def _sbd(x, y):
    """
    >>> _sbd([1,1,1], [1,1,1])
    (-2.2204460492503131e-16, array([1, 1, 1]))
    >>> _sbd([0,1,2], [1,2,3])
    (0.043817112532485103, array([1, 2, 3]))
    >>> _sbd([1,2,3], [0,1,2])
    (0.043817112532485103, array([0, 1, 2]))
    """
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    # print("ncc ", ncc, "   ", idx)
    dist = 1 - ncc[idx]
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

    # def similarity_loss(self, codes, decodes):
    #     # batchs size * timestamp size * variable size  ?? flatten?
    #     # batch [10,x,x] -> [C10 2, x, x]
    #
    #     idx_combination = list(it.combinations([i for i in range(len(decodes))], 2))
    #     # print('l idx: ', len(idx_combination))
    #     idx_list_1, idx_list_2 = [list(c) for c in zip(*idx_combination)]
    #     codes_dist = tf.convert_to_tensor(0.0)
    #     true_dist = tf.convert_to_tensor(0.0)
    #
    #     diff = tf.convert_to_tensor(0.0)
    #
    #     # ed
    #     # diff = tf.cast(l_codes-r_codes, tf.float64)
    #     # code_distances = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=1)  + 1.0e-12)
    #
    #     for i in range(len(idx_combination)):
    #         idx1, idx2 = idx_combination[i]
    #         # codes_sbd = _sbd_tf(tf.reshape(codes[idx1], [-1]), tf.reshape(codes[idx2], [-1])) # change to ED
    #         # inputs_sbd = _sbd_tf(tf.reshape(inputs[idx1], [-1]), tf.reshape(inputs[idx2], [-1])) # change to decoder
    #
    #         decode_sbd = _sbd(tf.reshape(decodes[idx1], [-1]).numpy(), tf.reshape(decodes[idx2], [-1]).numpy())
    #         ed_diff = tf.cast(tf.reshape(codes[idx1], [-1]) - tf.reshape(codes[idx2], [-1]), tf.float32)
    #         codes_ed = tf.sqrt(tf.reduce_sum(tf.square(ed_diff), axis=0) + 1.0e-12)
    #         # codes_sbd = _sbd_tf(tf.reshape(codes[idx1], [-1]), tf.reshape(codes[idx2], [-1]))
    #
    #         decode_sbd = tf.cast(decode_sbd, dtype=tf.float32)
    #         # print(decode_sbd, codes_ed)
    #         # print(decode_sbd.dtype,codes_ed.dtype)
    #         diff += tf.math.square(tf.subtract(decode_sbd, codes_ed))
    #         # codes_dist = tf.add(_sbd_tf(tf.reshape(codes[idx1], [-1]), tf.reshape(codes[idx2], [-1])), codes_dist)
    #         # codes_dist.append(_sbd_tf(codes[idx1], codes[idx2]))
    #         # true_dist = tf.add( _sbd_tf(tf.reshape(inputs[idx1], [-1]), tf.reshape(inputs[idx2], [-1])), true_dist)
    #         # true_dist.append(_sbd_tf(tf.reshape(inputs[idx1], [-1]), tf.reshape(inputs[idx2], [-1])))
    #
    #     # dist_mae = mean_absolute_error(codes_dist, true_dist)
    #     # dist_mse = mean_squared_error(codes_dist, true_dist)
    #     # return tf.math.square(codes_dist-true_dist)
    #
    #     return diff / len(idx_combination)

    def similarity_loss(self, codes, decodes):

        combination_length = (len(codes) * len(codes) - len(codes)) / 2

        sbd_distances = _sbd_tf_2d(tf.reshape(decodes, (decodes.shape[0], -1)),
                                   tf.reshape(decodes, (decodes.shape[0], -1)))
        sbd_reshape = tf.reshape(sbd_distances, (len(codes), -1))

        d2 = euclidean(tf.reshape(codes, (codes.shape[0], -1)), tf.reshape(codes, (codes.shape[0], -1)), True)

        with_diagonal = tf.linalg.band_part(sbd_reshape - d2, -1, 0)
        without_diagonal = tf.linalg.set_diag(with_diagonal, [0 for i in range(len(codes))])

        nt = tf.math.reduce_sum(tf.math.square(without_diagonal)) / combination_length

        return nt


# @tf.function
def train_step(inputs, auto_encoder, optimizer=_optimizer, loss=_mse_loss, ld=0.5):
    # print('---')

    with tf.GradientTape() as tape:

        codes = auto_encoder.encode(inputs, training=True)
        decodes = auto_encoder.decode(codes, training=True)
        loss = loss(inputs, decodes)
        if ld == 0:
            similarity_loss = 0
        else:
            similarity_loss = auto_encoder.similarity_loss(codes, decodes)

        # print('loss')
        # print(similarity_loss)
        print("reconstruction loss: ", loss, " ", "similarity_loss: ", similarity_loss)

        total_loss = (1-ld) * loss + ld * similarity_loss
        # total_loss = loss + (1e-1) * similarity_loss
        # total_loss = similarity_loss # use this line to check if similarity loss correctly implemented
        trainables = auto_encoder.encode.trainable_variables + auto_encoder.decode.trainable_variables
        # total_loss = tf.convert_to_tensor(0)
    gradients = tape.gradient(total_loss, trainables)
    optimizer.apply_gradients(zip(gradients, trainables))
    return total_loss, loss, similarity_loss


def main():
    x = [1, 1, 1]
    y = [1, 1, 1]
    # x = [1,2,3,4]
    # y = [1,2,3,4]
    import py_ts_data

    X_train, y_train, X_test, y_test, info = py_ts_data.load_data("Libras", variables_as_channels=True)
    print("Dataset shape: Train: {}, Test: {}".format(X_train.shape, X_test.shape))

    print("sbd: ")
    dist = _sbd(np.reshape(X_train[0], [-1]), np.reshape(X_train[3], [-1]))
    print("dist ", dist)
    dist_tf = _sbd_tf(tf.reshape(X_train[0], [-1]), tf.reshape(X_train[3], [-1]))
    print("dist tf ", dist_tf)

    idx_combination = list(it.combinations([i for i in range(10)], 2))
    print('l idx: ', idx_combination)


if __name__ == "__main__":
    # import sys
    # import doctest
    # sys.exit(doctest.testmod()[0])
    main()