import cntk as C
import numpy as np


def lrelu(input, leak=0.2, name=""):
    return C.param_relu(C.constant((np.ones(input.shape)*leak).astype(np.float32)), input, name=name)


def bn(input, activation=None, name=""):
    if activation is not None:
        x = C.layers.BatchNormalization(map_rank=1, name=name+"_bn" if name else "")(input)
        x = activation(x, name=name)
    else:
        x = C.layers.BatchNormalization(map_rank=1, name=name)(input)
    return x


def bn_relu(input, name=""):
    return bn(input, activation=C.relu, name=name)


def bn_lrelu(input, name=""):
    return bn(input, activation=C.leaky_relu, name=name)

def conv(input, filter_shape, num_filters, strides=(1,1), init=C.he_normal(), activation=None, pad=True, name=""):
    return C.layers.Convolution(filter_shape, num_filters, strides=strides, pad=pad, activation=activation, init=init, bias=False, name=name)(input)


def conv_bn(input, filter_shape, num_filters, strides=(1,1), init=C.he_normal(), activation=None, name=""):
    x = conv(input, filter_shape, num_filters, strides, init, name=name+"_conv" if name else "")
    x = bn(x, activation, name=name)
    return x


def conv_bn_nopad(input, filter_shape, num_filters, strides=(1,1), init=C.he_normal(), activation=None, name=""):
    x = conv(input, filter_shape, num_filters, strides, init, pad=False, name=name+"_conv" if name else "")
    x = bn(x, activation, name=name)
    return x


def conv_bn_relu(input, filter_shape, num_filters, strides=(1,1), init=C.he_normal(), name=""):
    return conv_bn(input, filter_shape, num_filters, strides, init, activation=C.relu, name=name)


def conv_bn_lrelu(input, filter_shape, num_filters, strides=(1,1), init=C.he_normal(), name=""):
    return conv_bn(input, filter_shape, num_filters, strides, init, activation=C.leaky_relu, name=name)


def conv_bn_relu_nopad(input, filter_shape, num_filters, strides=(1,1), init=C.he_normal(), name=""):
    return conv_bn_nopad(input, filter_shape, num_filters, strides, init, activation=C.relu, name=name)


def conv_bn_lrelu_nopad(input, filter_shape, num_filters, strides=(1,1), init=C.he_normal(), name=""):
    return conv_bn_nopad(input, filter_shape, num_filters, strides, init, activation=C.leaky_relu, name=name)

def flatten(input, name=""):
    assert (len(input.shape) == 3)
    return C.reshape(input, input.shape[0]*input.shape[1]* input.shape[2], name=name)


def flatten_2D(input, name):
    assert (len(input.shape) >= 3)
    return C.reshape(input, (input.shape[-3], input.shape[-2]* input.shape[-1]), name=name)


def broadcast_xy(input_vec, h, w):
    """ broadcast input vector of length d to tensor (d x h x w) """
    assert(h > 0 and w > 0)
    d = input_vec.shape[0]
    # reshape vector to d x 1 x 1
    x = C.reshape(input_vec, (d, 1, 1))
    # create a zeros-like tensor of size (d x h x w)
    t = np.zeros((d, h, w), dtype=np.float32)
    y = C.constant(t)
    z = C.reconcile_dynamic_axes(y, x)
    z = z + x
    return z


def conv_from_weights(x, weights, bias=None, padding=True, name=""):
    """ weights is a numpy array """
    k = C.parameter(shape=weights.shape, init=weights)
    y = C.convolution(k, x, auto_padding=[False, padding, padding])
    if bias:
        b = C.parameter(shape=bias.shape, init=bias)
        y = y + bias
    y = C.alias(y, name=name)
    return y


def bi_recurrence(input, fwd, bwd, name=""):
    F = C.layers.Recurrence(fwd, go_backwards=False, name='fwd_rnn')(input)
    B = C.layers.Recurrence(bwd, go_backwards=True, name='bwd_rnn')(input)
    h = C.splice(F, B, name=name)
    return h