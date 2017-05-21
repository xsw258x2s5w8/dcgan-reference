"""Utilities used in the Kadenze Academy Course on Deep Learning w/ Tensorflow.

Creative Applications of Deep Learning w/ Tensorflow.
Kadenze, Inc.
Parag K. Mital

Copyright Parag K. Mital, June 2016.
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import urllib
import numpy as np
import zipfile
import os
from scipy.io import wavfile
from scipy.misc import imsave



class batch_norm(object):
    def __init__(self,epsilon=1e-5,momentum=0.9,name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon=epsilon
            self.momentum=momentum
            
            self.name=name
            
    def __call__(self,x,train):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            center=True, scale=True, is_training=train, scope=self.name)



def corrupt(x):
    """Take an input tensor and add uniform masking.

    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.
    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    """
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))


def convolve(img, kernel):
    """Use Tensorflow to convolve a 4D image with a 4D kernel.

    Parameters
    ----------
    img : np.ndarray
        4-dimensional image shaped N x H x W x C
    kernel : np.ndarray
        4-dimensional image shape K_H, K_W, C_I, C_O corresponding to the
        kernel's height and width, the number of input channels, and the
        number of output channels.  Note that C_I should = C.

    Returns
    -------
    result : np.ndarray
        Convolved result.
    """
    g = tf.Graph()
    with tf.Session(graph=g):
        convolved = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
        res = convolved.eval()
    return res


def normalize(a, s=0.1):
    '''Normalize the image range for visualization'''
    return np.uint8(np.clip(
        (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5,
        0, 1) * 255)


# %%
def weight_variable(shape, **kwargs):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    if isinstance(shape, list):
        initial = tf.random_normal(tf.stack(shape), mean=0.0, stddev=0.01)
        initial.set_shape(shape)
    else:
        initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial, **kwargs)


# %%
def bias_variable(shape, **kwargs):
    '''Helper function to create a bias variable initialized with
    a constant value.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    if isinstance(shape, list):
        initial = tf.random_normal(tf.stack(shape), mean=0.0, stddev=0.01)
        initial.set_shape(shape)
    else:
        initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial, **kwargs)


def binary_cross_entropy(z, x, name=None):
    """Binary Cross Entropy measures cross entropy of a binary variable.

    loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Parameters
    ----------
    z : tf.Tensor
        A `Tensor` of the same type and shape as `x`.
    x : tf.Tensor
        A `Tensor` of type `float32` or `float64`.
    """
    with tf.variable_scope(name or 'bce'):
        eps = 1e-12
        return (-(x * tf.log(z + eps) +
                  (1. - x) * tf.log(1. - z + eps)))


def conv2d(x, n_output,
           k_h=5, k_w=5, d_h=2, d_w=2,
           padding='SAME', name='conv2d', reuse=None):
    """Helper for creating a 2d convolution operation.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to convolve.
    n_output : int
        Number of filters.
    k_h : int, optional
        Kernel height
    k_w : int, optional
        Kernel width
    d_h : int, optional
        Height stride
    d_w : int, optional
        Width stride
    padding : str, optional
        Padding type: "SAME" or "VALID"
    name : str, optional
        Variable scope

    Returns
    -------
    op : tf.Tensor
        Output of convolution
    """
    with tf.variable_scope(name or 'conv2d', reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[k_h, k_w, x.get_shape()[-1], n_output],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(
            name='conv',
            input=x,
            filter=W,
            strides=[1, d_h, d_w, 1],
            padding=padding)

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=conv,
            bias=b)

    return h, W


def deconv2d(x, n_output_h, n_output_w, n_output_ch, n_input_ch=None,
             k_h=5, k_w=5, d_h=2, d_w=2,
             padding='SAME', name='deconv2d', reuse=None):
    """Deconvolution helper.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to convolve.
    n_output_h : int
        Height of output
    n_output_w : int
        Width of output
    n_output_ch : int
        Number of filters.
    k_h : int, optional
        Kernel height
    k_w : int, optional
        Kernel width
    d_h : int, optional
        Height stride
    d_w : int, optional
        Width stride
    padding : str, optional
        Padding type: "SAME" or "VALID"
    name : str, optional
        Variable scope

    Returns
    -------
    op : tf.Tensor
        Output of deconvolution
    """
    with tf.variable_scope(name or 'deconv2d', reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[k_h, k_w, n_output_ch, n_input_ch or x.get_shape()[-1]],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d_transpose(
            name='conv_t',
            value=x,
            filter=W,
            output_shape=tf.stack(
                [tf.shape(x)[0], n_output_h, n_output_w, n_output_ch]),
            strides=[1, d_h, d_w, 1],
            padding=padding)

        conv.set_shape([None, n_output_h, n_output_w, n_output_ch])

        b = tf.get_variable(
            name='b',
            shape=[n_output_ch],
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(name='h', value=conv, bias=b)

    return h, W


def lrelu(features, leak=0.2,name="lrelu"):
    """Leaky rectifier.

    Parameters
    ----------
    features : tf.Tensor
        Input to apply leaky rectifier to.
    leak : float, optional
        Percentage of leak.

    Returns
    -------
    op : tf.Tensor
        Resulting output of applying leaky rectifier activation.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * features + f2 * abs(features)


def linear(x, n_output, name=None, activation=None, reuse=None,with_w=False):
    """Fully connected layer.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to connect
    n_output : int
        Number of output neurons
    name : None, optional
        Scope to apply

    Returns
    -------
    h, W : tf.Tensor, tf.Tensor
        Output of fully connected layer and the weight matrix
    """
    if len(x.get_shape()) != 2:
        x = flatten(x, reuse=reuse)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fc", reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)

        if activation:
            h = activation(h)
        
        if with_w:
            return h,W,b
        else:
            return h, W


def flatten(x, name=None, reuse=None):
    """Flatten Tensor to 2-dimensions.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to flatten.
    name : None, optional
        Variable scope for flatten operations

    Returns
    -------
    flattened : tf.Tensor
        Flattened tensor.
    """
    with tf.variable_scope('flatten'):
        dims = x.get_shape().as_list()
        if len(dims) == 4:
            flattened = tf.reshape(
                x,
                shape=[-1, dims[1] * dims[2] * dims[3]])
        elif len(dims) == 2 or len(dims) == 1:
            flattened = x
        else:
            raise ValueError('Expected n dimensions of 1, 2 or 4.  Found:',
                             len(dims))

        return flattened


def to_tensor(x):
    """Convert 2 dim Tensor to a 4 dim Tensor ready for convolution.

    Performs the opposite of flatten(x).  If the tensor is already 4-D, this
    returns the same as the input, leaving it unchanged.

    Parameters
    ----------
    x : tf.Tesnor
        Input 2-D tensor.  If 4-D already, left unchanged.

    Returns
    -------
    x : tf.Tensor
        4-D representation of the input.

    Raises
    ------
    ValueError
        If the tensor is not 2D or already 4D.
    """
    if len(x.get_shape()) == 2:
        n_input = x.get_shape().as_list()[1]
        x_dim = np.sqrt(n_input)
        if x_dim == int(x_dim):
            x_dim = int(x_dim)
            x_tensor = tf.reshape(
                x, [-1, x_dim, x_dim, 1], name='reshape')
        elif np.sqrt(n_input / 3) == int(np.sqrt(n_input / 3)):
            x_dim = int(np.sqrt(n_input / 3))
            x_tensor = tf.reshape(
                x, [-1, x_dim, x_dim, 3], name='reshape')
        else:
            x_tensor = tf.reshape(
                x, [-1, 1, 1, n_input], name='reshape')
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    return x_tensor
