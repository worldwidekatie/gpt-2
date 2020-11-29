import numpy as np
import tensorflow as tf
import json
import numbers
import re
import six
# from tensorflow.contrib.training import HParams
# from tensorboard.plugins.hparams import api as hp
import hparam
from hparam import HParams

def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.compat.v1.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.compat.v1.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.compat.v1.exp(x)
    return ex / tf.compat.v1.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.compat.v1.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.compat.v1.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.compat.v1.variable_scope(scope):
        n_state = x.shape[-1]
        g = tf.compat.v1.get_variable('g', [n_state], initializer=tf.compat.v1.constant_initializer(1))
        b = tf.compat.v1.get_variable('b', [n_state], initializer=tf.compat.v1.constant_initializer(0))
        u = tf.compat.v1.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.compat.v1.reduce_mean(tf.compat.v1.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.compat.v1.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.compat.v1.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.compat.v1.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.compat.v1.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.compat.v1.get_variable('w', [1, nx, nf], initializer=tf.compat.v1.random_normal_initializer(stddev=w_init_stdev))
        b = tf.compat.v1.get_variable('b', [nf], initializer=tf.compat.v1.constant_initializer(0))
        c = tf.compat.v1.reshape(tf.compat.v1.matmul(tf.compat.v1.reshape(x, [-1, nx]), tf.compat.v1.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.compat.v1.matrix_band_part(tf.compat.v1.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.compat.v1.range(nd)[:,None]
    j = tf.compat.v1.range(ns)
    m = i >= j - ns + nd
    return tf.compat.v1.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.compat.v1.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.compat.v1.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.compat.v1.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.compat.v1.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.compat.v1.matmul(q, k, transpose_b=True)
        w = w * tf.compat.v1.rsqrt(tf.compat.v1.cast(v.shape[-1], w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.compat.v1.matmul(w, v)
        return a

    with tf.compat.v1.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.compat.v1.split(c, 3, axis=2))
        present = tf.compat.v1.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.compat.v1.unstack(past, axis=1)
            k = tf.compat.v1.concat([pk, k], axis=-2)
            v = tf.compat.v1.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    with tf.compat.v1.variable_scope(scope):
        nx = x.shape[-1]
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2


def block(x, scope, *, past, hparams):
    with tf.compat.v1.variable_scope(scope):
        nx = x.shape[-1]
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.compat.v1.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.compat.v1.tile(tf.compat.v1.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.compat.v1.shape(tokens)[0]
    nsteps = tf.compat.v1.shape(tokens)[1]
    return expand_tile(past_length + tf.compat.v1.range(nsteps), batch_size)


def model(hparams, X, past=None, scope='model', reuse=False):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.compat.v1.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.compat.v1.random_normal_initializer(stddev=0.01))
        wte = tf.compat.v1.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.compat.v1.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.compat.v1.shape(past)[-2]
        h = tf.compat.v1.gather(wte, X) + tf.compat.v1.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.compat.v1.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        results['present'] = tf.compat.v1.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.compat.v1.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.compat.v1.matmul(h_flat, wte, transpose_b=True)
        logits = tf.compat.v1.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
