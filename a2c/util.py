import tensorflow as tf
import numpy as np

def constant(p):
    return 1

def linear(p):
    return 1-p

def middle_drop(p):
    eps = 0.75
    if 1-p<eps:
        return eps*0.1
    return 1-p

def double_linear_con(p):
    p *= 2
    eps = 0.125
    if 1-p<eps:
        return eps
    return 1-p

def double_middle_drop(p):
    eps1 = 0.75
    eps2 = 0.25
    if 1-p<eps1:
        if 1-p<eps2:
            return eps2*0.5
        return eps1*0.1
    return 1-p

schedules = {
    'linear':linear,
    'constant':constant,
    'double_linear_con': double_linear_con,
    'middle_drop': middle_drop,
    'double_middle_drop': double_middle_drop
}

class Scheduler(object):

    def __init__(self, v, nvalues, schedule):
        self.n = 0.
        self.v = v
        self.nvalues = nvalues
        self.schedule = schedules[schedule]

    def value(self):
        current_value = self.v*self.schedule(self.n/self.nvalues)
        self.n += 1.
        return current_value

    def value_steps(self, steps):
        return self.v*self.schedule(steps/self.nvalues)
def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]



def batch_to_seq(h, nbatch, nsteps, flat=False):
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

def seq_to_batch(h, flat = False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert(len(shape) > 1)
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])


def _lstm(xs, s, scope, nh, init_scale=1.0):
    nbatch, nin = [v.value for v in xs[0].get_shape()]
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh*4], initializer=ortho_init(1.))
        wh = tf.get_variable("wh", [nh, nh*4], initializer=ortho_init(1.))
        b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))
    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, x  in enumerate(xs):
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s

def _ln(x, g, b, e=1e-5, axes=[1]):
    u, s = tf.nn.moments(x, axes=axes, keep_dims=True)
    x = (x-u)/tf.sqrt(s+e)
    x = x*g+b
    return x

def _lnlstm(xs, s, scope, nh, init_scale=1.0):
    nbatch, nin = [v.value for v in xs[0].get_shape()]
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh*4], initializer=ortho_init(init_scale))
        gx = tf.get_variable("gx", [nh*4], initializer=tf.constant_initializer(1.0))
        bx = tf.get_variable("bx", [nh*4], initializer=tf.constant_initializer(0.0))

        wh = tf.get_variable("wh", [nh, nh*4], initializer=ortho_init(init_scale))
        gh = tf.get_variable("gh", [nh*4], initializer=tf.constant_initializer(1.0))
        bh = tf.get_variable("bh", [nh*4], initializer=tf.constant_initializer(0.0))

        b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))

        gc = tf.get_variable("gc", [nh], initializer=tf.constant_initializer(1.0))
        bc = tf.get_variable("bc", [nh], initializer=tf.constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, x  in enumerate(xs):
        z = _ln(tf.matmul(x, wx), gx, bx) + _ln(tf.matmul(h, wh), gh, bh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(_ln(c, gc, bc))
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s



def lstm(X, S, nlstm, nenv = 8, nsteps = 128, scope  ='lstm', layer_norm = True):
    nbatch = nenv * nsteps
    h = tf.layers.flatten(X)

    xs = batch_to_seq(h, nenv, nsteps)
    if layer_norm:
        h5, snew = _lnlstm(xs, S, scope=scope, nh=nlstm)
    else:
        h5, snew = _lstm(xs, S, scope=scope, nh=nlstm)
    
    h = seq_to_batch(h5)
    initial_state = np.zeros(S.shape.as_list(), dtype=float)
    
    return h, snew, initial_state


def multilayer_lstm(X, S_list, lstm_layers, scope = 'lstm_net', nenv = 8, nsteps = 128, layer_norm = False):
    h = X
    states_list = []
    init_states_list = []
    with tf.variable_scope(scope):
        for i in range(len(lstm_layers)):
            nlstm = lstm_layers[i]
            S = S_list[i]

            h, state, init_state =  lstm(h, S, nlstm, nenv, nsteps, 'lstm_' + str(i), layer_norm = layer_norm) 
            states_list.append(state)
            init_states_list.append(init_state)
    return h, states_list, init_states_list


def fc(x, nh, scope, init_scale=1.0, init_bias=0.0, layer_norm = False):
    nin = x.get_shape()[1].value
    with tf.variable_scope(scope):
        w = tf.get_variable("w", [nin, nh], initializer = ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer = tf.random_uniform_initializer(-np.sqrt(3/nin), np.sqrt(3/nin)))
        h = tf.matmul(x, w) + b
        if layer_norm:
            h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
    return h


def multilayer_fc(X, layers = [64, 64], scope = 'fc_net', activation=tf.tanh, layer_norm = False):
    num_layers = len(layers)
    with tf.variable_scope(scope):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            
            num_hidden = layers[i]
            h = fc(h, nh=num_hidden, scope = 'fc_%d' %i, init_scale=np.sqrt(2), layer_norm = layer_norm)
            if activation is not None:
                h = activation(h)
    return h

