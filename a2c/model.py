import numpy as np
import pickle

from .network import LSTMNetwork
from .util import Scheduler

import tensorflow as tf
from tf_agents.utils import tensor_normalizer as tens_norm
from tf_agents.specs import tensor_spec
from tf_agents.utils import common



class Model(object):
    def __init__(self, nactions, nobs, sess, scope = '',
                 lstm_units = 48, gamma = 0.99, ent_coef = 0.01, vf_coef = 0.5, 
                 max_grad_norm = 40, lr = 0.001, lr_half_period = int(1e6), anneal_lr = True,
                 normalize_advs = True, path = './experiments/run/'):

        #save parameters
        self.scope = scope
        self.nactions = nactions
        self.nobs = nobs
            
        self.init_lr = lr
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.normalize_advs = normalize_advs
        self.sess = sess
        self.path = path + scope + '/'

        self.anneal_lr = anneal_lr
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        with tf.variable_scope(scope):
            # CREATE VARIABLES AFFECTING TRAINING
            self.lr = tf.Variable(self.init_lr, dtype = tf.float32, name = 'learning_rate', trainable = False)
            self.train_epochs = tf.Variable(0, dtype = tf.int32, name = 'total_updates', trainable = False)
            self.timesteps = tf.Variable(0, dtype = tf.int32, name = 'timesteps', trainable = False)
            
            # ops for updating variables
            self.update_lr = tf.assign(self.lr, self.lr * np.power(0.5, 1 / lr_half_period))
            self.update_train_epochs = self.train_epochs.assign_add(1)
            self.update_timesteps = self.timesteps.assign_add(1)
            
            # CREATE PLACEHOLDERS
            self.OBS = tf.placeholder(shape = [None, nobs], dtype = 'float32', name = 'observations')
            self.A = tf.placeholder(shape = [None, ], dtype = 'int32', name = 'actions')
            self.A_OH = tf.one_hot(self.A, self.nactions, dtype = 'float32', name = 'actions_oh')
            self.ADVS = tf.placeholder(shape = [None, ], dtype = 'float32', name =  'advantages')
            self.R = tf.placeholder(shape = [None, ], dtype = 'float32', name = 'returns')
            
            #Network
            with tf.variable_scope('network'):
                network = LSTMNetwork(self.OBS, nactions, lstm_units)
            
            #Loss functions
            self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.R - tf.reshape(network.value, [-1])))
            self.entropy =  - tf.reduce_sum(network.probs * tf.log(network.probs + 1e-7))
            self.responsible_outputs = tf.reduce_sum(network.probs * self.A_OH, [1])
            self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7) * self.ADVS)
            self.loss = self.vf_coef * self.value_loss + self.policy_loss - self.ent_coef * self.entropy 
            
            #Get gradients from local network using local losses
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss, params)
            self.var_norms = tf.global_norm(params)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, max_grad_norm)
            trainer = tf.train.AdamOptimizer(learning_rate = self.lr)
            self._train = trainer.apply_gradients(zip(grads, params))
            
          
            
        self.network = network
        self.saver = tf.train.Saver(var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope))
        self.writer = tf.summary.FileWriter(path + scope + '/summary/' )
        self.state_init = network.state_init
        sess.run(tf.global_variables_initializer())
        
    def step(self, obs, inp_state):
        
        actions, probs, values, lstm_outputs, lstm_states, state = self.network.step(self.sess, obs, inp_state)
        _ = self.sess.run(self.update_timesteps)  
        
        return actions, probs, values, lstm_outputs, lstm_states, state
    
    def predict_value(self, obs, inp_state):
        
        values = self.network.predict_value(self.sess, obs, inp_state)
        
        return values
                                       
    def train(self, obs, actions, values, returns, advs, inp_state):
        # feed values
        feed_dict = {self.OBS : obs, self.A : actions, self.ADVS : advs,
                     self.R : returns, self.network.state[0] : inp_state[0], self.network.state[1] : inp_state[1]}
      
        self.sess.run(self.update_train_epochs)
        
        ploss, vloss, entloss, _ = self.sess.run([self.policy_loss, self.value_loss, self.entropy, self._train],
                                                 feed_dict)

               
        if self.anneal_lr:
            self.sess.run([self.update_lr])
            
            
        return ploss, vloss, entloss


    def save_model(self,):
        self.saver.save(self.sess, self.path + 'model/model.cptk')
            
    def load_model(self, reset_lr = False):
        ckpt = tf.train.get_checkpoint_state(self.path + 'model/')
        if ckpt is None:
            print('Could not load model "%s" at %s' % (self.scope, self.path + 'model/'))
        else:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            if reset_lr:
                print('Resetting lr to %f!' % self.init_lr)
                self.sess.run([self.lr.assign(self.init_lr)])
                
            train_epochs, ts, lr = self.sess.run([self.train_epochs, self.timesteps, self.lr])
            print('Successfully loaded model "%s":' % self.scope)
            print('  "%s" was trained for %d epochs, using %d timesteps' %(self.scope, train_epochs, ts))
            print('  "%s" has following parameters: lr = %f' % (self.scope, lr))