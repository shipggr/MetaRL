import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim  
    
        
class LSTMNetwork:
    def __init__(self, obs, nactions, lstm_units = 48,):

            self.nactions = nactions
            self.OBS = obs
            
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units, state_is_tuple = True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            
            self.state = (c_in, h_in)
            self.state_init = [c_init, h_init]

            step_size = tf.shape(self.OBS)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            
            self.lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(lstm_cell, tf.expand_dims(self.OBS, [0]), 
                                                                   initial_state = tf.contrib.rnn.LSTMStateTuple(c_in, h_in),
                                                                   sequence_length = step_size, time_major = False)
            lstm_c, lstm_h = self.lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
     
            self.policy = slim.fully_connected(tf.reshape(self.lstm_outputs, [-1, lstm_units]),nactions,
                                               activation_fn = None,)
            
            self.value = slim.fully_connected(tf.reshape(self.lstm_outputs, [-1, lstm_units]), 1,
                                              activation_fn = None, )
        
            self.probs = tf.nn.softmax(self.policy)
            self.sample_action = tf.squeeze(tf.multinomial(self.policy, 1), axis = 1)
            self.sample_action_oh = tf.one_hot(self.sample_action, depth = nactions)


            
            
  
    def step(self, sess, obs, inp_state):
        # feed values
        feed_dict = {self.OBS : obs, self.state[0] : inp_state[0], self.state[1] : inp_state[1]}
        #print(feed_dict)
        actions, probs, values, lstm_outputs, lstm_states, state = sess.run([self.sample_action, self.probs,  self.value, 
                                                                             self.lstm_outputs, self.lstm_state, 
                                                                             self.state_out], feed_dict)

        return actions, probs, values, lstm_outputs, lstm_states, state

    def predict_value(self, sess, obs, inp_state):
        
        # feed values
        feed_dict = {self.OBS : obs, self.state[0] : inp_state[0], self.state[1] : inp_state[1]}
        values = sess.run([self.value], feed_dict)
        return values
