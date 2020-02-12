
import time
from bandits import *
import numpy as np
from collections import defaultdict
import scipy.signal
import tensorflow as tf



def one_hot_encode(actions, max_actions = 2):
    res = np.zeros((len(actions), max_actions))
    for i, a in enumerate(actions):
        res[i, a] = 1
    return res

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def write_to_buffer(episode_buffer, obs, a, p, v, lstm_o, lstm_s, r, d,):
    episode_buffer['obs'].append(obs)
    episode_buffer['actions'].append(a)
    episode_buffer['values'].append(v)
    episode_buffer['rewards'].append(r)
    episode_buffer['dones'].append(d)
    episode_buffer['lstm_outputs'].append(lstm_o)
    episode_buffer['lstm_states'].append(lstm_s)
    
def read_from_buffer(episode_buffer,):
    obses = np.concatenate(episode_buffer['obs'])
    actions = np.concatenate(episode_buffer['actions'])
    values = np.reshape(episode_buffer['values'], (-1))
    rewards = np.array(episode_buffer['rewards'])
    dones = np.array(episode_buffer['dones'])
    
    return obses, actions, values, rewards, dones


class Runner:
    
    def __init__(self, model,):
        self.model = model
        self.rewards_history = defaultdict(list)
        self.losses_history = defaultdict(list)
        
    def run_episode(self, bandit):
        
        self.episode_buffer = defaultdict(list)
        episode_reward = 0
        
        bandit.reset()
        a, r = [0], 0
        s = self.model.state_init
        
        for st in range(bandit.maxtimestep):
            
            obs = np.concatenate([one_hot_encode(a), [[r]], [[st]]], 1)
            
            a, p, v, lstm_o, lstm_s, s = self.model.step(obs, s,)
            r, d, ts, corr = bandit.act(a[0])
            
            episode_reward += r
            write_to_buffer(self.episode_buffer, obs, a, p, v, lstm_o, lstm_s, r, d,)
            
        self.rewards_history[bandit.name].append(episode_reward)
        
    def train_model(self,):
        obses, actions, values, rewards, dones = read_from_buffer(self.episode_buffer)

        rewards_plus = np.asarray(rewards.tolist() + [0])
        value_plus = np.asarray(values.tolist() + [0])

        returns = discount(rewards_plus, self.model.gamma)[:-1]

        advs = rewards + self.model.gamma * value_plus[1:] - value_plus[:-1]
        advs = discount(advs, self.model.gamma)
        
        ploss, vloss, entloss = self.model.train(obses, actions, values, returns, advs, self.model.state_init)
        self.losses_history['Losses/Policy Loss'].append(ploss)
        self.losses_history['Losses/Value Loss'].append(vloss)
        self.losses_history['Losses/Entropy Loss'].append(entloss)
        
    def run_training(self, neps, bandits, save_summary_every = 10, save_model_every = 500):
        
        self.rewards_history = defaultdict(list)
        self.losses_history = defaultdict(list)
        episodes_start = 0
        time_start = time.time()
        
        for ep in range(1, neps + 1):
            for bandit in bandits:
                self.run_episode(bandit)
                self.train_model()
            if ep % save_summary_every == 0:
                summary = tf.Summary()
                for key in self.losses_history:
                    summary.value.add(tag = key, simple_value = np.nanmean(self.losses_history[key]))
                for key in self.rewards_history:
                    summary.value.add(tag = 'Perf/%s Reward' % key, 
                                      simple_value = np.nanmean(self.rewards_history[key]))

                model_steps = self.model.sess.run(self.model.timesteps)
                self.model.writer.add_summary(summary, model_steps)
                self.model.writer.flush()
                self.losses_history = defaultdict(list)
                self.rewards_history = defaultdict(list)
                
            if ep % save_model_every == 0:
                self.model.save_model()
                speed = (ep - episodes_start) / (time.time() - time_start)
                print()
                print('Savinig model after %d episodes of training' % ep) 
                print('Average env speed is %.1f ep/second' % (speed))
                episodes_start, time_start = ep, time.time()
                
    def run_evaluation(self, neps, bandits):
        
        self.rewards_history = defaultdict(list)
        self.stats = defaultdict(lambda : defaultdict(list))
        
        episodes_start = 0
        time_start = time.time()
        
        for ep in range(1, neps + 1):
            for bandit in bandits:
                self.run_episode(bandit)
                for k, val in self.episode_buffer.items():
                    self.stats[bandit.name][k].extend(val)
        return self.rewards_history, self.stats