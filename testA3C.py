
# coding: utf-8

# In[1]:


import os
import sys

from chainerrl.agents import a3c
from chainerrl.agents import PPO
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl import policies
import chainer

import chainerrl_autoencoder.experiments_ae as experiments_ae

import logging
import sys
import argparse

import gym
from gym.envs.registration import register

import numpy as np
import marlo
import time

import envs_setup
import plots

print(experiments_ae.__file__)

from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

gpu = 0
steps = 10 ** 6
eval_n_runs = 10
eval_interval = 10000
update_interval = 2048
outdir = 'results'
lr = 3e-4
bound_mean = False
normalize_obs = False\

print('Training with autoencoder reduction')


# In[2]:


class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.SoftmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


# In[3]:


def phi(obs):
    return obs.astype(np.float32)


# In[4]:

envs_setup.start('MarLo-FindTheGoal-v0', width=600, height=400)
env = envs_setup.env

# In[5]:


obs = env.reset()
env.render()

action = env.action_space.sample()
obs, r, done, info = env.step(action)
print('reward: ', r)
print('done: ', done)

print('actions: ', str(env.action_space))


# In[6]:


timestep_limit = env.spec.tags.get(
    'wrapper_config.TimeLimit.max_episode_steps')
obs_space = env.observation_space
action_space = env.action_space

print("observation_space: ", obs_space.low.size)

model = A3CFFSoftmax(3750, action_space.n)


# In[7]:


opt = chainer.optimizers.Adam(alpha=lr, eps=1e-5)
opt.setup(model)


# In[8]:


# Initialize the agent
agent = PPO(
    model, opt,
    gpu=gpu,
    phi=phi,
    update_interval=update_interval,
    minibatch_size=64, epochs=10,
    clip_eps_vf=None, entropy_coef=0.0,
)

# Linearly decay the learning rate to zero


def lr_setter(env, agent, value):
    agent.optimizer.alpha = value


lr_decay_hook = experiments_ae.LinearInterpolationHook(
    steps, 3e-4, 0, lr_setter)

# Linearly decay the clipping parameter to zero


def clip_eps_setter(env, agent, value):
    agent.clip_eps = value


clip_eps_decay_hook = experiments_ae.LinearInterpolationHook(
    steps, 0.2, 0, clip_eps_setter)


# In[ ]:

from chainerrl_autoencoder.experiments_ae.train_agent import train_agent_with_evaluation
# Start training/evaluation
train_agent_with_evaluation(
    agent=agent,
    env=env,
    eval_env=env,
    outdir=outdir,
    steps=steps,
    eval_n_runs=eval_n_runs,
    eval_interval=eval_interval,
    max_episode_len=timestep_limit,
    step_hooks=[
        lr_decay_hook,
        clip_eps_decay_hook
    ],
)
