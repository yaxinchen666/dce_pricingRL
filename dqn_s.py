import tensorflow as tf
from tensorflow import keras
from collections import deque
import numpy as np
import random


class DQNAgent(object):
    def __init__(self, env, agent_params):
        self.env = env
        self.learning_starts = agent_params['learning_starts']  # time step before learning starts
        self.replay_buffer_size = agent_params['replay_buffer_size']
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.state_size = agent_params['state_size']  # size of states (input size of nn)
        self.actions_size = agent_params['actions_size']
        self.exp_start = agent_params['exp_start']  # initial epsilon (exploration rate)
        self.exp_end = agent_params['exp_end']  # final epsilon
        self.exp_steps = agent_params['exp_steps']  # time steps to anneal epsilon
        self.gamma = agent_params['gamma']  # discount factor
        self.optimizer = agent_params['optimizer']
        self.learning_rate = agent_params['learning_rate']
        self.batch_size = agent_params['batch_size']
        self.target_update_freq = agent_params['target_update_freq']

        # self.tbCallBack = keras.callbacks.TensorBoard()
        self.model = self.dqn_model()
        self.target_model = self.dqn_model()
        self.update_target_model()

        self.t = 0  # current time to decide whether to start learning & anneal epsilon

    # loss function
    def huber_loss(self, y_true, y_pred, delta=1.0):
        x = y_true - y_pred
        return tf.where(
            tf.abs(x) < delta,
            tf.square(x) * 0.5,
            delta * (tf.abs(x) - 0.5 * delta)
        )

    # Neural Network for DQN
    def dqn_model(self):
        init = keras.initializers.glorot_normal()
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(16, input_dim=self.state_size, kernel_initializer=init, activation='relu'))
        model.add(keras.layers.Dense(16, kernel_initializer=init, activation='relu'))
        model.add(keras.layers.Dense(self.actions_size, kernel_initializer=init, activation='linear'))
        model.compile(loss=self.huber_loss, optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    # copy weights from model to target model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # return current epsilon according to t
    def interpolation(self, t):
        if t < self.learning_starts:
            return self.exp_start
        elif t < self.learning_starts + self.exp_steps:
            alpha = (self.exp_start - self.exp_end)/(-self.exp_steps)
            return self.exp_end + (t - (self.learning_starts + self.exp_steps)) * alpha
        else:
            return self.exp_end

    # select random action to explore / select action with max Q values
    def select_action(self, state):
        is_random = np.random.random() < self.interpolation(self.t) or self.t < self.learning_starts
        if is_random:
            return np.random.randint(self.actions_size)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values, axis=1)

    # store a transition into replay_buffer
    def store_effect(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    # step the env and store the transition
    def step_env(self, state):
        action = self.select_action(state)
        next_state, reward, done, info = self.env.step(action)
        self.store_effect(state, action, reward, next_state, done)

        if done:
            next_state = self.env.reset()
        return next_state  # self.last_obs = next_state

    # sample a random minibatch of transitions for training
    def sample(self):
        if self.replay_buffer.__len__() > self.batch_size:
            return random.sample(self.replay_buffer, self.batch_size)
        else:
            return []

    # train the neural network
    def train(self):
        if self.t > self.learning_starts:
            minibatch = self.sample()
            # _callback = 1
            state_batch = np.zeros((len(minibatch), self.state_size))
            target_q_batch = np.zeros((len(minibatch), self.actions_size))
            i = 0
            for state, action, reward, next_state, done in minibatch:
                target_q = self.target_model.predict(state)
                q_next_max = np.max(target_q)
                target_q_s = reward + self.gamma * q_next_max * (1-done)
                target_q[0][action] = target_q_s
                state_batch[i] = state[0]
                target_q_batch[i] = target_q[0]
                i += 1

                if i == 1:
                    print(target_q)
                '''
                target_q = self.target_model.predict(state)
                q_next_max = np.max(target_q)
                target_q_s = reward + self.gamma * q_next_max * (1-done)
                target_q[0][action] = target_q_s
                self.model.fit(state, target_q, epochs=1)
                '''
                '''
                if _callback:
                    _callback = 0
                    print(target_q)
                    self.model.fit(state, target_q, epochs=1, callback=[self.tbCallback])
                else:
                    self.model.fit(state, target_q, epochs=1)
                '''
            self.model.fit(state_batch, target_q_batch, epochs=1)
            if self.t % self.target_update_freq == 0:
                self.update_target_model()

        self.t += 1

    # save current weights of the neural network to path
    def save_nn_weights(self, path):
        self.model.save_weights(path)

    # load weights of the neural network from path
    def load_nn_weights(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)

    '''
    def prn_obj(obj):
        print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))
    '''
