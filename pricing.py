from dqn_s import DQNAgent
from env_1 import Env
from tensorflow import keras
import os
import numpy as np

LEARNING_START = 5000  # start learning after LEARNING_START steps
REPLAY_BUFFER_SIZE = int(1e4)
STATE_SIZE = 3  # time, utilization, price
ACTION_SIZE = 11  # -5%~5%
EXP_START = 1.0  # exploration
EXP_END = 0.1
EXP_STEPS = int(1e4)  # steps to anneal epsilon
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQ = 1000

DAYS = 30
NUM_INTERVAL = 4
ACTION_INTERVAL = 0.01
TOTAL_INVENTORY = 400
EXPECTED_PRICE_MEAN = 0.5
EXPECTED_PRICE_VAR = 0.1
INIT_PRICE = 0.5


if __name__ == "__main__":
    # using dqn
    Env_params = {
        'days': DAYS,
        'num_interval': NUM_INTERVAL,
        'action_size': ACTION_SIZE,
        'action_interval': ACTION_INTERVAL,
        'total_inventory': TOTAL_INVENTORY,
        'customers': np.random.poisson(TOTAL_INVENTORY/DAYS/NUM_INTERVAL, DAYS*NUM_INTERVAL),
        'expected_price_mean': EXPECTED_PRICE_MEAN,
        'expected_price_var': EXPECTED_PRICE_VAR,
        'init_price': INIT_PRICE
    }
    DQN_Env = Env(Env_params)
    DQNAgent_params = {
        'learning_starts': LEARNING_START,
        'replay_buffer_size': REPLAY_BUFFER_SIZE,
        'state_size': STATE_SIZE,
        'actions_size': ACTION_SIZE,
        'exp_start': EXP_START,
        'exp_end': EXP_END,
        'exp_steps': EXP_STEPS,
        'gamma': GAMMA,
        'optimizer': keras.optimizers.Adam(),
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'target_update_freq': TARGET_UPDATE_FREQ
    }
    Agent = DQNAgent(env=DQN_Env, agent_params=DQNAgent_params)
    path = "./nnWeights/nnWeight_0409_3/weight320.h5"  # load model
    if os.path.exists(path):
        print("Load weights from ", path)
        Agent.load_nn_weights(path)

    dqn_reward = []
    for i in range(100):
        state = Agent.env.reset()
        total_reward = 0
        done = 0
        for day in range(DAYS):
            # print(Agent.env.current_inventory, Agent.env.p)
            for interval in range(NUM_INTERVAL):
                q_values = Agent.model.predict(state)
                action = np.argmax(q_values, axis=1)
                state, reward, done, info = Agent.env.step(action)
                total_reward += reward
                if done:
                    break
            if done:
                break
        dqn_reward.append(total_reward)
    print("Total reward(using dqn):", np.mean(dqn_reward))

    # randomly pricing
    reward = []
    for iter in range(100):
        inventory = TOTAL_INVENTORY
        current_reward = 0.0
        for day in range(DAYS):
            price = np.random.random()
            customers = \
                np.random.poisson(TOTAL_INVENTORY/DAYS, DAYS)
            expected_price = np.random.normal(EXPECTED_PRICE_MEAN, EXPECTED_PRICE_VAR, customers[day])
            for i in range(len(expected_price)):
                if expected_price[i] >= price:
                    if inventory > 0:
                        inventory = inventory - 1
                        current_reward += price
        reward.append(current_reward)
    print("Total reward(randomly pricing):", np.mean(reward))

    # fix price
    reward = []
    for iter in range(100):
        current_reward = 0.0
        inventory = TOTAL_INVENTORY
        for day in range(DAYS):
            price = 0.5
            customers = \
                np.random.poisson(TOTAL_INVENTORY/DAYS, DAYS)
            expected_price = np.random.normal(EXPECTED_PRICE_MEAN, EXPECTED_PRICE_VAR, customers[day])
            for i in range(len(expected_price)):
                if expected_price[i] >= price:
                    if inventory > 0:
                        inventory = inventory - 1
                        current_reward += price
        reward.append(current_reward)
    print("Total reward(price=100):", np.mean(reward))
