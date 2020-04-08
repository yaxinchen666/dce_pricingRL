from dqn_s import DQNAgent
from env_1 import Env
from tensorflow import keras
import numpy as np
import os
import tqdm

LEARNING_START = 5000  # start learning after LEARNING_START steps
REPLAY_BUFFER_SIZE = int(1e4)
STATE_SIZE = 3  # utility rate, time, price
ACTION_SIZE = 11  # -5%~5%
EXP_START = 1.0  # exploration
EXP_END = 0.1
EXP_STEPS = int(1e4)  # steps to anneal epsilon
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQ = 1000

EPISODE = 20000
T = 240

DAYS = 30
NUM_INTERVAL = 4
ACTION_INTERVAL = 0.01
TOTAL_INVENTORY = 400
EXPECTED_PRICE_MEAN = 0.5
EXPECTED_PRICE_VAR = 0.1
INIT_PRICE = 0.5


if __name__ == "__main__":
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
    Env_params = {
        'days': DAYS,
        'num_interval': NUM_INTERVAL,
        'action_size': ACTION_SIZE,
        'action_interval': ACTION_INTERVAL,
        'total_inventory': TOTAL_INVENTORY,
        'customers': np.random.poisson(TOTAL_INVENTORY/DAYS/NUM_INTERVAL, DAYS*NUM_INTERVAL),
        'expected_price_mean': EXPECTED_PRICE_MEAN,
        'expected_price_var': EXPECTED_PRICE_VAR,
        'init_price': 0.5
    }
    T_Env = Env(Env_params)
    Agent = DQNAgent(env=T_Env, agent_params=DQNAgent_params)
    # Agent.load_nn_weights("")

    for episode in tqdm.tqdm(range(EPISODE)):
        if episode % 100 == 0:
            print("Episode: ", episode)
            poisson_l = np.random.randint(0.5*TOTAL_INVENTORY/DAYS/NUM_INTERVAL, 1.5*TOTAL_INVENTORY/DAYS/NUM_INTERVAL)
            state = T_Env.reset(is_random=1, set_dis=1, poisson_lam=poisson_l)
        else:
            lstm_price = 0.4 + 0.2 * np.random.random()  # TODO
            state = T_Env.reset(lstm_p=lstm_price)

        for t in range(T):
            state = Agent.step_env(state)
            Agent.train()

        if episode % 2000 == 0:
            if not os.path.exists("./nnWeight"):
                os.makedirs("./nnWeight")
            path = "./nnWeight/weight"+str(int(episode))+".h5"
            Agent.save_nn_weights(path)
