import numpy as np
import math


class Env(object):
    def __init__(self, env_params):
        self.days = env_params['days']
        self.num_interval = env_params['num_interval']  # number of intervals per day
        self.action_size = env_params['action_size']
        self.action_interval = env_params['action_interval']
        self.total_inventory = env_params['total_inventory']
        self.customers = env_params['customers']  # customer model
        self.expected_price_mean = env_params['expected_price_mean']  # model of expected price of customer
        self.expected_price_var = env_params['expected_price_var']  # model of expected price of customer

        self.current_inventory = env_params['total_inventory']  # utilization = 1 - current_inventory/total_inventory
        self.current_day = env_params['days'] - 1  # remaining_time = current_day/days; --
        self.current_interval = 0  # 0 ~ (num_interval-1); ++

        self.p = env_params['init_price']  # price initialed according to the price given by LSTM
        # p = (pt-p0)/p0, where p0 is the lowest price(cost), pt is the price we set at time t,
        # p is like the percent of the profit, 0<=p<=1, i.e. p0 <= price <= 2 * p0

    # opening price
    def set_price(self, lstm_p):
        self.p = lstm_p

    # update profit percent according to action
    def update_p(self, action):
        # action is chosen by DQNAgent
        # for action_interval=1%, action_size=11,
        # action 0 corresponds to action_percent=95%, action 10 corresponds to action_percent=105%
        action_percent = 1 - ((self.action_size - 1) / 2 - action) * self.action_interval
        if (1 + self.p) * action_percent > 2:
            self.p = 1
        elif (1 + self.p) * action_percent < 1:
            self.p = 0
        else:
            self.p = (1 + self.p) * action_percent - 1

    # TODO change environment setting
    def reset(self, is_random=0, lstm_p=0.5, set_dis=0, poisson_lam=4, normal_mean=0.5, normal_var=0.1):
        # default: remaining_time=1 [=days], utilization=0%, price=0.5 [=1.5*lowest_price]
        # return [remaining_time, utilization, price]
        if is_random:
            self.current_inventory = np.random.randint(0, self.total_inventory)
            self.current_day = np.random.randint(0, self.days)
            self.current_interval = np.random.randint(0, self.num_interval)
            self.p = 0.3 + 0.4 * np.random.random()
        else:
            self.current_inventory = self.total_inventory
            self.current_day = self.days - 1
            self.current_interval = 0
            self.p = lstm_p
        if set_dis:
            self.customers = \
                np.random.poisson(poisson_lam, self.days * self.num_interval)
            self.expected_price_mean = normal_mean
            self.expected_price_var = normal_var
        else:
            self.customers = \
                np.random.poisson(self.total_inventory/self.days/self.num_interval, self.days * self.num_interval)
            self.expected_price_mean = 0.5
            self.expected_price_var = 0.1
        # the number of customers at current_interval of current_day:
        # self.customers[self.current_day*self.num_interval + self.num_interval - self.current_interval]
        remaining_time = self.current_day/self.days
        # remaining_time = self.current_day/self.days*(1+(self.num_interval - self.current_interval)/self.num_interval)
        utilization = 1 - self.current_inventory/self.total_inventory
        return np.array([[remaining_time, utilization, self.p]])

    # Env changes after action
    def step(self, action):
        # when (remaining_time=0 or utilization=100%): done=1
        reward = 0
        done = 0
        info = 0
        self.update_p(action)
        num_customers = math.ceil(self.customers[(self.current_day+1)*self.num_interval - self.current_interval-1])
        expected_price = np.random.normal(self.expected_price_mean, self.expected_price_var, num_customers)
        for i in range(num_customers):
            if expected_price[i] >= self.p:
                if self.current_inventory > 0:
                    self.current_inventory -= 1
                    reward += self.p
                else:
                    break

        if self.current_interval == self.num_interval - 1:
            self.current_day -= 1  # TODO: interact with LSTM to set the opening price
        self.current_interval = (self.current_interval+1) % self.num_interval

        remaining_time = self.current_day/self.days
        utilization = 1 - self.current_inventory/self.total_inventory
        next_state = np.array([[remaining_time, utilization, self.p]])

        if self.current_day < 0 or self.current_inventory <= 0:
            done = 1
        return next_state, reward, done, info
