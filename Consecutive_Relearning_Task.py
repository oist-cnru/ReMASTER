"""
usage: Consecutive_Relearning_Task.py [-h] [--model=MTSRNN] [--noise=0.2] [--singlev=False] [--lowstop3=False] [--seed=0]

Codes for consecutive relearning task using ReMASTER or LSTM.

optional arguments:
 -h, --help     show this help message and exit
 --model        The model used, either MTSRNN or LSTM (default: MTSRNN)
 --noise        Scale of initial neuronal noise, only works for MTSRNN (default: 0.2)
 --singlev      If True, the higher level does not learn the value function with gamma2, only works for MTSRNN (default: False)
 --lowstop3     If True, the lower-level synaptic weight will be frozen in phase 3, only works for MTSRNN (default: False)
 --seed         Random seed (default: 0)
"""

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import os, time, warnings
import argparse
from copy import deepcopy
from datetime import datetime
from gym.utils import seeding
from gym import spaces
import gym


now = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    help="The model used, either MTSRNN or LSTM (default: MTSRNN)", default='MTSRNN')
parser.add_argument('--noise', type=float,
                    help="Scale of initial neuronal noise, only works for MTSRNN (default: 0.2)", default=0.2)
parser.add_argument('--singlev', type=bool,
                    help="If True, the higher level does not learn the value function with gamma2, only works for MTSRNN (default: False)",
                    default=False)
parser.add_argument('--lowstop3', type=bool,
                    help="If True, the lower-level synaptic weight will be frozen in phase 3, only works for MTSRNN (default: False)",
                    default=False)
parser.add_argument('--seed', type=int, help="Random seed (default: 0)", default=0)

args = parser.parse_args()

savepath = '../data/'
perfsavepath = '../perf_data/'

model_name = args.model
scale_epsilon = [args.noise, args.noise]
singleV = args.singlev
lowstop3 = args.lowstop3
seed = args.seed

filename = 'ReMASTER-Consecutive-Relearning-Task-' + model_name


if os.path.exists(savepath):
    warnings.warn('{} exists (possibly so do data).'.format(savepath))
else:
    os.makedirs(savepath)

if os.path.exists(perfsavepath):
    warnings.warn('{} exists (possibly so do data).'.format(perfsavepath))
else:
    os.makedirs(perfsavepath)


#############################################################################################################
###############################        Structure & Hyper-Parameters      ####################################
#############################################################################################################
# ------------- set the random seed ----------------

np.random.seed(seed)
tf.set_random_seed(seed)

plotting = False
recording = False  # Tensorflow summary
saving = True  # matfile
testing = False
saving_interval = 50  # interval of episodes to save

# ------------- Task setting -----------------
num_episodes = 12000
max_steps = 128
step_record = 50
buffer_size = 500000
radius = 0.5  # radius of the two-wheel robot

# ------------- BPTT setting -----------------
truncated_backprop_length = 25
step_train_begin = 20 * max_steps
step_train_freq = 2

# ------------- inputs&outputs -----------------
dim_position = 2
dim_action = 2
dim_input = 12
dim_output = dim_action + 1

# ------------- Exploration setting -----------------
noise_decrease = 3.3333e-4
min_noise = 0.1  # for motor noise
SIG_EPS = 1e-3  # to avoid divergence due to sig ~= 0
noise_scale = 0.75

episode_test = 40  # Every 40 episode, there is a tesing episode without any noise input
episode_plot = 10

# ------------- Network Layers -----------------
# N_level = 2
if model_name == 'LSTM':
    N_level = 1
    num_state = 75
elif model_name == 'MTSRNN':
    N_level = 2
    num_state = np.array([100, 50], dtype=np.int)
    total_num_state = np.sum(num_state)
else:
    pass

tau = np.array([2, 8], dtype=np.float32)

# ------------- Initial weights -----------------
init_weight_range = 0.05
init_weight_mean = 0.0
bias_shift = 0.0  # for making units activated, similiar to relu
input_connect_std = 0.5  # initial weight std for input connection

# ------------- agent settings -----------------
speed = 0.8

# ------------- Learning parameters -----------------
learning_rate = 2e-4
batch_size = 16

# These betas are the coefficients for alpha of Adam optimizer
beta = 0.0  # proportion of regularization loss
beta_v = 1.5  # proportion of value learning
beta_a = 0.5  # proportion of action learning

K = 0.16

if model_name == 'MTSRNN':
    gamma = 1.0 - K / tau  # discount factor
elif model_name == 'LSTM':
    gamma = 1.0 - K / (np.prod(tau) ** (1.0 / len(tau)))

optimizer_name = 'RMSProp'


np_actfun = lambda x: np.tanh(x)
tf_actfun = lambda x: tf.tanh(x)


def get_noise(noise_input, dimension):
    glbs = globals()
    noise_decrease = glbs["noise_decrease"]

    if not "scale" in noise_input:
        noise_input["scale"] == 1

    if not "type" in noise_input:
        noise_input["type"] == 'OU'

    if "episode" in noise_input:
        episode = noise_input["episode"]

    amp = noise_input["scale"] * np.exp(-noise_decrease * episode) + min_noise

    rand = np.zeros([dimension])

    if noise_input["type"] == 'OU' and not (testing and episode % episode_test == 0):
        th_ou = 0.3
        mu_ou = 0
        sig_ou = amp * np.sqrt(th_ou) * np.sqrt(2)
        dt_ou = 1
        rand_step = np.random.normal(0, 1, size=[dimension])
        rand_0 = noise_input["prev_noise"]
        rand_0 = rand_0 + th_ou * (mu_ou - rand_0) * dt_ou + sig_ou * np.sqrt(dt_ou) * rand_step
        rand = np.clip(rand_0, -3 * amp, 3 * amp)

    elif noise_input["type"] == 'Gaussian' and not (testing and episode % episode_test == 0):
        rand = np.clip(np.random.normal(0, amp, size=[dimension]), -3 * amp, 3 * amp)

    return rand


def get_input(x, dim_input):
    '''
    Radical Basis Functions, extracting the feature of input x.
    '''
    y = np.reshape(x, [1, dim_input])

    return y


def flatten(x_levels, dim):
    """
    :param x_levels: a list which has N_levels of elements
    :return: a flattened narray
    """
    num_state_sum = np.zeros(N_level + 1, dtype=int)
    num_state_sum[0] = 0
    for lev in range(N_level):
        num_state_sum[lev + 1] = np.sum(num_state[0:lev + 1])

    x_flat = np.zeros([dim, ])
    for lev in range(N_level):
        x_flat[num_state_sum[lev]:num_state_sum[lev + 1]] = np.reshape(x_levels[lev], [-1, ])

    return x_flat


def levelize(x_flat, num_state=num_state):
    """
    :param x_flat: a flattened vector
    :param num_state: num_state for each level
    :return: a list which has N_levels of elements
    """
    num_state_sum = np.zeros(N_level + 1, dtype=int)
    num_state_sum[0] = 0
    for lev in range(N_level):
        num_state_sum[lev + 1] = np.sum(num_state[0:lev + 1])

    x_levels = []
    for lev in range(N_level):
        x_levels.append(np.reshape(x_flat[num_state_sum[lev]:num_state_sum[lev + 1]], [1, -1]))

    return x_levels


class TaskT(gym.Env):
    metadata = {'name': 'TaskT', 'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
    spec = {'id': 'TaskT'}

    def __init__(self, sections=1, seq='RGB', final_reward=False, reward_obs=True, R=4, saving=False,
                 log_dir="./TaskT_log/", reward_scales=[0.8, 2.0, 5.0]):
        """
        Sequential target reaching task.
        :param sections: how many targets to reach to finish the task
        :param seq: any combination of 'R', 'G', 'B' to indicated the required sequence of target-reaching.
        :param final_reward: if True, only final target provides reward, otherwise all targets provide reward.
        :param reward_obs: whether reward is one element of observation
        :param R: difficulty (distance between targets)
        :param saving: whether to save steps/rewards into txt file
        :param log_dir: directory to save steps/rewards
        """
        self.sections = sections
        self.saving = saving
        self.log_dir = log_dir
        self.final_reward = final_reward
        self.reward_obs = reward_obs
        self.sequence = seq
        self.R = R
        self.reward = 0.0
        self.reward_signal = 0.0
        self.dim_position = 2
        self.dim_action = 2
        self.speed = 0.8
        self.radius = 0.5
        self.max_steps = 128
        self.steps = 0

        self.reward1 = reward_scales[0]
        self.reward2 = reward_scales[1]
        self.reward3 = reward_scales[2]

        self.init_position = np.array([7.5, 7.5], dtype=np.float32)
        self.init_position[0] += np.float32(15 * (np.random.rand() - 0.5))
        self.init_position[1] += np.float32(15 * (np.random.rand() - 0.5))
        self.old_position = self.init_position
        self.new_position = self.init_position
        self.orientation = 2 * np.pi * np.random.rand()

        self.init_state = 0
        self.size = 1
        self.action_space = spaces.Box(low=-1., high=1., shape=(2,))
        if reward_obs:
            self.observation_space = spaces.Box(low=-1., high=5., shape=(12,))
        else:
            self.observation_space = spaces.Box(low=-1., high=1., shape=(11,))

        self.reward_range = (-np.Inf, np.Inf)

        self._seed()

        if self.saving:
            if os.path.exists(log_dir):
                warnings.warn('{} exists (possibly so do data).'.format(log_dir))
            else:
                os.makedirs(log_dir)

            path = self.log_dir + 'TaskT' + '.txt'
            self.file_pointer = open(path, 'w+')

        self.red_position = np.float32(R * (np.random.rand(self.dim_position) - 0.5)) + np.array([7.5, 7.5],
                                                                                                 dtype=np.float32)
        while True:
            self.green_position = np.float32(R * (np.random.rand(self.dim_position) - 0.5)) + np.array([7.5, 7.5],
                                                                                                       dtype=np.float32)
            if (np.sum((self.red_position - self.green_position) ** 2)) > 2:
                break
        while True:
            self.blue_position = np.float32(R * (np.random.rand(self.dim_position) - 0.5)) + np.array([7.5, 7.5],
                                                                                                      dtype=np.float32)
            if (np.sum((self.blue_position - self.green_position) ** 2)) > 2 and (
            np.sum((self.blue_position - self.red_position) ** 2)) > 2:
                break

        self.first_experience = 0
        self.second_experience = 0
        self.third_experience = 0

        if self.sequence[0] == 'R':
            self.first_position = self.red_position
        elif self.sequence[0] == 'G':
            self.first_position = self.green_position
        elif self.sequence[0] == 'B':
            self.first_position = self.blue_position

        if self.sections >= 2:
            if self.sequence[1] == 'R':
                self.second_position = self.red_position
            elif self.sequence[1] == 'G':
                self.second_position = self.green_position
            elif self.sequence[1] == 'B':
                self.second_position = self.blue_position

        if self.sections >= 3:
            if self.sequence[2] == 'R':
                self.third_position = self.red_position
            elif self.sequence[2] == 'G':
                self.third_position = self.green_position
            elif self.sequence[2] == 'B':
                self.third_position = self.blue_position

        self.done = 0

        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.reward = 0.0
        self.steps = 0
        self.reward_signal = 0.0
        self.init_position = np.array([7.5, 7.5], dtype=np.float32)
        self.init_position[0] += np.float32(15 * (np.random.rand() - 0.5))
        self.init_position[1] += np.float32(15 * (np.random.rand() - 0.5))
        self.old_position = self.init_position
        self.new_position = self.init_position
        self.orientation = 2 * np.pi * np.random.rand()

        R = self.R
        self.red_position = np.float32(R * (np.random.rand(self.dim_position) - 0.5)) + np.array([7.5, 7.5],
                                                                                                 dtype=np.float32)
        while True:
            self.green_position = np.float32(R * (np.random.rand(self.dim_position) - 0.5)) + np.array([7.5, 7.5],
                                                                                                       dtype=np.float32)
            if (np.sum((self.red_position - self.green_position) ** 2)) > 2:
                break
        while True:
            self.blue_position = np.float32(R * (np.random.rand(self.dim_position) - 0.5)) + np.array([7.5, 7.5],
                                                                                                      dtype=np.float32)
            if (np.sum((self.blue_position - self.green_position) ** 2)) > 2 and (
            np.sum((self.blue_position - self.red_position) ** 2)) > 2:
                break

        self.first_experience = 0
        self.second_experience = 0
        self.third_experience = 0

        if self.sequence[0] == 'R':
            self.first_position = self.red_position
        elif self.sequence[0] == 'G':
            self.first_position = self.green_position
        elif self.sequence[0] == 'B':
            self.first_position = self.blue_position

        if self.sections >= 2:
            if self.sequence[1] == 'R':
                self.second_position = self.red_position
            elif self.sequence[1] == 'G':
                self.second_position = self.green_position
            elif self.sequence[1] == 'B':
                self.second_position = self.blue_position

        if self.sections >= 3:
            if self.sequence[2] == 'R':
                self.third_position = self.red_position
            elif self.sequence[2] == 'G':
                self.third_position = self.green_position
            elif self.sequence[2] == 'B':
                self.third_position = self.blue_position

        self.done = 0
        return self.get_obs()

    def get_obs(self):

        lambd = 3.0

        position = self.new_position
        theta = self.orientation

        red_dis = np.sqrt(np.sum((position - self.red_position) ** 2))
        green_dis = np.sqrt(np.sum((position - self.green_position) ** 2))
        blue_dis = np.sqrt(np.sum((position - self.blue_position) ** 2))

        if 0 <= theta and theta < np.pi / 2:
            dw1 = min((15 - position[1]) / abs(np.sin(theta)), (15 - position[0]) / abs(np.cos(theta)))
            dw2 = min((position[1] - 0) / abs(np.sin(theta)), (position[0] - 0) / abs(np.cos(theta)))
        elif np.pi / 2 <= theta and theta < np.pi:
            dw1 = min((15 - position[1]) / abs(np.sin(theta)), (position[0] - 0) / abs(np.cos(theta)))
            dw2 = min((position[1] - 0) / abs(np.sin(theta)), (15 - position[0]) / abs(np.cos(theta)))
        elif np.pi <= theta and theta < 3 * np.pi / 2:
            dw1 = min((position[1] - 0) / abs(np.sin(theta)), (position[0] - 0) / abs(np.cos(theta)))
            dw2 = min((15 - position[1]) / abs(np.sin(theta)), (15 - position[0]) / abs(np.cos(theta)))
        else:
            dw1 = min((position[1] - 0) / abs(np.sin(theta)), (15 - position[0]) / abs(np.cos(theta)))
            dw2 = min((15 - position[1]) / abs(np.sin(theta)), (position[0] - 0) / abs(np.cos(theta)))

        tr = np.arctan2(self.red_position[1] - position[1], self.red_position[0] - position[0]) - theta
        tg = np.arctan2(self.green_position[1] - position[1], self.green_position[0] - position[0]) - theta
        tb = np.arctan2(self.blue_position[1] - position[1], self.blue_position[0] - position[0]) - theta

        if self.reward_obs:
            obs = np.array([np.exp(-red_dis / lambd),
                            np.exp(-green_dis / lambd),
                            np.exp(-blue_dis / lambd),
                            np.exp(-dw1 / lambd),
                            np.exp(-dw2 / lambd),
                            np.sin(tr),
                            np.cos(tr),
                            np.sin(tg),
                            np.cos(tg),
                            np.sin(tb),
                            np.cos(tb),
                            self.reward_signal])
        else:
            obs = np.array([np.exp(-red_dis / lambd),
                            np.exp(-green_dis / lambd),
                            np.exp(-blue_dis / lambd),
                            np.exp(-dw1 / lambd),
                            np.exp(-dw2 / lambd),
                            np.sin(tr),
                            np.cos(tr),
                            np.sin(tg),
                            np.cos(tg),
                            np.sin(tb),
                            np.cos(tb)])
        return obs

    def get_init_position(self):
        return self.init_position

    def reward_fun(self):

        position = self.old_position
        new_position = self.new_position

        if new_position[0] > 15 or new_position[0] < 0 or new_position[1] > 15 or new_position[1] < 0:
            r = -0.1
        # self.done = 1
        else:
            if not self.first_experience:
                target_position = self.first_position
                dis2 = np.sum((new_position - target_position) ** 2)
                if dis2 < 0.16:
                    self.first_experience = 1
                    if self.sections == 1:
                        self.done = 1
                    r = self.reward1 / (1 + np.sqrt(dis2)) if (self.sections == 1 or (not self.final_reward)) else 0.0
                else:
                    r = 0.0
            elif self.sections >= 2 and (not self.second_experience):
                target_position = self.second_position
                dis2 = np.sum((new_position - target_position) ** 2)
                if dis2 < 0.16:
                    self.second_experience = 1
                    if self.sections == 2:
                        self.done = 1
                    r = self.reward2 / (1 + np.sqrt(dis2)) if (self.sections == 2 or (not self.final_reward)) else 0.0
                else:
                    r = 0.0
            elif self.sections >= 3:
                target_position = self.third_position
                dis2 = np.sum((new_position - target_position) ** 2)
                if dis2 < 0.16:
                    self.third_experience = 1
                    if self.sections == 3:
                        self.done = 1
                    r = self.reward3 / (1 + np.sqrt(dis2)) if self.sections == 3 else 0.0
                else:
                    r = 0.0
            else:
                r = 0.0
        return r

    def step(self, action, saving=None):

        if self.done:
            warnings.warn("Task already done!! Not good to continue!")

        if saving is None:
            saving = self.saving

        self.old_position = deepcopy(self.new_position)
        self.steps += 1
        action = np.reshape(action, [self.dim_action])

        current_action_1 = action[0]
        current_action_2 = action[1]

        self.new_position[0] = self.old_position[0] + np.clip(
            self.speed * np.cos(self.orientation) * (current_action_1 + current_action_2) / 2, -self.speed, self.speed)
        self.new_position[1] = self.old_position[1] + np.clip(
            self.speed * np.sin(self.orientation) * (current_action_1 + current_action_2) / 2, -self.speed, self.speed)

        self.orientation += np.clip((current_action_1 - current_action_2) / 2 / self.radius, -np.pi,
                                    np.pi)
        while self.orientation >= 2 * np.pi:
            self.orientation -= 2 * np.pi
        while self.orientation < 0:
            self.orientation += 2 * np.pi

        self.reward = self.reward_fun()
        self.reward_signal = self.reward

        if self.new_position[0] > 15 or self.new_position[0] < 0 or self.new_position[1] > 15 or self.new_position[
            1] < 0:
            self.new_position = self.old_position

        if self.steps >= self.max_steps:
            self.done = 1

        if saving and self.saving:
            self.savelog(self.reward, self.done)

        return self.get_obs(), self.reward, self.done, {}

    def savelog(self, r, d):
        if self.saving:
            self.file_pointer.write("%f, %f \n" % (r, d))
        else:
            warnings.warn('cannot save!')

    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500

        world_width = 15
        scale = screen_width / world_width
        target_radius = 0.3
        car_radius = 0.1

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(0, 15, 0, 15)
            self.red = rendering.make_circle(target_radius)
            self.green = rendering.make_circle(target_radius)
            self.blue = rendering.make_circle(target_radius)
            self.red.set_color(1., 0., 0.)
            self.green.set_color(0., 1., 0.)
            self.blue.set_color(0., 0., 1.)
            self.red_trans = rendering.Transform()
            self.green_trans = rendering.Transform()
            self.blue_trans = rendering.Transform()
            self.red.add_attr(self.red_trans)
            self.green.add_attr(self.green_trans)
            self.blue.add_attr(self.blue_trans)
            self.viewer.add_geom(self.red)
            self.viewer.add_geom(self.green)
            self.viewer.add_geom(self.blue)

            self.car = rendering.make_circle(car_radius)
            self.car.set_color(0., 0., 0.)
            self.car_trans = rendering.Transform()
            self.car.add_attr(self.car_trans)
            self.viewer.add_geom(self.car)

            self.car_orientation = rendering.make_polygon([(0, 0.2), (0, -0.2), (0.4, 0)])
            self.car_orientation.set_color(1., 0., 1.)
            self.rotation = rendering.Transform()
            self.car_orientation.add_attr(self.rotation)
            self.viewer.add_geom(self.car_orientation)

        self.red_trans.set_translation(self.red_position[0], self.red_position[1])
        self.green_trans.set_translation(self.green_position[0], self.green_position[1])
        self.blue_trans.set_translation(self.blue_position[0], self.blue_position[1])
        self.car_trans.set_translation(self.new_position[0], self.new_position[1])
        self.rotation.set_translation(self.new_position[0], self.new_position[1])
        self.rotation.set_rotation(self.orientation)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def getWindow(self):
        return self.viewer

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class MTSRNN():
    def __init__(self):
        """
        Build the graph.
        """
        self.session = tf.InteractiveSession()
        self.define_weights()
        self.build_graph_batch()
        self.initial_variables()

    def __call__(self, previous_c_flat, previous_mu_flat, current_input):

        glbs = globals()
        episode = glbs["episode"]
        current_e = np.random.normal(size=[1, total_num_state])  # for matrix multiplication
        previous_c_flat = np.reshape(previous_c_flat, [1, total_num_state])
        previous_mu_flat = np.reshape(previous_mu_flat, [1, total_num_state])
        inputs = np.zeros([1, truncated_backprop_length, dim_input])
        inputs[0, 0, :] = current_input[0, :]

        v, pi, c, mu, z, sig = self.session.run([self.onestep_value_output_levels,
                                                 self.onestep_policy_output_levels,
                                                 self.onestep_c_levels,
                                                 self.onestep_mu_levels,
                                                 self.onestep_z_levels,
                                                 self.onestep_sig_levels],
                                                feed_dict={self.batchX_ph: inputs,
                                                           self.init_c: previous_c_flat,
                                                           self.init_mu: previous_mu_flat,
                                                           self.coef_eps: np.exp(-noise_decrease * episode)})
        v = np.reshape(v, [N_level, 1])
        pi = np.reshape(pi, [N_level, dim_action])

        v = np.swapaxes(np.reshape(v, [N_level, -1, 1]), 0, 1)
        v = np.reshape(v[0], [N_level, 1])
        pi = np.swapaxes(np.reshape(pi, [N_level, -1, dim_action]), 0, 1)
        pi = np.reshape(pi[0], [N_level, dim_action])

        return v, pi, c, mu, z, sig

    def define_weights(self):
        self.num_state_sum = np.zeros(N_level + 1, dtype=int)
        self.num_state_sum[0] = 0
        for lev in range(N_level):
            self.num_state_sum[lev + 1] = np.sum(num_state[0:lev + 1])
        with tf.variable_scope("weights") as scope:
            self.Wyu = tf.Variable(tf.truncated_normal([dim_input, num_state[0]], mean=0.0, stddev=input_connect_std),
                                   name='Wyu', dtype=tf.float32)
            self.Wcs_this = []
            self.Wcu_this = []
            self.Wcu_lower = []
            self.Wcu_higher = []
            self.bu = []
            self.bs = []
            self.Wo = []
            self.bo = []
            self.Wo_mua = []
            self.bo_mua = []
            self.Wo_siga = []
            self.bo_siga = []
            for lev in range(N_level):
                self.Wcu_this.append(tf.Variable(
                    tf.random_uniform([num_state[lev], num_state[lev]], minval=-init_weight_range + init_weight_mean,
                                      maxval=init_weight_range + init_weight_mean), name='Wcu' + str(lev + 1),
                    dtype=tf.float32))
                self.Wcs_this.append(tf.Variable(tf.zeros([num_state[lev], num_state[lev]]), name='Wcs' + str(lev + 1),
                                                 dtype=tf.float32, trainable=False))
                if lev > 0:
                    self.Wcu_lower.append(tf.Variable(
                        tf.random_uniform([num_state[lev - 1], num_state[lev]],
                                          minval=-init_weight_range + init_weight_mean,
                                          maxval=init_weight_range + init_weight_mean), name='Wcu_lower' + str(lev + 1),
                        dtype=tf.float32))
                if lev < N_level - 1:
                    self.Wcu_higher.append(tf.Variable(
                        tf.random_uniform([num_state[lev + 1], num_state[lev]],
                                          minval=-init_weight_range + init_weight_mean,
                                          maxval=init_weight_range + init_weight_mean),
                        name='Wcu_higher' + str(lev + 1), dtype=tf.float32))
                self.bu.append(
                    tf.Variable(tf.zeros([num_state[lev]]) + bias_shift, name='bu' + str(lev + 1), dtype=tf.float32))
                self.bs.append(
                    tf.Variable(tf.zeros([num_state[lev]]) + bias_shift + 2 * np.log(scale_epsilon[lev]),
                                name='bs' + str(lev + 1), dtype=tf.float32, trainable=False))

                isfirstlayer = lev == 0

                if not isfirstlayer:
                    self.Wo_mua.append(
                        tf.Variable(tf.zeros([num_state[lev], dim_action]), name='Wo_mua' + str(lev + 1),
                                    dtype=tf.float32, trainable=isfirstlayer))
                    self.bo_mua.append(
                        tf.Variable(tf.zeros([dim_action]), name='bo_mua' + str(lev + 1),
                                    dtype=tf.float32, trainable=isfirstlayer))

                    self.Wo_siga.append(
                        tf.Variable(tf.zeros([num_state[lev], dim_action]), name='Wo_siga' + str(lev + 1),
                                    dtype=tf.float32, trainable=isfirstlayer))
                    self.bo_siga.append(
                        tf.Variable(tf.zeros([dim_action]), name='bo_siga' + str(lev + 1),
                                    dtype=tf.float32, trainable=isfirstlayer))

                else:
                    self.Wo_mua.append(
                        tf.Variable(tf.random_uniform([num_state[lev], dim_action], minval=-init_weight_range,
                                                      maxval=init_weight_range), name='Wo_mua' + str(lev + 1),
                                    dtype=tf.float32))
                    self.bo_mua.append(
                        tf.Variable(tf.random_uniform([dim_action], minval=-init_weight_range,
                                                      maxval=init_weight_range), name='bo_mua' + str(lev + 1),
                                    dtype=tf.float32))

                    self.Wo_siga.append(
                        tf.Variable(tf.random_uniform([num_state[lev], dim_action], minval=-init_weight_range,
                                                      maxval=init_weight_range), name='Wo_siga' + str(lev + 1),
                                    dtype=tf.float32))
                    self.bo_siga.append(
                        tf.Variable(tf.random_uniform([dim_action], minval=-init_weight_range,
                                                      maxval=init_weight_range), name='bo_siga' + str(lev + 1),
                                    dtype=tf.float32))

                self.Wo.append(tf.Variable(tf.random_uniform([num_state[lev], 1], minval=-init_weight_range,
                                                             maxval=init_weight_range), name='Wo' + str(lev + 1),
                                           dtype=tf.float32))
                self.bo.append(tf.Variable(tf.random_uniform([1], minval=-init_weight_range,
                                                             maxval=init_weight_range), name='bo' + str(lev + 1),
                                           dtype=tf.float32))

    def gaussian_noisy_layer(self, input_layer, sig, scale):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=scale, dtype=tf.float32)
        return input_layer + sig * noise

    def build_graph_batch(self):
        with tf.name_scope('input'):
            self.v_step_ph = tf.placeholder(tf.float32, [None, truncated_backprop_length], name='V_step')  # For padding
            self.batchX_ph = tf.placeholder(tf.float32, [None, truncated_backprop_length, dim_input], name='input')
            self.batchP_ph = tf.placeholder(tf.float32, [None, truncated_backprop_length, N_level, dim_action],
                                            name='target_policy')  # target policy(mean action)
            self.batchV_ph = tf.placeholder(tf.float32, [None, truncated_backprop_length, N_level, 1],
                                            name='target_value')  # target value V for each level
            self.init_c = tf.placeholder(tf.float32, [None, total_num_state], name='init_c')
            self.init_mu = tf.placeholder(tf.float32, [None, total_num_state], name='init_mu')
            self.reward_ph = tf.placeholder(tf.float32, None, name='episode_mean_reward')
            self.coef_eps = tf.placeholder(tf.float32, None, name='coef_epsilon')

            self.input_series = tf.unstack(self.batchX_ph, axis=1, name='input_series')

            self.policy_target_series = tf.unstack(self.batchP_ph, axis=1, name='policy_target_series')

            self.value_target_series = tf.unstack(self.batchV_ph, axis=1, name='value_target_series')

            self.init_c_levels = []
            self.init_mu_levels = []
            for lev in range(N_level):
                self.init_c_levels.append(self.init_c[:, self.num_state_sum[lev]:self.num_state_sum[lev + 1]])
                self.init_mu_levels.append(self.init_mu[:, self.num_state_sum[lev]:self.num_state_sum[lev + 1]])

        with tf.name_scope('MTRNN'):

            self.c_series = []
            self.mu_series = []
            self.z_series = []
            self.sig_series = []

            self.value_output_series = []
            self.policy_output_series = []

            self.prev_c_levels = self.init_c_levels
            self.prev_mu_levels = self.init_mu_levels

            for stp, curr_input in enumerate(self.input_series):

                self.curr_value_output_levels = []
                self.curr_policy_output_levels = []
                self.curr_c_levels = []
                self.curr_mu_levels = []
                self.curr_z_levels = []
                self.curr_sig_levels = []

                for lev in range(N_level):
                    with tf.name_scope('RNN_Level_' + str(lev + 1)):
                        self.prev_mu = self.prev_mu_levels[lev]
                        self.prev_c_this = tf.reshape(self.prev_c_levels[lev], [-1, num_state[lev]])

                        if lev == 0:  # lowest layer
                            self.curr_input = tf.reshape(curr_input, [-1, dim_input])
                            self.prev_c_higher = tf.reshape(self.prev_c_levels[lev + 1], [-1, num_state[lev + 1]])
                            self.curr_mu = (1 - 1 / tau[lev]) * self.prev_mu + 1 / tau[lev] * (
                                tf.matmul(self.curr_input, self.Wyu) +
                                tf.matmul(self.prev_c_this, self.Wcu_this[lev]) +
                                tf.matmul(self.prev_c_higher, self.Wcu_higher[lev]) + self.bu[lev])
                        elif lev == N_level - 1:
                            self.curr_c_lower = tf.reshape(self.curr_c_levels[lev - 1], [-1, num_state[lev - 1]])
                            self.curr_mu = (1 - 1 / tau[lev]) * self.prev_mu + 1 / tau[lev] * (
                                tf.matmul(self.prev_c_this, self.Wcu_this[lev]) +
                                tf.matmul(self.curr_c_lower, self.Wcu_lower[lev - 1]) + self.bu[lev])
                        else:
                            self.prev_c_higher = tf.reshape(self.prev_c_levels[lev + 1], [-1, num_state[lev + 1]])
                            self.curr_c_lower = tf.reshape(self.curr_c_levels[lev - 1], [-1, num_state[lev - 1]])
                            self.curr_mu = (1 - 1 / tau[lev]) * self.prev_mu + 1 / tau[lev] * (
                                tf.matmul(self.prev_c_this, self.Wcu_this[lev]) +
                                tf.matmul(self.prev_c_higher, self.Wcu_higher[lev]) +
                                tf.matmul(self.curr_c_lower, self.Wcu_lower[lev - 1]) + self.bu[lev])

                        self.curr_sig = tf.exp(0.5 * (tf.matmul(self.prev_c_this, self.Wcs_this[lev]) + self.bs[lev]))

                        self.curr_z = self.gaussian_noisy_layer(self.curr_mu, self.curr_sig * self.coef_eps, scale=1)
                        self.curr_z_levels.append(self.curr_z)
                        self.curr_c_levels.append(tf_actfun(self.curr_z))

                        self.curr_value_output = tf.matmul(self.curr_c_levels[lev], self.Wo[lev]) + self.bo[lev]
                        self.curr_policy_output = tf.tanh(
                            tf.matmul(self.curr_c_levels[lev], self.Wo_mua[lev]) + self.bo_mua[lev])
                        self.curr_mu_levels.append(self.curr_mu)
                        self.curr_sig_levels.append(self.curr_sig)
                        self.curr_value_output_levels.append(self.curr_value_output)
                        self.curr_policy_output_levels.append(self.curr_policy_output)

                if stp == 0:  # for forward computing
                    self.onestep_value_output_levels = tf.reshape(self.curr_value_output_levels, [-1, N_level, 1])
                    self.onestep_policy_output_levels = tf.reshape(self.curr_policy_output_levels,
                                                                   [-1, N_level, dim_action])
                    self.onestep_mu_levels = self.curr_mu_levels
                    self.onestep_c_levels = self.curr_c_levels
                    self.onestep_z_levels = self.curr_z_levels
                    self.onestep_sig_levels = self.curr_sig_levels

                self.sig_series.append(self.curr_sig_levels)
                self.mu_series.append(self.curr_mu_levels)
                self.c_series.append(self.curr_c_levels)
                self.z_series.append(self.curr_z_levels)

                curr_value_output_levels_tensor = tf.transpose(
                    tf.reshape(self.curr_value_output_levels, [N_level, -1, 1]), perm=[1, 0, 2])

                self.value_output_series.append(tf.reshape(curr_value_output_levels_tensor, [-1, N_level, 1],
                                                           name='value_output_series'))

                curr_policy_output_levels_tensor = tf.transpose(
                    tf.reshape(self.curr_policy_output_levels, [N_level, -1, dim_action]), perm=[1, 0, 2])

                self.policy_output_series.append(tf.reshape(curr_policy_output_levels_tensor, [-1, N_level, dim_action],
                                                            name='policy_output_series'))

                self.prev_c_levels = self.curr_c_levels
                self.prev_mu_levels = self.curr_mu_levels
                self.prev_sig_levels = self.curr_sig_levels
                self.prev_z_levels = self.curr_z_levels

        with tf.name_scope('Regularizors'):
            self.regularizers = tf.nn.l2_loss(self.Wyu)
            for lev in range(N_level):
                self.regularizers += tf.nn.l2_loss(self.Wcu_this[lev])
                self.regularizers += tf.nn.l2_loss(self.Wo[lev])
                self.regularizers += tf.nn.l2_loss(self.Wo_mua[lev])
                self.regularizers += tf.nn.l2_loss(self.Wo_siga[lev])
                if lev > 0:
                    self.regularizers += tf.nn.l2_loss(self.Wcu_lower[lev - 1])
                if lev < N_level - 1:
                    self.regularizers += tf.nn.l2_loss(self.Wcu_higher[lev])
                if optimizer_name == 'RMSProp':
                    self.optimizer_w = tf.train.RMSPropOptimizer(learning_rate * beta, decay=0.99)
                elif optimizer_name == 'Adam':
                    self.optimizer_w = tf.train.AdamOptimizer(learning_rate * beta)
                self.train_step_w = self.optimizer_w.minimize(self.regularizers)

        with tf.name_scope('Loss') as scope:

            self.policy_output_series_tensor = tf.reshape(self.policy_output_series,
                                                          [truncated_backprop_length, -1, N_level, dim_action],
                                                          name='policy_output_series_tensor')
            self.policy_target_series_tensor = tf.reshape(self.policy_target_series,
                                                          [truncated_backprop_length, -1, N_level, dim_action],
                                                          name='policy_target_series_tensor')
            self.value_output_series_tensor = tf.reshape(self.value_output_series,
                                                         [truncated_backprop_length, -1, N_level, 1],
                                                         name='value_output_series_tensor')
            self.value_target_series_tensor = tf.reshape(self.value_target_series,
                                                         [truncated_backprop_length, -1, N_level, 1],
                                                         name='value_target_series_tensor')

            self.policy_output_tensor = tf.einsum('bt,tbij->btij', self.v_step_ph,
                                                  self.policy_output_series_tensor,
                                                  name='policy_output_tensor')
            self.policy_target_tensor = tf.einsum('bt,tbij->btij', self.v_step_ph,
                                                  self.policy_target_series_tensor,
                                                  name='policy_target_tensor')

            self.value_output_tensor = tf.einsum('bt,tbij->btij', self.v_step_ph,
                                                 self.value_output_series_tensor,
                                                 name='value_output_tensor')
            self.value_target_tensor = tf.einsum('bt,tbij->btij', self.v_step_ph,
                                                 self.value_target_series_tensor,
                                                 name='value_target_tensor')

            # -----------policy part-----------
            with tf.name_scope('critic_loss') as scope:
                self.losses_value = beta_v * tf.losses.mean_squared_error(self.value_target_tensor,
                                                                          self.value_output_tensor)
                if optimizer_name == 'RMSProp':
                    self.optimizer_v = tf.train.RMSPropOptimizer(learning_rate * beta_v, decay=0.99)
                elif optimizer_name == 'Adam':
                    self.optimizer_v = tf.train.AdamOptimizer(learning_rate * beta_v)

                self.train_step_v = self.optimizer_v.minimize(self.losses_value)

            with tf.name_scope('actor_loss') as scope:
                self.losses_policy = beta_a * tf.losses.mean_squared_error(self.policy_target_tensor,
                                                                           self.policy_output_tensor)
                if optimizer_name == 'RMSProp':
                    self.optimizer_a = tf.train.RMSPropOptimizer(learning_rate * beta_a, decay=0.99)
                elif optimizer_name == 'Adam':
                    self.optimizer_a = tf.train.AdamOptimizer(learning_rate * beta_a)
                self.train_step_a = self.optimizer_a.minimize(self.losses_policy)

            with tf.name_scope('total_loss') as scope:
                self.total_loss = self.losses_value + self.losses_policy + self.regularizers

                self.train_step = tf.group(self.train_step_w, self.train_step_v, self.train_step_a, name='train_op')
                self.Wcu_this_low_ph = tf.placeholder(tf.float32)

        self.Wcs_this_low_ph = tf.placeholder(tf.float32)
        self.bu_low_ph = tf.placeholder(tf.float32)
        self.bs_low_ph = tf.placeholder(tf.float32)
        self.Wo_mua_low_ph = tf.placeholder(tf.float32)
        self.bo_mua_low_ph = tf.placeholder(tf.float32)
        self.Wo_siga_low_ph = tf.placeholder(tf.float32)
        self.bo_siga_low_ph = tf.placeholder(tf.float32)
        self.Wo_low_ph = tf.placeholder(tf.float32)
        self.bo_low_ph = tf.placeholder(tf.float32)
        self.Wyu_ph = tf.placeholder(tf.float32)
        self.assign_low_weights = [tf.assign(self.Wcu_this[0], self.Wcu_this_low_ph),
                                   tf.assign(self.Wcs_this[0], self.Wcs_this_low_ph),
                                   tf.assign(self.bu[0], self.bu_low_ph),
                                   tf.assign(self.bs[0], self.bs_low_ph),
                                   tf.assign(self.Wo_mua[0], self.Wo_mua_low_ph),
                                   tf.assign(self.bo_mua[0], self.bo_mua_low_ph),
                                   tf.assign(self.Wo_siga[0], self.Wo_siga_low_ph),
                                   tf.assign(self.bo_siga[0], self.bo_siga_low_ph),
                                   tf.assign(self.Wo[0], self.Wo_low_ph),
                                   tf.assign(self.bo[0], self.bo_low_ph),
                                   tf.assign(self.Wyu, self.Wyu_ph)]
        if recording:
            tf.summary.histogram('W_o_1', self.Wo[0])
            tf.summary.histogram('b_o_1', self.bo[0])
            tf.summary.histogram('W_yu', self.Wyu)
            tf.summary.histogram('W_cu_1_to_1', self.Wcu_this[0])
            tf.summary.histogram('W_cu_2_to_1', self.Wcu_higher[0])
            tf.summary.histogram('W_cu_1_to_2', self.Wcu_lower[0])
            tf.summary.histogram('W_cu_2_to_2', self.Wcu_this[1])
            if N_level >= 3:
                tf.summary.histogram('W_cu_2_to_3', self.Wcu_lower[1])
                tf.summary.histogram('W_cu_3_to_2', self.Wcu_lower[1])
                tf.summary.histogram('W_cu_3_to_3', self.Wcu_this[2])
            tf.summary.scalar('episode_mean_reward', self.reward_ph)
            tf.summary.scalar('loss_critic', self.losses_value)
            tf.summary.scalar('loss_actor', self.losses_policy)
            tf.summary.scalar('total_loss', self.total_loss)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(LOG_DIR + filename + now.strftime("%Y%m%d-%H%M%S") + "/",
                                                      self.session.graph)

    def record_low_weights(self):
        self.Wcu_this_low_old = self.Wcu_this[0].eval()

        self.Wcs_this_low_old = self.Wcs_this[0].eval()
        self.bu_low_old = self.bu[0].eval()
        self.bs_low_old = self.bs[0].eval()
        self.Wo_mua_low_old = self.Wo_mua[0].eval()
        self.bo_mua_low_old = self.bo_mua[0].eval()
        self.Wo_siga_low_old = self.Wo_siga[0].eval()
        self.bo_siga_low_old = self.bo_siga[0].eval()
        self.Wo_low_old = self.Wo[0].eval()
        self.bo_low_old = self.bo[0].eval()
        self.Wyu_old = self.Wyu.eval()

    def recover_low_weights(self):
        self.session.run(self.assign_low_weights, feed_dict={self.Wcu_this_low_ph: self.Wcu_this_low_old,
                                                             self.Wcs_this_low_ph: self.Wcs_this_low_old,
                                                             self.bu_low_ph: self.bu_low_old,
                                                             self.bs_low_ph: self.bs_low_old,
                                                             self.Wo_mua_low_ph: self.Wo_mua_low_old,
                                                             self.bo_mua_low_ph: self.bo_mua_low_old,
                                                             self.Wo_siga_low_ph: self.Wo_siga_low_old,
                                                             self.bo_siga_low_ph: self.bo_siga_low_old,
                                                             self.Wo_low_ph: self.Wo_low_old,
                                                             self.bo_low_ph: self.bo_low_old,
                                                             self.Wyu_ph: self.Wyu_old})

    def initial_variables(self):
        # Initialize
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

    def train_batch(self, feed_dict_train_batch):
        self.session.run(self.train_step, feed_dict=feed_dict_train_batch)

    def summary_write(self, feed_dict_train_batch, global_step):
        summary = self.session.run(self.merged, feed_dict=feed_dict_train_batch)
        self.train_writer.add_summary(summary, global_step)


class LSTM():
    def __init__(self):
        self.session = tf.InteractiveSession()
        self.define_weights()
        self.build_graph()
        self.initial_variables()

    def __call__(self, prev_c, prev_h, input):

        x = np.zeros([1, truncated_backprop_length, dim_input])
        x[0, 0, :] = np.reshape(input, [dim_input])

        feed_dict = {self.batchX_ph: x,
                     self.init_c: np.reshape(prev_c, [1, num_state]),
                     self.init_h: np.reshape(prev_h, [1, num_state])}

        v, pi, c, h = self.session.run([self.next_v,
                                        self.next_pi,
                                        self.next_c,
                                        self.next_h], feed_dict=feed_dict)
        return v, pi, c, h

    def define_weights(self):
        with tf.name_scope('output_weights'):
            self.W_v = tf.Variable(tf.zeros([num_state, 1], dtype=tf.float32), name='W_v')
            self.b_v = tf.Variable(tf.zeros([1, 1], dtype=tf.float32), name='b_v')
            self.W_a = tf.Variable(tf.zeros([num_state, dim_action], dtype=tf.float32), name='W_v')
            self.b_a = tf.Variable(tf.zeros([1, dim_action], dtype=tf.float32), name='b_v')

    def build_graph(self):
        with tf.name_scope('input'):
            self.v_step_ph = tf.placeholder(tf.float32, [None, truncated_backprop_length], name='V_step')  # For padding
            self.batchX_ph = tf.placeholder(tf.float32, [None, truncated_backprop_length, dim_input], name='input')
            self.batchP_ph = tf.placeholder(tf.float32, [None, truncated_backprop_length, dim_action],
                                            name='target_policy')  # target policy(mean action)
            self.batchV_ph = tf.placeholder(tf.float32, [None, truncated_backprop_length, 1],
                                            name='target_value')  # target value V
            self.reward_ph = tf.placeholder(tf.float32, None, name='reward')
            self.init_h = tf.placeholder(tf.float32, [None, num_state], name='init_h')
            self.init_c = tf.placeholder(tf.float32, [None, num_state], name='init_c')

            self.init_state = tf.nn.rnn_cell.LSTMStateTuple(self.init_c, self.init_h)

            self.policy_target_series = tf.transpose(self.batchP_ph, perm=[1, 0, 2], name='policy_target_series')
            self.value_target_series = tf.transpose(self.batchV_ph, perm=[1, 0, 2], name='value_target_series')

        with tf.name_scope('LSTM'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_state, state_is_tuple=True)
            self.policy_output_series = []
            self.value_output_series = []

            prev_state = self.init_state
            for stp in range(truncated_backprop_length):
                output, next_state = tf.nn.dynamic_rnn(lstm_cell, self.batchX_ph[:, stp, None, :],
                                                       initial_state=prev_state, time_major=False)
                output = tf.reshape(output, [-1, num_state], name='output')
                policy_output = tf.tanh(tf.matmul(output, self.W_a) + self.b_a, name='policy_output')
                value_output = tf.add(tf.matmul(output, self.W_v), self.b_v, name='value_output')

                self.policy_output_series.append(policy_output)
                self.value_output_series.append(value_output)

                if stp == 0:
                    self.next_v = value_output
                    self.next_pi = policy_output
                    self.next_c, self.next_h = next_state

                prev_state = next_state

            self.policy_output_series_tensor = tf.reshape(self.policy_output_series,
                                                          [truncated_backprop_length, -1, dim_action],
                                                          name='policy_output_series_tensor')
            self.policy_target_series_tensor = tf.reshape(self.policy_target_series,
                                                          [truncated_backprop_length, -1, dim_action],
                                                          name='policy_target_series_tensor')
            self.value_output_series_tensor = tf.reshape(self.value_output_series,
                                                         [truncated_backprop_length, -1, 1],
                                                         name='value_output_series_tensor')
            self.value_target_series_tensor = tf.reshape(self.value_target_series,
                                                         [truncated_backprop_length, -1, 1],
                                                         name='value_target_series_tensor')

            self.policy_output_tensor = tf.einsum('bt,tbi->bti', self.v_step_ph,
                                                  self.policy_output_series_tensor,
                                                  name='policy_output_tensor')
            self.policy_target_tensor = tf.einsum('bt,tbi->bti', self.v_step_ph,
                                                  self.policy_target_series_tensor,
                                                  name='policy_target_tensor')

            self.value_output_tensor = tf.einsum('bt,tbi->bti', self.v_step_ph,
                                                 self.value_output_series_tensor,
                                                 name='value_output_tensor')
            self.value_target_tensor = tf.einsum('bt,tbi->bti', self.v_step_ph,
                                                 self.value_target_series_tensor,
                                                 name='value_target_tensor')

        with tf.name_scope('loss'):
            self.loss_a = tf.losses.mean_squared_error(self.policy_target_tensor, self.policy_output_tensor)
            self.loss_v = tf.losses.mean_squared_error(self.value_target_tensor, self.value_output_tensor)

            if optimizer_name == 'Adam':
                self.train_step_a = tf.train.AdamOptimizer(beta_a * learning_rate).minimize(self.loss_a)
                self.train_step_v = tf.train.AdamOptimizer(beta_v * learning_rate).minimize(self.loss_v)
            elif optimizer_name == 'RMSProp':
                self.train_step_a = tf.train.RMSPropOptimizer(beta_a * learning_rate, decay=0.99).minimize(self.loss_a)
                self.train_step_v = tf.train.RMSPropOptimizer(beta_v * learning_rate, decay=0.99).minimize(self.loss_v)
            self.train_step = tf.group([self.train_step_a, self.train_step_v])

        if recording:
            tf.summary.scalar('episode_mean_reward', self.reward_ph)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(LOG_DIR + filename + now.strftime("%Y%m%d-%H%M%S") + "/",
                                                      self.session.graph)

    def initial_variables(self):
        # Initialize
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

    def train_batch(self, train_dict):
        self.session.run(self.train_step, feed_dict=train_dict)

    def summary_write(self, train_dict, global_step):
        summary = self.session.run(self.merged, feed_dict=train_dict)
        self.train_writer.add_summary(summary, global_step)


class EpisodeRecorder():
    def __init__(self):
        """
        :param E_all: A matrix with shape [episode_length, total_num_states]
        """
        self.cs_series_all = []
        self.mus_series_all = []
        self.sigs_series_all = []
        self.inputs_series_all = []
        self.position_series_all = []
        self.value_outputs_series_all = []
        self.policy_outputs_series_all = []
        self.action_series_all = []
        self.rewards_all = []
        self.done_all = []
        self.td_errors = []
        self.data = {}
        self.performance = 0
        self.num_state_sum = np.zeros(N_level + 1, dtype=int)
        self.num_state_sum[0] = 0

        for lev in range(N_level):
            self.num_state_sum[lev + 1] = np.sum(num_state[0:lev + 1])

    def record_step(self,
                    previous_position_,
                    previous_observations_,
                    previous_action_,
                    previous_value_output_levels_,
                    previous_policy_output_levels_,
                    previous_c_levels_,
                    previous_mu_levels_,
                    previous_sig_levels_,
                    reward_,
                    done_):
        previous_input_ = get_input(previous_observations_, dim_input=dim_input)
        self.action_series_all.append(previous_action_)
        self.cs_series_all.append(previous_c_levels_)
        self.mus_series_all.append(previous_mu_levels_)
        self.sigs_series_all.append(previous_sig_levels_)
        self.value_outputs_series_all.append(previous_value_output_levels_)
        self.policy_outputs_series_all.append(previous_policy_output_levels_)
        self.inputs_series_all.append(previous_input_)
        self.position_series_all.append(previous_position_)
        self.rewards_all.append(reward_)
        self.done_all.append(done_)

    def data_cleanup(self, **kwargs):
        input_series_all = np.reshape(self.inputs_series_all, [-1, dim_input])
        position_series_all = np.reshape(self.position_series_all, [-1, dim_position])
        value_output_series_all = np.reshape(self.value_outputs_series_all, [-1, N_level, 1])
        policy_output_series_all = np.reshape(self.policy_outputs_series_all, [-1, N_level, dim_action])
        action_series_all = np.reshape(self.action_series_all, [-1, dim_action])
        self.data = {'input': input_series_all, 'reward': self.rewards_all, 'num_state': num_state,
                     'position': position_series_all, 'value_output': value_output_series_all,
                     'tau': np.float64(tau), 'gamma': gamma,
                     'action': action_series_all, 'policy_output': policy_output_series_all}

        for lev in range(N_level):
            layer_str = "%.1d" % (lev + 1)
            step = len(self.cs_series_all)
            self.data['c_' + layer_str] = np.reshape([self.cs_series_all[j][lev] for j in range(step)],
                                                     [step, num_state[lev]])
            self.data['mu_' + layer_str] = np.reshape([self.mus_series_all[j][lev] for j in range(step)],
                                                      [step, num_state[lev]])
            self.data['sig_' + layer_str] = np.reshape([self.sigs_series_all[j][lev] for j in range(step)],
                                                       [step, num_state[lev]])
        for key, value in kwargs.items():
            self.data[key] = value
        return self.data

    def save_data(self, task, episode):
        # should be done after data_cleanup
        global filename
        global episode_beg
        episode_str = "%.5d" % (episode + episode_beg)
        self.data['red_pos'] = task.red_position
        self.data['green_pos'] = task.green_position
        self.data['blue_pos'] = task.blue_position
        scipy.io.savemat(savepath + filename + episode_str + '.mat', self.data)

    def plot_episode(self, task, episode):
        position_series_all = np.reshape(self.position_series_all, [-1, dim_position])
        plt.plot(position_series_all[:, 0], position_series_all[:, 1])
        plt.plot(position_series_all[0, 0], position_series_all[0, 1], 'kx')
        plt.plot([task.red_position[0]], [task.red_position[1]], 'ro')
        plt.plot([task.green_position[0]], [task.green_position[1]], 'go')
        plt.plot([task.blue_position[0]], [task.blue_position[1]], 'bo')
        plt.xlim(0, 15)
        plt.ylim(0, 15)
        episode_str = "%.5d" % episode
        plt.title("episode " + episode_str)
        plt.show()

    def print_result(self, episode):
        print("Episode: %5d, rewards_mean = %7.5f, steps = %5d" % (
            episode, np.mean(self.rewards_all), len(self.cs_series_all)))

    def get_performance(self):
        self.performance = np.mean(self.rewards_all)
        return self.performance

    def record_end(self, last_position_, last_observations_):
        last_input = get_input(last_observations_, dim_input)
        self.inputs_series_all.append(last_input)
        self.position_series_all.append(last_position_)


class EpisodeRecorder_LSTM():
    def __init__(self):
        self.cs_series_all = []
        self.hs_series_all = []
        self.inputs_series_all = []
        self.position_series_all = []
        self.value_outputs_series_all = []
        self.policy_outputs_series_all = []
        self.action_series_all = []
        self.rewards_all = []
        self.done_all = []
        self.td_errors = []
        self.data = {}
        self.performance = 0

    def record_step(self,
                    previous_position_,
                    previous_observations_,
                    previous_action_,
                    previous_value_output_,
                    previous_policy_output_,
                    previous_c_,
                    previous_h_,
                    reward_,
                    done_):
        previous_input_ = get_input(previous_observations_, dim_input=dim_input)
        self.action_series_all.append(previous_action_)
        self.cs_series_all.append(previous_c_)
        self.hs_series_all.append(previous_h_)
        self.value_outputs_series_all.append(previous_value_output_)
        self.policy_outputs_series_all.append(previous_policy_output_)
        self.inputs_series_all.append(previous_input_)
        self.position_series_all.append(previous_position_)
        self.rewards_all.append(reward_)
        self.done_all.append(done_)

    def data_cleanup(self, **kwargs):
        input_series_all = np.reshape(self.inputs_series_all, [-1, dim_input])
        position_series_all = np.reshape(self.position_series_all, [-1, dim_position])
        value_output_series_all = np.reshape(self.value_outputs_series_all, [-1, 1])
        policy_output_series_all = np.reshape(self.policy_outputs_series_all, [-1, dim_action])
        action_series_all = np.reshape(self.action_series_all, [-1, dim_action])
        self.data = {'input': input_series_all, 'reward': self.rewards_all, 'num_state': num_state,
                     'position': position_series_all, 'value_output': value_output_series_all, 'gamma': gamma,
                     'action': action_series_all, 'policy_output': policy_output_series_all}

        self.data['c'] = np.reshape(self.cs_series_all, [-1, num_state])
        self.data['h'] = np.reshape(self.hs_series_all, [-1, num_state])
        for key, value in kwargs.items():
            self.data[key] = value

        return self.data

    def save_data(self, task, episode):
        # should be done after data_cleanup
        global filename
        global episode_beg
        episode_str = "%.5d" % (episode + episode_beg)
        self.data['red_pos'] = task.red_position
        self.data['green_pos'] = task.green_position
        self.data['blue_pos'] = task.blue_position
        scipy.io.savemat(savepath + filename + episode_str + '.mat', self.data)

    def plot_episode(self, task, episode):
        position_series_all = np.reshape(self.position_series_all, [-1, dim_position])
        plt.plot(position_series_all[:, 0], position_series_all[:, 1])
        plt.plot(position_series_all[0, 0], position_series_all[0, 1], 'kx')
        plt.plot([task.red_position[0]], [task.red_position[1]], 'ro')
        plt.plot([task.green_position[0]], [task.green_position[1]], 'go')
        plt.plot([task.blue_position[0]], [task.blue_position[1]], 'bo')
        plt.xlim(0, 15)
        plt.ylim(0, 15)
        episode_str = "%.5d" % episode
        plt.title("episode " + episode_str)
        plt.show()

    def print_result(self, episode):
        print("Episode: %5d, rewards_mean = %7.5f, steps = %5d" % (
            episode, np.mean(self.rewards_all), len(self.cs_series_all)))

    def get_performance(self):
        self.performance = np.mean(self.rewards_all)
        return self.performance

    def record_end(self, last_position_, last_observations_):
        last_input = get_input(last_observations_, dim_input)
        self.inputs_series_all.append(last_input)
        self.position_series_all.append(last_position_)


class ReplayBuffer():
    """
    Record the state transition (s->s_next,a,r,done) and necessary states for MTRNN
    """

    def __init__(self, state_dim=dim_input, action_dim=dim_action, state_dtype=np.float32, action_dtype=np.float32,
                 size=buffer_size):
        self.s_t = np.zeros(shape=(size, state_dim), dtype=state_dtype)
        self.s_tp1 = np.zeros(shape=(size, state_dim), dtype=state_dtype)
        self.a_t = np.zeros(shape=(size, action_dim), dtype=action_dtype)
        self.r_t = np.zeros(shape=(size,), dtype=state_dtype)
        self.bp_t = np.zeros(shape=(size, action_dim), dtype=state_dtype)  # behavior policy
        self.done_t = np.zeros(shape=(size,), dtype=state_dtype)  # If 1., an episode has ended at t plus 1.
        self.step_t = np.zeros(shape=(size,), dtype=np.int32)
        # Steps taken from episode starts. For Padding. step_t=0 means it is the first step(prev_c, prev_mu = zeros)
        self.done_step_t = np.zeros(shape=(size,), dtype=np.int32)  # how many steps to finish this episode
        self.ic_t = np.zeros(shape=(size, total_num_state), dtype=np.float32)  # initial c
        self.imu_t = np.zeros(shape=(size, total_num_state), dtype=np.float32)  # initial mu
        self.step_tp1 = np.zeros(shape=(size,), dtype=np.int32)
        self.ic_tp1 = np.zeros(shape=(size, total_num_state), dtype=np.float32)
        self.imu_tp1 = np.zeros(shape=(size, total_num_state), dtype=np.float32)
        self.mu_t = np.zeros(shape=(size, total_num_state), dtype=np.float32)
        self.z_t = np.zeros(shape=(size, total_num_state), dtype=np.float32)

        self.size = size
        self.filled = -1  # up to this index, the buffer is filled
        self.head = -1  # index to which an experience to be stored next

    def __len__(self):
        return self.filled

    def append(self, s, sp, a, r, bp, done, step, ic, icp, imu, imup, mu, z):

        self.filled = max(self.filled, self.head)
        self.head += 1

        if self.head < self.size:
            self.s_t[self.head] = s
            self.s_tp1[self.head] = sp
            self.a_t[self.head] = a
            self.r_t[self.head] = r
            self.bp_t[self.head] = bp
            self.done_t[self.head] = done
            self.ic_t[self.head] = ic
            self.ic_tp1[self.head] = icp
            self.imu_t[self.head] = imu
            self.imu_tp1[self.head] = imup
            self.step_t[self.head] = step
            self.step_tp1[self.head] = step + 1
            self.mu_t[self.head] = mu
            self.z_t[self.head] = z
        else:
            self.head = 0
            self.s_t[self.head] = s
            self.s_tp1[self.head] = sp
            self.a_t[self.head] = a
            self.r_t[self.head] = r
            self.bp_t[self.head] = bp
            self.done_t[self.head] = done
            self.ic_t[self.head] = ic
            self.ic_tp1[self.head] = icp
            self.imu_t[self.head] = imu
            self.imu_tp1[self.head] = imup
            self.step_t[self.head] = step
            self.step_tp1[self.head] = step + 1
            self.mu_t[self.head] = mu
            self.z_t[self.head] = z

        if self.head >= (max_steps + truncated_backprop_length):
            index = self.head - (max_steps + truncated_backprop_length)
            rest_step = 0
            if self.done_t[index] == -1:  # means this is a fake padding step
                while (not self.done_t[index - rest_step] == 1) and (
                not self.step_tp1[index - rest_step] == max_steps - 1):
                    rest_step += 1
                done_step = self.step_tp1[index - rest_step]
            else:
                while (not self.done_t[index + rest_step]) and self.step_t[
                            index + rest_step] < max_steps - 2 and index + rest_step < self.head - 1:
                    rest_step += 1
                done_step = self.step_tp1[index] + rest_step  # compute totollay how many steps to finish this episode

            self.done_step_t[index] = done_step

    def padding_tail(self, s, sp, a, r, bp, done, step, ic, icp, imu, imup, mu, z):

        self.filled = max(self.filled, self.head)
        self.head += 1

        if self.head < self.size:
            self.s_t[self.head] = s
            self.s_tp1[self.head] = sp
            self.a_t[self.head] = a
            self.r_t[self.head] = r
            self.bp_t[self.head] = bp
            self.done_t[self.head] = -1
            self.ic_t[self.head] = ic
            self.ic_tp1[self.head] = icp
            self.imu_t[self.head] = imu
            self.imu_tp1[self.head] = imup
            self.step_t[self.head] = step
            self.step_tp1[self.head] = step + 1
            self.mu_t[self.head] = mu
            self.z_t[self.head] = z
        else:
            self.head = 0
            self.s_t[self.head] = s
            self.s_tp1[self.head] = sp
            self.a_t[self.head] = a
            self.r_t[self.head] = r
            self.bp_t[self.head] = bp
            self.done_t[self.head] = -1
            self.ic_t[self.head] = ic
            self.ic_tp1[self.head] = icp
            self.imu_t[self.head] = imu
            self.imu_tp1[self.head] = imup
            self.step_t[self.head] = step
            self.step_tp1[self.head] = step + 1
            self.mu_t[self.head] = mu
            self.z_t[self.head] = z

        if self.head >= (max_steps + truncated_backprop_length):
            index = self.head - (max_steps + truncated_backprop_length)
            rest_step = 0
            if self.done_t[index] == -1:  # means this is a fake padding step
                while (not self.done_t[index - rest_step] == 1) and (
                        not self.step_tp1[index - rest_step] == max_steps - 1):
                    rest_step += 1
                done_step = self.step_tp1[index - rest_step]
            else:
                while (not self.done_t[index + rest_step]) and self.step_t[
                            index + rest_step] < max_steps - 2 and index + rest_step < self.head - 1:
                    rest_step += 1
                done_step = self.step_tp1[index] + rest_step  # compute totollay how many steps to finish this episode

            self.done_step_t[index] = done_step

    def sample(self, truncated_steps=truncated_backprop_length):

        if self.head >= max_steps + truncated_backprop_length and self.head + truncated_backprop_length <= self.filled - (
            max_steps + truncated_backprop_length):
            aval_indice = np.hstack(
                [np.arange(truncated_backprop_length, self.head - (max_steps + truncated_backprop_length), 1),
                 np.arange(self.head + truncated_backprop_length, self.filled - (max_steps + truncated_backprop_length),
                           1)])
        elif self.head < max_steps + truncated_backprop_length:
            aval_indice = np.arange(self.head + truncated_backprop_length,
                                    self.filled - (max_steps + truncated_backprop_length), 1)
        elif self.head + truncated_backprop_length > self.filled - (max_steps + truncated_backprop_length):
            aval_indice = np.arange(truncated_backprop_length, self.head - (max_steps + truncated_backprop_length), 1)
        else:
            aval_indice = np.arange(0, self.filled - (max_steps + truncated_backprop_length), 1)

        index = int(np.random.choice(aval_indice))
        step_ = self.step_t[index]

        if step_ + 1 < truncated_backprop_length:
            ret_tuple = (self.s_t[index - step_:index + 1],
                         np.vstack((self.s_t[index - step_:index + 1], self.s_tp1[index:index + 1])),
                         self.a_t[index - step_:index + 1],
                         self.r_t[index - step_:index + 1],
                         self.bp_t[index - step_:index + 1],
                         self.done_t[index - step_:index + 1],
                         self.ic_t[index],
                         self.ic_tp1[index],
                         self.imu_t[index],
                         self.imu_tp1[index],
                         self.step_t[index],
                         self.step_tp1[index],
                         self.done_step_t[index],
                         self.mu_t[index - step_:index + 1],
                         self.z_t[index - step_:index + 1])
        else:
            ret_tuple = (self.s_t[index - truncated_steps + 1:index + 1],
                         self.s_tp1[index - truncated_steps + 1:index + 1],
                         self.a_t[index - truncated_steps + 1:index + 1],
                         self.r_t[index - truncated_steps + 1:index + 1],
                         self.bp_t[index - truncated_steps + 1:index + 1],
                         self.done_t[index - truncated_steps + 1:index + 1],
                         self.ic_t[index],
                         self.ic_tp1[index],
                         self.imu_t[index],
                         self.imu_tp1[index],
                         self.step_t[index],
                         self.step_tp1[index],
                         self.done_step_t[index],
                         self.mu_t[index - truncated_steps + 1:index + 1],
                         self.z_t[index - truncated_steps + 1:index + 1])
        return ret_tuple


class ReplayBuffer_LSTM():
    """
    Record the state transition (s->s_next,a,r,done) and necessary states for MTRNN
    """

    def __init__(self, state_dim=dim_input, action_dim=dim_action, state_dtype=np.float32, action_dtype=np.float32,
                 size=buffer_size):
        self.s_t = np.zeros(shape=(size, state_dim), dtype=state_dtype)
        self.s_tp1 = np.zeros(shape=(size, state_dim), dtype=state_dtype)
        self.a_t = np.zeros(shape=(size, action_dim), dtype=action_dtype)
        self.r_t = np.zeros(shape=(size,), dtype=state_dtype)
        self.bp_t = np.zeros(shape=(size, action_dim), dtype=state_dtype)  # behavior policy
        self.done_t = np.zeros(shape=(size,), dtype=state_dtype)  # If 1., an episode has ended at t plus 1.
        self.step_t = np.zeros(shape=(size,), dtype=np.int32)
        # Steps taken from episode starts. For Padding. step_t=0 means it is the first step(prev_c, prev_h = zeros)
        self.done_step_t = np.zeros(shape=(size,), dtype=np.int32)  # how many steps to finish this episode

        self.ic_t = np.zeros(shape=(size, num_state), dtype=np.float32)  # initial c
        self.ih_t = np.zeros(shape=(size, num_state), dtype=np.float32)  # initial h
        self.step_tp1 = np.zeros(shape=(size,), dtype=np.int32)
        self.ic_tp1 = np.zeros(shape=(size, num_state), dtype=np.float32)
        self.ih_tp1 = np.zeros(shape=(size, num_state), dtype=np.float32)

        self.size = size
        self.filled = -1  # up to this index, the buffer is filled
        self.head = -1  # index to which an experience to be stored next

    def __len__(self):
        return self.filled

    def append(self, s, sp, a, r, bp, done, step, ic, icp, ih, ihp):

        self.filled = max(self.filled, self.head)
        self.head += 1

        if self.head < self.size:
            self.s_t[self.head] = s
            self.s_tp1[self.head] = sp
            self.a_t[self.head] = a
            self.r_t[self.head] = r
            self.bp_t[self.head] = bp
            self.done_t[self.head] = done
            self.ic_t[self.head] = ic
            self.ic_tp1[self.head] = icp
            self.ih_t[self.head] = ih
            self.ih_tp1[self.head] = ihp
            self.step_t[self.head] = step
            self.step_tp1[self.head] = step + 1
        else:
            self.head = 0
            self.s_t[self.head] = s
            self.s_tp1[self.head] = sp
            self.a_t[self.head] = a
            self.r_t[self.head] = r
            self.bp_t[self.head] = bp
            self.done_t[self.head] = done
            self.ic_t[self.head] = ic
            self.ic_tp1[self.head] = icp
            self.ih_t[self.head] = ih
            self.ih_tp1[self.head] = ihp
            self.step_t[self.head] = step
            self.step_tp1[self.head] = step + 1

        if self.head >= (max_steps + truncated_backprop_length):
            index = self.head - (max_steps + truncated_backprop_length)
            rest_step = 0
            if self.done_t[index] == -1:  # means this is a fake padding step
                while (not self.done_t[index - rest_step] == 1) and (
                not self.step_tp1[index - rest_step] == max_steps - 1):
                    rest_step += 1
                done_step = self.step_tp1[index - rest_step]
            else:
                while (not self.done_t[index + rest_step]) and self.step_t[
                            index + rest_step] < max_steps - 2 and index + rest_step < self.head - 1:
                    rest_step += 1
                done_step = self.step_tp1[index] + rest_step  # compute totollay how many steps to finish this episode

            self.done_step_t[index] = done_step

    def padding_tail(self, s, sp, a, r, bp, done, step, ic, icp, ih, ihp):

        self.filled = max(self.filled, self.head)
        self.head += 1

        if self.head < self.size:
            self.s_t[self.head] = s
            self.s_tp1[self.head] = sp
            self.a_t[self.head] = a
            self.r_t[self.head] = r
            self.bp_t[self.head] = bp
            self.done_t[self.head] = -1
            self.ic_t[self.head] = ic
            self.ic_tp1[self.head] = icp
            self.ih_t[self.head] = ih
            self.ih_tp1[self.head] = ihp
            self.step_t[self.head] = step
            self.step_tp1[self.head] = step + 1
        else:
            self.head = 0
            self.s_t[self.head] = s
            self.s_tp1[self.head] = sp
            self.a_t[self.head] = a
            self.r_t[self.head] = r
            self.bp_t[self.head] = bp
            self.done_t[self.head] = -1
            self.ic_t[self.head] = ic
            self.ic_tp1[self.head] = icp
            self.ih_t[self.head] = ih
            self.ih_tp1[self.head] = ihp
            self.step_t[self.head] = step
            self.step_tp1[self.head] = step + 1

        if self.head >= (max_steps + truncated_backprop_length):
            index = self.head - (max_steps + truncated_backprop_length)
            rest_step = 0
            if self.done_t[index] == -1:  # means this is a fake padding step
                while (not self.done_t[index - rest_step] == 1) and (
                        not self.step_tp1[index - rest_step] == max_steps - 1):
                    rest_step += 1
                done_step = self.step_tp1[index - rest_step]
            else:
                while (not self.done_t[index + rest_step]) and self.step_t[
                            index + rest_step] < max_steps - 2 and index + rest_step < self.head - 1:
                    rest_step += 1
                done_step = self.step_tp1[index] + rest_step  # compute totollay how many steps to finish this episode

            self.done_step_t[index] = done_step

    def sample(self, truncated_steps=truncated_backprop_length):

        if self.head >= max_steps + truncated_backprop_length and self.head + truncated_backprop_length <= self.filled - (
            max_steps + truncated_backprop_length):
            aval_indice = np.hstack(
                [np.arange(truncated_backprop_length, self.head - (max_steps + truncated_backprop_length), 1),
                 np.arange(self.head + truncated_backprop_length, self.filled - (max_steps + truncated_backprop_length),
                           1)])
        elif self.head < max_steps + truncated_backprop_length:
            aval_indice = np.arange(self.head + truncated_backprop_length,
                                    self.filled - (max_steps + truncated_backprop_length), 1)
        elif self.head + truncated_backprop_length > self.filled - (max_steps + truncated_backprop_length):
            aval_indice = np.arange(truncated_backprop_length, self.head - (max_steps + truncated_backprop_length), 1)
        else:
            aval_indice = np.arange(0, self.filled - (max_steps + truncated_backprop_length), 1)

        index = int(np.random.choice(aval_indice))
        step_ = self.step_t[index]

        if step_ + 1 < truncated_backprop_length:
            ret_tuple = (self.s_t[index - step_:index + 1],
                         np.vstack((self.s_t[index - step_:index + 1], self.s_tp1[index:index + 1])),
                         self.a_t[index - step_:index + 1],
                         self.r_t[index - step_:index + 1],
                         self.bp_t[index - step_:index + 1],
                         self.done_t[index - step_:index + 1],
                         self.ic_t[index],
                         self.ic_tp1[index],
                         self.ih_t[index],
                         self.ih_tp1[index],
                         self.step_t[index],
                         self.step_tp1[index],
                         self.done_step_t[index])
        else:
            ret_tuple = (self.s_t[index - truncated_steps + 1:index + 1],
                         self.s_tp1[index - truncated_steps + 1:index + 1],
                         self.a_t[index - truncated_steps + 1:index + 1],
                         self.r_t[index - truncated_steps + 1:index + 1],
                         self.bp_t[index - truncated_steps + 1:index + 1],
                         self.done_t[index - truncated_steps + 1:index + 1],
                         self.ic_t[index],
                         self.ic_tp1[index],
                         self.ih_t[index],
                         self.ih_tp1[index],
                         self.step_t[index],
                         self.step_tp1[index],
                         self.done_step_t[index])
        return ret_tuple


class AGENT_MTSRNN():
    def __init__(self, nn: MTSRNN, recorder: EpisodeRecorder, memory: ReplayBuffer):
        self.nn = nn
        self.recorder = recorder
        self.memory = memory
        self.cumu_reward = 0.
        self.last_ep_mean_r = 0.

    def init_episode(self, episode, input, rand_noise):
        self.cumu_reward = 0
        self.recorder = EpisodeRecorder()
        self.previous_mu_levels = levelize(np.zeros([total_num_state]), num_state)
        self.previous_c_levels = levelize(np.zeros([total_num_state]), num_state)

        init_action = self.select(input, rand_noise)

        self.previous_c_levels = deepcopy(self.current_c_levels)
        self.previous_mu_levels = deepcopy(self.current_mu_levels)
        self.previous_sig_levels = deepcopy(self.current_sig_levels)
        self.previous_value_output_levels = deepcopy(self.current_value_output_levels)
        self.previous_policy_output_levels = deepcopy(self.current_policy_output_levels)

        return init_action

    def get_input_dict(self):
        pass

    def get_train_dict(self, samples, episode):
        batchX = np.zeros([batch_size, truncated_backprop_length, dim_input])
        batchX_p = np.zeros([batch_size, truncated_backprop_length, dim_input])
        init_c_feed = np.zeros([batch_size, total_num_state])
        init_c_feed_p = np.zeros([batch_size, total_num_state])
        init_mu_feed = np.zeros([batch_size, total_num_state])
        init_mu_feed_p = np.zeros([batch_size, total_num_state])
        v_step = np.zeros([batch_size, truncated_backprop_length])

        batchP = np.zeros([batch_size, truncated_backprop_length, N_level, dim_action])
        batchV = np.zeros([batch_size, truncated_backprop_length, N_level, 1])

        td_error = np.zeros([N_level, ])

        glb = globals()
        noise_decrease = glb["noise_decrease"]
        coef_eps = np.exp(-noise_decrease * episode)

        for b, sample in enumerate(samples):

            [prev_input_series, curr_input_series, prev_a_series, reward_series, prev_bp_series, done_series,
             prev_ic, curr_ic, prev_imu, curr_imu, prev_step, curr_step, done_step, prev_mu, prev_z] = sample

            # Generate input batch
            if curr_step > done_step:
                if prev_step >= truncated_backprop_length:
                    if curr_step - done_step > 0 and curr_step - done_step < truncated_backprop_length:
                        v_step[b] = np.hstack([np.ones([truncated_backprop_length - curr_step + done_step]),
                                               np.zeros([curr_step - done_step])])
                    else:
                        pass  # zeros
                    batchX[b] = np.reshape(prev_input_series, [truncated_backprop_length, dim_input])
                else:
                    v_step[b, 0:done_step] = 1
                    batchX[b, 0:(prev_step + 1)] = np.reshape(prev_input_series, [-1, dim_input])
                if curr_step >= truncated_backprop_length:
                    batchX_p[b] = np.reshape(curr_input_series, [truncated_backprop_length, dim_input])
                else:
                    batchX_p[b, 0:(curr_step + 1)] = np.reshape(curr_input_series, [-1, dim_input])

            else:
                if prev_step >= truncated_backprop_length:
                    v_step[b] = np.ones([truncated_backprop_length])
                    batchX[b] = np.reshape(prev_input_series, [truncated_backprop_length, dim_input])
                else:
                    v_step[b, 0:(prev_step + 1)] = 1
                    batchX[b, 0:(prev_step + 1)] = np.reshape(prev_input_series, [-1, dim_input])

                if curr_step >= truncated_backprop_length:
                    batchX_p[b] = np.reshape(curr_input_series, [truncated_backprop_length, dim_input])
                else:
                    batchX_p[b, 0:(curr_step + 1)] = np.reshape(curr_input_series, [-1, dim_input])

            init_c_feed[b] = np.reshape(prev_ic, [total_num_state, ])
            init_c_feed_p[b] = np.reshape(curr_ic, [total_num_state, ])
            init_mu_feed[b] = np.reshape(prev_imu, [total_num_state, ])
            init_mu_feed_p[b] = np.reshape(curr_imu, [total_num_state, ])

            # get output of retraced and new output

            feed_dict_input = {self.nn.batchX_ph: batchX[b:(b + 1)],
                               self.nn.init_c: init_c_feed[b:(b + 1)],
                               self.nn.init_mu: init_mu_feed[b:(b + 1)],
                               self.nn.coef_eps: coef_eps}

            feed_dict_input_p = {self.nn.batchX_ph: batchX_p[b:(b + 1)],
                                 self.nn.init_c: init_c_feed_p[b:(b + 1)],
                                 self.nn.init_mu: init_mu_feed_p[b:(b + 1)],
                                 self.nn.coef_eps: coef_eps}

            retrace_value_outputs, retrace_policy_outputs = self.nn.session.run(
                [self.nn.value_output_series_tensor, self.nn.policy_output_series_tensor],
                feed_dict=feed_dict_input)

            current_value_outputs, _ = self.nn.session.run(
                [self.nn.value_output_series_tensor, self.nn.policy_output_series_tensor],
                feed_dict=feed_dict_input_p)

            td_prime = np.zeros([N_level, ])
            rho_prime = 1.
            for stp in reversed(range(min(prev_step + 1, truncated_backprop_length))):

                if prev_step < truncated_backprop_length - 1:
                    current_value_output_levels = np.reshape(current_value_outputs[stp + 1, 0], [N_level, 1])
                else:
                    current_value_output_levels = np.reshape(current_value_outputs[stp, 0], [N_level, 1])

                retrace_value_output_levels = np.reshape(retrace_value_outputs[stp, 0], [N_level, 1])
                retrace_policy_output_levels = np.reshape(retrace_policy_outputs[stp, 0], [N_level, dim_action])
                previous_value_target_levels = retrace_value_output_levels
                previous_policy_target_levels = retrace_policy_output_levels

                rho = np.exp(
                    - 0.5 * (prev_a_series[stp] - np.reshape(retrace_policy_output_levels[0], [dim_action])) ** 2 / (
                        noise_scale * np.exp(- noise_decrease * episode) + min_noise + SIG_EPS) ** 2) / (
                      prev_bp_series[stp]) / (
                          noise_scale * np.exp(-noise_decrease * episode) + min_noise + SIG_EPS)

                for lev in range(N_level):

                    if done_series[stp]:
                        td_error[lev] = reward_series[stp] - np.sum(retrace_value_output_levels[lev])
                    else:
                        td_error[lev] = reward_series[stp] + gamma[lev] * np.sum(
                            current_value_output_levels[lev]) - np.sum(
                            retrace_value_output_levels[lev])

                    if not (singleV and lev == 1):
                        previous_value_target_levels[lev] += np.minimum(batch_size, np.prod(rho)) * td_error[lev] + (
                                                                                                                        1. -
                                                                                                                        done_series[
                                                                                                                            stp]) * rho_prime * \
                                                                                                                    gamma[
                                                                                                                        lev] * \
                                                                                                                    td_prime[
                                                                                                                        lev]
                    else:
                        pass

                    if lev == 0:
                        previous_policy_target_levels[lev] += np.minimum(batch_size, np.prod(rho)) * (
                            prev_a_series[stp] - retrace_policy_output_levels[lev]) * td_error[lev] / 2 / ((
                                                                                                               noise_scale * np.exp(
                                                                                                                   -noise_decrease * episode) + min_noise + SIG_EPS) ** 2)
                    else:
                        previous_policy_target_levels[lev] -= previous_policy_target_levels[lev]

                rho_prime *= min(1.0, np.prod(rho))
                td_prime += 0.0 * gamma * td_error

                batchV[b, stp] = np.reshape(previous_value_target_levels, [N_level, 1])
                batchP[b, stp] = np.reshape(previous_policy_target_levels, [N_level, dim_action])

        episode_mean_reward = self.last_ep_mean_r

        feed_dict_train = {self.nn.batchX_ph: batchX,
                           self.nn.batchP_ph: batchP,
                           self.nn.coef_eps: coef_eps,
                           self.nn.batchV_ph: batchV,
                           self.nn.reward_ph: episode_mean_reward,
                           self.nn.init_c: init_c_feed,
                           self.nn.init_mu: init_mu_feed,
                           self.nn.v_step_ph: v_step}

        return feed_dict_train

    def select(self, input, rand_noise):
        v, pi, c, mu, z, sig = self.nn(flatten(self.previous_c_levels, total_num_state),
                                       flatten(self.previous_mu_levels, total_num_state),
                                       input)

        self.current_value_output_levels = v
        self.current_policy_output_levels = pi
        self.current_c_levels = c
        self.current_z_levels = z
        self.current_mu_levels = mu
        self.current_sig_levels = sig

        a = np.reshape(pi[0], [dim_action]) + rand_noise

        return a

    def record_and_learn(self,
                         current_position,
                         current_observations,
                         previous_position,
                         previous_observations,
                         previous_action,
                         step,
                         reward,
                         done,
                         episode,
                         global_step):

        current_input = get_input(current_observations, dim_input=dim_input)
        previous_input = get_input(previous_observations, dim_input=dim_input)

        self.recorder.record_step(previous_position_=previous_position,
                                  previous_observations_=previous_observations,
                                  previous_action_=previous_action,
                                  previous_value_output_levels_=self.previous_value_output_levels,
                                  previous_policy_output_levels_=self.previous_policy_output_levels,
                                  previous_c_levels_=self.previous_c_levels,
                                  previous_mu_levels_=self.previous_mu_levels,
                                  previous_sig_levels_=self.previous_sig_levels,
                                  reward_=reward,
                                  done_=done)

        previous_policy = np.reshape(self.previous_policy_output_levels[0], [dim_action])
        bp = np.exp(- 0.5 * (previous_action - previous_policy) ** 2 / (noise_scale * np.exp(
            - noise_decrease * episode) + min_noise + SIG_EPS) ** 2) / (
                 noise_scale * np.exp(-noise_decrease * episode) + min_noise + SIG_EPS)

        if step >= truncated_backprop_length:
            ic = flatten(self.recorder.cs_series_all[step - truncated_backprop_length], total_num_state)
            imu = flatten(self.recorder.mus_series_all[step - truncated_backprop_length], total_num_state)
        else:
            ic = np.zeros(total_num_state)
            imu = np.zeros(total_num_state)

        if step + 1 >= truncated_backprop_length:
            icp = flatten(self.recorder.cs_series_all[step + 1 - truncated_backprop_length], total_num_state)
            imup = flatten(self.recorder.mus_series_all[step + 1 - truncated_backprop_length], total_num_state)
        else:
            icp = np.zeros(total_num_state)
            imup = np.zeros(total_num_state)

        mu = flatten(self.current_mu_levels, total_num_state)
        z = flatten(self.current_z_levels, total_num_state)

        self.memory.append(s=previous_input,
                           sp=current_input,
                           a=previous_action,
                           r=reward,
                           bp=bp,
                           done=done,
                           step=step,
                           ic=ic,
                           icp=icp,
                           imu=imu,
                           imup=imup,
                           mu=mu,
                           z=z)

        self.previous_c_levels = deepcopy(self.current_c_levels)
        self.previous_mu_levels = deepcopy(self.current_mu_levels)
        self.previous_sig_levels = deepcopy(self.current_sig_levels)
        self.previous_value_output_levels = deepcopy(self.current_value_output_levels)
        self.previous_policy_output_levels = deepcopy(self.current_policy_output_levels)

        if global_step > step_train_begin and global_step % step_train_freq == 0 and not (
            testing and episode % episode_test == 0):
            samples = []
            for b in range(batch_size):
                sample = self.memory.sample(truncated_steps=truncated_backprop_length)
                samples.append(sample)
            train_dict = self.get_train_dict(samples, episode)
            self.nn.train_batch(train_dict)
            if recording:
                self.nn.summary_write(train_dict, global_step)

        if done or step == max_steps - 2:
            self.recorder.record_end(current_position, get_input(current_observations, dim_input=dim_input))
            for rest_step in range(truncated_backprop_length - 1):
                step += 1
                if step >= truncated_backprop_length:
                    ic = flatten(self.recorder.cs_series_all[step - truncated_backprop_length], total_num_state)
                    imu = flatten(self.recorder.mus_series_all[step - truncated_backprop_length], total_num_state)
                else:
                    ic = np.zeros(total_num_state)
                    imu = np.zeros(total_num_state)

                if step + 1 >= truncated_backprop_length:
                    icp = flatten(self.recorder.cs_series_all[step + 1 - truncated_backprop_length], total_num_state)
                    imup = flatten(self.recorder.mus_series_all[step + 1 - truncated_backprop_length], total_num_state)
                else:
                    icp = np.zeros(total_num_state)
                    imup = np.zeros(total_num_state)

                mu = flatten(self.current_mu_levels, total_num_state)
                z = flatten(self.current_z_levels, total_num_state)

                self.memory.padding_tail(s=previous_input,
                                         sp=current_input,
                                         a=previous_action,
                                         r=0.,
                                         bp=bp,
                                         done=-1,
                                         step=step,
                                         ic=ic,
                                         icp=icp,
                                         imu=imu,
                                         imup=imup,
                                         mu=mu,
                                         z=z)


class AGENT_LSTM():
    def __init__(self, nn: LSTM, recorder: EpisodeRecorder_LSTM, memory: ReplayBuffer_LSTM):
        self.nn = nn
        self.recorder = recorder
        self.memory = memory
        self.cumu_reward = 0.
        self.last_ep_mean_r = 0.

    def init_episode(self, episode, input, rand_noise):
        self.cumu_reward = 0
        self.recorder = EpisodeRecorder_LSTM()
        self.previous_c = np.zeros([num_state])
        self.previous_h = np.zeros([num_state])

        init_action = self.select(input, rand_noise)

        self.previous_c = deepcopy(self.current_c)
        self.previous_h = deepcopy(self.current_h)

        self.previous_value_output = deepcopy(self.current_value_output)
        self.previous_policy_output = deepcopy(self.current_policy_output)

        return init_action

    def get_input_dict(self):
        pass

    def get_train_dict(self, samples, episode):
        batchX = np.zeros([batch_size, truncated_backprop_length, dim_input])
        batchX_p = np.zeros([batch_size, truncated_backprop_length, dim_input])
        init_c_feed = np.zeros([batch_size, num_state])
        init_c_feed_p = np.zeros([batch_size, num_state])
        init_h_feed = np.zeros([batch_size, num_state])
        init_h_feed_p = np.zeros([batch_size, num_state])
        v_step = np.zeros([batch_size, truncated_backprop_length])

        batchP = np.zeros([batch_size, truncated_backprop_length, dim_action])
        batchV = np.zeros([batch_size, truncated_backprop_length, 1])

        for b, sample in enumerate(samples):

            [prev_input_series, curr_input_series, prev_a_series, reward_series, prev_bp_series, done_series,
             prev_ic, curr_ic, prev_ih, curr_ih, prev_step, curr_step, done_step] = sample

            # Generate input batch
            if curr_step > done_step:
                if prev_step >= truncated_backprop_length:
                    if curr_step - done_step > 0 and curr_step - done_step < truncated_backprop_length:
                        v_step[b] = np.hstack([np.ones([truncated_backprop_length - curr_step + done_step]),
                                               np.zeros([curr_step - done_step])])
                    else:
                        pass  # zeros
                    batchX[b] = np.reshape(prev_input_series, [truncated_backprop_length, dim_input])
                else:
                    v_step[b, 0:done_step] = 1
                    batchX[b, 0:(prev_step + 1)] = np.reshape(prev_input_series, [-1, dim_input])

                if curr_step >= truncated_backprop_length:
                    batchX_p[b] = np.reshape(curr_input_series, [truncated_backprop_length, dim_input])
                else:
                    batchX_p[b, 0:(curr_step + 1)] = np.reshape(curr_input_series, [-1, dim_input])

            else:
                if prev_step >= truncated_backprop_length:
                    v_step[b] = np.ones([truncated_backprop_length])
                    batchX[b] = np.reshape(prev_input_series, [truncated_backprop_length, dim_input])
                else:
                    v_step[b, 0:(prev_step + 1)] = 1
                    batchX[b, 0:(prev_step + 1)] = np.reshape(prev_input_series, [-1, dim_input])

                if curr_step >= truncated_backprop_length:
                    batchX_p[b] = np.reshape(curr_input_series, [truncated_backprop_length, dim_input])
                else:
                    batchX_p[b, 0:(curr_step + 1)] = np.reshape(curr_input_series, [-1, dim_input])

            init_c_feed[b] = np.reshape(prev_ic, [num_state, ])
            init_c_feed_p[b] = np.reshape(curr_ic, [num_state, ])
            init_h_feed[b] = np.reshape(prev_ih, [num_state, ])
            init_h_feed_p[b] = np.reshape(curr_ih, [num_state, ])

            # get output of retraced and new output

            feed_dict_input = {self.nn.batchX_ph: batchX[b:(b + 1)],
                               self.nn.init_c: init_c_feed[b:(b + 1)],
                               self.nn.init_h: init_h_feed[b:(b + 1)]}

            feed_dict_input_p = {self.nn.batchX_ph: batchX_p[b:(b + 1)],
                                 self.nn.init_c: init_c_feed_p[b:(b + 1)],
                                 self.nn.init_h: init_h_feed_p[b:(b + 1)]}

            retrace_value_outputs, retrace_policy_outputs = self.nn.session.run(
                [self.nn.value_output_series_tensor, self.nn.policy_output_series_tensor],
                feed_dict=feed_dict_input)

            current_value_outputs, _ = self.nn.session.run(
                [self.nn.value_output_series_tensor, self.nn.policy_output_series_tensor],
                feed_dict=feed_dict_input_p)

            td_prime = 0.
            rho_prime = 0.
            for stp in reversed(range(min(prev_step + 1, truncated_backprop_length))):

                if prev_step < truncated_backprop_length - 1:
                    current_value_output = current_value_outputs[stp + 1, 0, 0]
                else:
                    current_value_output = current_value_outputs[stp, 0, 0]

                retrace_value_output = retrace_value_outputs[stp, 0, 0]
                retrace_policy_output = np.reshape(retrace_policy_outputs[stp, 0], [dim_action])
                previous_value_target = retrace_value_output
                previous_policy_target = retrace_policy_output

                rho = np.exp(
                    - 0.5 * (prev_a_series[stp] - np.reshape(retrace_policy_output, [dim_action])) ** 2 / (
                        noise_scale * np.exp(- noise_decrease * episode) + min_noise + SIG_EPS) ** 2) / prev_bp_series[
                          stp] / (
                          noise_scale * np.exp(-noise_decrease * episode) + min_noise + SIG_EPS)

                if done_series[stp]:
                    td_error = reward_series[stp] - np.sum(retrace_value_output)
                else:
                    td_error = reward_series[stp] + gamma * current_value_output - retrace_value_output

                previous_value_target += (min(batch_size, np.prod(rho)) * td_error + (1. - done_series[stp]) * min(
                    1., rho_prime) * gamma * 0.0 * td_prime)

                previous_policy_target += min(batch_size, np.prod(rho)) * (
                    prev_a_series[stp] - retrace_policy_output) * td_error / 2 / (noise_scale * np.exp(
                    - noise_decrease * episode) + min_noise + SIG_EPS) ** 2

                rho_prime = np.prod(rho)
                td_prime = td_error

                batchV[b, stp, 0] = previous_value_target
                batchP[b, stp] = np.reshape(previous_policy_target, [dim_action])

        episode_mean_reward = self.last_ep_mean_r

        feed_dict_train = {self.nn.batchX_ph: batchX,
                           self.nn.batchP_ph: batchP,
                           self.nn.batchV_ph: batchV,
                           self.nn.reward_ph: episode_mean_reward,
                           self.nn.init_c: init_c_feed,
                           self.nn.init_h: init_h_feed,
                           self.nn.v_step_ph: v_step}

        return feed_dict_train

    def select(self, input, rand_noise):
        v, pi, c, h = self.nn(self.previous_c,
                              self.previous_h,
                              input)

        self.current_value_output = v
        self.current_policy_output = pi
        self.current_c = c
        self.current_h = h

        a = np.reshape(pi, [dim_action]) + rand_noise

        return a

    def record_and_learn(self,
                         current_position,
                         current_observations,
                         previous_position,
                         previous_observations,
                         previous_action,
                         step,
                         reward,
                         done,
                         episode,
                         global_step):

        current_input = get_input(current_observations, dim_input=dim_input)
        previous_input = get_input(previous_observations, dim_input=dim_input)

        self.recorder.record_step(previous_position_=previous_position,
                                  previous_observations_=previous_observations,
                                  previous_action_=previous_action,
                                  previous_value_output_=self.previous_value_output,
                                  previous_policy_output_=self.previous_policy_output,
                                  previous_c_=self.previous_c,
                                  previous_h_=self.previous_h,
                                  reward_=reward,
                                  done_=done)

        previous_policy = np.reshape(self.previous_policy_output, [dim_action])
        bp = np.exp(- 0.5 * (previous_action - previous_policy) ** 2 / (noise_scale * np.exp(
            - noise_decrease * episode) + min_noise + SIG_EPS) ** 2) / (
                 noise_scale * np.exp(-noise_decrease * episode) + min_noise + SIG_EPS)

        if step >= truncated_backprop_length:
            ic = np.reshape(self.recorder.cs_series_all[step - truncated_backprop_length], [num_state])
            ih = np.reshape(self.recorder.hs_series_all[step - truncated_backprop_length], [num_state])
        else:
            ic = np.zeros(num_state)
            ih = np.zeros(num_state)

        if step + 1 >= truncated_backprop_length:
            icp = np.reshape(self.recorder.cs_series_all[step + 1 - truncated_backprop_length], [num_state])
            ihp = np.reshape(self.recorder.hs_series_all[step + 1 - truncated_backprop_length], [num_state])
        else:
            icp = np.zeros(num_state)
            ihp = np.zeros(num_state)

        self.memory.append(s=previous_input,
                           sp=current_input,
                           a=previous_action,
                           r=reward,
                           bp=bp,
                           done=done,
                           step=step,
                           ic=ic,
                           icp=icp,
                           ih=ih,
                           ihp=ihp)

        self.previous_c = deepcopy(self.current_c)
        self.previous_h = deepcopy(self.current_h)
        self.previous_value_output = deepcopy(self.current_value_output)
        self.previous_policy_output = deepcopy(self.current_policy_output)

        if global_step > step_train_begin and global_step % step_train_freq == 0 and not (
            testing and episode % episode_test == 0):
            samples = []
            for b in range(batch_size):
                sample = self.memory.sample(truncated_steps=truncated_backprop_length)
                samples.append(sample)
            train_dict = self.get_train_dict(samples, episode)
            self.nn.train_batch(train_dict)
            if recording:
                self.nn.summary_write(train_dict, global_step)

        if done or step == max_steps - 2:
            self.recorder.record_end(current_position, get_input(current_observations, dim_input=dim_input))
            for rest_step in range(truncated_backprop_length - 1):
                step += 1
                if step >= truncated_backprop_length:
                    ic = np.reshape(self.recorder.cs_series_all[step - truncated_backprop_length], [num_state])
                    ih = np.reshape(self.recorder.hs_series_all[step - truncated_backprop_length], [num_state])
                else:
                    ic = np.zeros(num_state)
                    ih = np.zeros(num_state)

                if step + 1 >= truncated_backprop_length:
                    icp = np.reshape(self.recorder.cs_series_all[step + 1 - truncated_backprop_length], [num_state])
                    ihp = np.reshape(self.recorder.hs_series_all[step + 1 - truncated_backprop_length], [num_state])
                else:
                    icp = np.zeros(num_state)
                    ihp = np.zeros(num_state)

                self.memory.padding_tail(s=previous_input,
                                         sp=current_input,
                                         a=previous_action,
                                         r=0.,
                                         bp=bp,
                                         done=-1,
                                         step=step,
                                         ic=ic,
                                         icp=icp,
                                         ih=ih,
                                         ihp=ihp)


def run_episode(task, agent, episode, global_step, render=False):
    step = 0
    done = False
    rand_noise = np.random.normal(0, noise_scale * np.exp(-noise_decrease * episode) + min_noise, [dim_action])

    previous_observations = task.reset()
    previous_position = deepcopy(task.old_position)
    previous_action = agent.init_episode(episode, get_input(previous_observations, dim_input=dim_input), rand_noise)

    while not done and step < max_steps - 1:
        noise_input = {}
        noise_input["type"] = 'OU'
        noise_input["episode"] = episode
        noise_input["prev_noise"] = rand_noise
        noise_input["scale"] = noise_scale
        rand_noise = get_noise(noise_input, dimension=dim_action)

        if render:
            task.render()
        try:
            current_observations, reward, done, _ = task.step(action=previous_action)
        except:
            current_observations, reward, done = task.step(action=previous_action)

        current_position = deepcopy(task.new_position)
        current_action = agent.select(get_input(current_observations, dim_input=dim_input), rand_noise)
        agent.record_and_learn(current_position, current_observations, previous_position, previous_observations,
                               previous_action, step, reward, done, episode, global_step)
        # must be executed once and only once for each step

        step += 1
        global_step += 1

        previous_position = deepcopy(current_position)
        previous_observations = deepcopy(current_observations)
        previous_action = deepcopy(current_action)

    agent.last_ep_mean_r = np.mean(agent.recorder.rewards_all)
    agent.recorder.data_cleanup()

    if saving and episode % saving_interval == 0:
        agent.recorder.save_data(task, episode)
    agent.recorder.print_result(episode)

    return step, global_step


def run_episode_lowstop(task, agent, episode, global_step, render=False):
    step = 0
    done = False
    rand_noise = np.random.normal(0, noise_scale * np.exp(-noise_decrease * episode) + min_noise, [dim_action])

    previous_observations = task.reset()
    previous_position = deepcopy(task.old_position)
    previous_action = agent.init_episode(episode, get_input(previous_observations, dim_input=dim_input), rand_noise)

    while not done and step < max_steps - 1:
        noise_input = {}
        noise_input["type"] = 'OU'
        noise_input["episode"] = episode
        noise_input["prev_noise"] = rand_noise
        noise_input["scale"] = noise_scale
        rand_noise = get_noise(noise_input, dimension=dim_action)

        if render:
            task.render()
        try:
            current_observations, reward, done, _ = task.step(action=previous_action)
        except:
            current_observations, reward, done = task.step(action=previous_action)

        current_position = deepcopy(task.new_position)
        current_action = agent.select(get_input(current_observations, dim_input=dim_input), rand_noise)
        agent.nn.record_low_weights()
        agent.record_and_learn(current_position, current_observations, previous_position, previous_observations,
                               previous_action, step, reward, done, episode, global_step)
        agent.nn.recover_low_weights()
        # must be executed once and only once for each step

        step += 1
        global_step += 1

        previous_position = deepcopy(current_position)
        previous_observations = deepcopy(current_observations)
        previous_action = deepcopy(current_action)
    agent.last_ep_mean_r = np.mean(agent.recorder.rewards_all)
    if model_name == 'LSTM':
        agent.recorder.data_cleanup(name='LSTM')
    elif model_name == 'MTSRNN':
        agent.recorder.data_cleanup(w11=agent.nn.Wcu_this[0].eval(), w22=agent.nn.Wcu_this[1].eval(),
                                    b1=agent.nn.bu[0].eval(),
                                    w21=agent.nn.Wcu_higher[0].eval(), w12=agent.nn.Wcu_lower[0].eval(),
                                    b2=agent.nn.bu[1].eval(),
                                    wo_mua_l=agent.nn.Wo_mua[0].eval(), bo_mua_l=agent.nn.bo_mua[0].eval(),
                                    wo_l=agent.nn.Wo[0].eval(), bo_l=agent.nn.bo[0].eval(),
                                    wo_h=agent.nn.Wo[1].eval(), bo_h=agent.nn.bo[1].eval())
    if saving and episode % saving_interval == 0:
        agent.recorder.save_data(task, episode)
    agent.recorder.print_result(episode)
    return step, global_step


# ## Run experiment

global_step = 0
performance_curve = np.array([])
step_curve = np.array([])

if model_name == 'MTSRNN':
    recorder = EpisodeRecorder()
    memory = ReplayBuffer()
    nn = MTSRNN()
    agent = AGENT_MTSRNN(nn, recorder, memory)
elif model_name == 'LSTM':
    recorder = EpisodeRecorder_LSTM()
    memory = ReplayBuffer_LSTM()
    nn = LSTM()
    agent = AGENT_LSTM(nn, recorder, memory)

## Phase 1

task = TaskT(3)
episode_beg = 0

for episode in range(num_episodes):
    steps_take, global_step = run_episode(task, agent, episode, global_step, render=False)
    if plotting and episode % episode_plot == 0:
        agent.recorder.plot_episode(task, episode)
    performance_curve = np.append(performance_curve, agent.recorder.get_performance())
    step_curve = np.append(step_curve, steps_take)

## Phase 2

m = 4  # episode discount in phase 2 and 3

task = TaskT(3, 'GBR')
episode_beg = num_episodes

noise_decrease = noise_decrease * m
global_step = 0

if model_name == 'MTSRNN':
    memory = ReplayBuffer()
    agent.memory = memory
elif model_name == 'LSTM':
    memory = ReplayBuffer_LSTM()
    agent.memory = memory

for episode in range(int(num_episodes / m)):
    steps_take, global_step = run_episode(task, agent, episode, global_step, render=False)
    if plotting and episode % episode_plot == 0:
        agent.recorder.plot_episode(task, episode)
    performance_curve = np.append(performance_curve, agent.recorder.get_performance())
    step_curve = np.append(step_curve, steps_take)

## Phase 3

task = TaskT(3, 'BGR')

global_step = 0
episode_beg = num_episodes + int(num_episodes / m)

if model_name == 'MTSRNN':
    memory = ReplayBuffer()
    agent.memory = memory
elif model_name == 'LSTM':
    memory = ReplayBuffer_LSTM()
    agent.memory = memory

if (not lowstop3) or (model_name == LSTM):
    for episode in range(int(num_episodes / m)):
        steps_take, global_step = run_episode(task, agent, episode, global_step, render=False)
        if plotting and episode % episode_plot == 0:
            agent.recorder.plot_episode(task, episode)
        performance_curve = np.append(performance_curve, agent.recorder.get_performance())
        step_curve = np.append(step_curve, steps_take)

else:
    for episode in range(int(num_episodes / m)):
        steps_take, global_step = run_episode_lowstop(task, agent, episode, global_step, render=False)
        if plotting and episode % episode_plot == 0:
            agent.recorder.plot_episode(task, episode)
        performance_curve = np.append(performance_curve, agent.recorder.get_performance())
        step_curve = np.append(step_curve, steps_take)


performance_data = {'rewards': np.reshape(performance_curve, [-1, 1]), 'steps': step_curve}

scipy.io.savemat(perfsavepath + filename + 'performance' + '.mat', performance_data)
