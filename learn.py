import os
import random
import gym
import tensorflow as tf
import numpy as np
from collections import deque
import scipy.misc
import argparse
import re
import time

parser = argparse.ArgumentParser()
parser.add_argument('-m', choices=['train', 'test'], default='train', help='Method of the program. train or test')
parser.add_argument('-e', choices=['Breakout-v4', 'BreakoutDetermenistic-v4'], default='BreakoutDetermenistic-v4', help='Method of the program. train or test')
parser.add_argument('-f', help='Model file location')

args = parser.parse_args()

METHOD = args.m
MODEL_FILE = args.f
ENVIRONMENT_NAME = args.e

if(not MODEL_FILE == None):
    MODEL_STEP = (int) (re.findall('\d+', MODEL_FILE)[0])


def preprocess_frame(frame):
    frame = frame[33:193]  # crop
    frame = np.mean(frame, axis=2).astype(np.uint8)  # Grayscale - change (R,G,B) to (COLOUR)
    frame = scipy.misc.imresize(frame, (84, 84), "nearest")  # resize
    return frame


class DQN:
    def __init__(self, n_actions, hidden=1024, learning_rate=0.00001,
                 frame_height=84, frame_width=84, agent_history_length=4):
        self.n_actions = n_actions
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length

        self.input = tf.placeholder(shape=[None, self.frame_height,
                                           self.frame_width, self.agent_history_length],
                                    dtype=tf.float32)
        # Normalizing the input
        self.inputscaled = self.input / 255

        # Convolutional layers
        self.conv1 = tf.layers.conv2d(
            inputs=self.inputscaled, filters=32, kernel_size=[8, 8], strides=4,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv1')
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=2,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv2')
        self.conv3 = tf.layers.conv2d(
            inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv3')
        self.conv4 = tf.layers.conv2d(
            inputs=self.conv3, filters=hidden, kernel_size=[6, 6], strides=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid", activation=tf.nn.relu, use_bias=False, name='conv4')

        # Splitting into value and advantage stream
        self.valuestream, self.advantagestream = tf.split(self.conv4, 2, 3)
        self.valuestream = tf.layers.flatten(self.valuestream)
        self.advantagestream = tf.layers.flatten(self.advantagestream)
        self.advantage = tf.layers.dense(
            inputs=self.advantagestream, units=self.n_actions,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name="advantage")
        self.value = tf.layers.dense(
            inputs=self.valuestream, units=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2), name='value')

        # Combining value and advantage into Q-values as described above
        self.q_values = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        self.best_action = tf.argmax(self.q_values, 1)

        # The next lines perform the parameter update. This will be explained in detail later.

        # targetQ according to Bellman equation:
        # Q = r + gamma*max Q', calculated in the function learn()
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        # Action that was performed
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        # Q value of the action that was performed
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, tf.one_hot(self.action, self.n_actions, dtype=tf.float32)),
                               axis=1)

        # Parameter updates
        self.loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.target_q, predictions=self.Q))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = self.optimizer.minimize(self.loss)


class ActionGetter:
    def __init__(self, n_actions, eps_initial=1, frame_start1 = 5000, frame_start2 = 1000000, eps_frame_start2 = 0.1, eps_final=0.01, max_frames=25000000):
        self.n_actions = n_actions
        self.eps_initial = eps_initial
        self.frame_start1 = frame_start1
        self.frame_start2 = frame_start2
        self.eps_frame_start2 = eps_frame_start2
        self.eps_final = eps_final
        self.max_frames = max_frames

    def get_action(self, session, frame_number, state, main_dqn, test=False):
        if test:
            return session.run(main_dqn.best_action, feed_dict={main_dqn.input: [state]})[0]
        elif frame_number < self.frame_start1:
            eps = self.eps_initial
        elif frame_number < self.frame_start2:
            eps = self.eps_initial - (((frame_number - self.frame_start1) * (self.eps_initial - self.eps_frame_start2))/(self.frame_start2 - self.frame_start1))
        elif frame_number >= self.frame_start2:
            eps = self.eps_frame_start2 - (((frame_number - self.frame_start2) * (self.eps_frame_start2 - self.eps_final))/(self.max_frames - self.frame_start2))

        p = np.random.random()
        if p < eps:
            return np.random.randint(0, self.n_actions)
        else:
            return session.run(main_dqn.best_action, feed_dict={main_dqn.input: [state]})[0]


class ReplayMemory:
    def __init__(self, size, frame_height, frame_width, batch_size):
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.batch_size = batch_size
        self.count = 0
        self.filled = False

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, frame_height, frame_width), dtype=np.uint8)
        self.dones = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.frame_height,
                                self.frame_width, 4), dtype=np.uint8)
        # Diff self.batch_size, self.agent_history_length, self.frame_height, self.frame_width

        self.next_states = np.empty((self.batch_size, self.frame_height,
                                     self.frame_width, 4), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, reward, done, next_frame):
        self.actions[self.count] = action
        self.frames[self.count] = next_frame # Diff frames[self.current, ...]
        self.rewards[self.count] = reward
        self.dones[self.count] = done
        if self.count == self.size - 1:
            self.filled = True
            self.count = 0
        else:
            self.count += 1

    def get_stacked_state(self, frame_number):
        frames_to_stack = deque()
        frames_to_stack.append(self.frames[frame_number])
        last_not_done = frame_number
        found_done = False
        for i in range(1, 4):
            if self.dones[frame_number - i] or found_done:
                frames_to_stack.appendleft(self.frames[last_not_done])
                found_done = True
            else:
                frames_to_stack.appendleft(self.frames[frame_number - i])
                last_not_done = frame_number - i

        stacked_state = np.stack(frames_to_stack, axis=2)
        return stacked_state

    def get_minibatch(self):
        indexes = []

        if self.filled:
            upper_limit = self.size
        else:
            upper_limit = self.count

        batch_size_count = 0
        while batch_size_count < self.batch_size:
            index = random.randint(4, upper_limit - 1)
            if index not in indexes and not self.dones[index - 1]:
                indexes.append(index)
                batch_size_count += 1


        count = 0
        for i in indexes:
            self.states[count] = self.get_stacked_state(i - 1)
            self.next_states[count] = self.get_stacked_state(i)
            count += 1


        return self.states, self.actions[indexes], self.rewards[indexes], self.next_states, self.dones[indexes]


def learn(session, replay_memory, main_dqn, target_dqn, batch_size, gamma):
    # Draw a minibatch from the replay memory
    states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()
    # The main network estimates which action is best (in the next
    # state s', new_states is passed!)
    # for every transition in the minibatch
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
    # The target network estimates the Q-values (in the next state s', new_states is passed!)
    # for every transition in the minibatch
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
    double_q = q_vals[range(batch_size), arg_q_max]
    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = rewards + (gamma*double_q * (1-terminal_flags))
    # Gradient descend step to update the parameters of the main network
    loss, _ = session.run([main_dqn.loss, main_dqn.update],
                          feed_dict={main_dqn.input:states,
                                     main_dqn.target_q:target_q,
                                     main_dqn.action:actions})
    return loss


class Atari:
    def __init__(self, envName, no_op_steps=10, agent_history_length=4):
        self.env = gym.make(envName)
        self.state = None
        self.last_lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = agent_history_length
        self.frame_deque = deque(maxlen=4)

    def reset(self, test=False):
        frame = self.env.reset()
        self.last_lives = 0
        if test:
            for _ in range(random.randint(1, self.no_op_steps)):
                frame, _, _, _ = self.env.step(1)
        processed_frame = preprocess_frame(frame)
        self.frame_deque.append(processed_frame)
        self.frame_deque.append(processed_frame)
        self.frame_deque.append(processed_frame)
        self.frame_deque.append(processed_frame)
        self.state = np.stack(self.frame_deque, axis=2)

        return processed_frame


    def step(self, action):
        new_frame, reward, terminal, info = self.env.step(action)

        terminal_life_lost = False
        # Checks if we have died
        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
            self.last_lives -= 1
        processed_new_frame = preprocess_frame(new_frame)
        self.frame_deque.append(processed_new_frame)
        new_state = np.stack(self.frame_deque, axis=2)
        self.state = new_state

        return processed_new_frame, reward, terminal, terminal_life_lost

FRAME_HEIGHT = 84
FRAME_WIDTH = 84

# Control parameters
MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
MODEL_SAVE_FREQ = 200000          # Number of frames the agent sees between saving model
EVAL_STEPS = 10000               # Number of frames for one evaluation
NETW_UPDATE_FREQ = 10000         # Number of chosen actions between updating the target network.
                                 # According to Mnih et al. 2015 this is measured in the number of
                                 # parameter updates (every four actions), however, in the
                                 # DeepMind code, it is clearly measured in the number
                                 # of actions the agent choses
DISCOUNT_FACTOR = 0.99           # gamma in the Bellman equation
REPLAY_MEMORY_START_SIZE = 50000 # Number of completely random actions,
                                 # before the agent starts learning
MAX_FRAMES = 30000000            # Total number of frames the agent sees
MEMORY_SIZE = 500000            # Number of transitions stored in the replay memory
NO_OP_STEPS = 10                 # Number of 'NOOP' or 'FIRE' actions at the beginning of an
                                 # evaluation episode
UPDATE_FREQ = 4                  # Every four actions a gradient descend step is performed
HIDDEN = 1024                    # Number of filters in the final convolutional layer. The output
                                 # has the shape (1,1,1024) which is split into two streams. Both
                                 # the advantage stream and value stream have the shape
                                 # (1,1,512). This is slightly different from the original
                                 # implementation but tests I did with the environment Pong
                                 # have shown that this way the score increases more quickly
LEARNING_RATE = 0.00001          # Set to 0.00025 in Pong for quicker results.
                                 # Hessel et al. 2017 used 0.0000625
BS = 32                          # Batch size

tf.reset_default_graph()
PATH = "output/"                 # Gifs and checkpoints will be saved here
SUMMARIES = "summaries"          # logdir for tensorboard

atari = Atari(ENVIRONMENT_NAME, NO_OP_STEPS)

with tf.variable_scope('mainDQN'):
    MAIN_DQN = DQN(atari.env.action_space.n, HIDDEN, LEARNING_RATE)   # (★★)
with tf.variable_scope('targetDQN'):
    TARGET_DQN = DQN(atari.env.action_space.n, HIDDEN)               # (★★)

TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')


init = tf.global_variables_initializer()
saver = tf.train.Saver()

main_dqn_vars = tf.trainable_variables("mainDQN")
target_dqn_vars = tf.trainable_variables("targetDQN")

def update_target_network(sess):
    for i, var in enumerate(main_dqn_vars):
        copy_op = target_dqn_vars[i].assign(var.value())
        sess.run(copy_op)

def save_model(saver, sess, path):
    saver.save(sess, path)
    print("Model saved in path: %s" % path)


def train():
    replay_memory = ReplayMemory(MEMORY_SIZE,
                                 FRAME_HEIGHT,
                                 FRAME_WIDTH,
                                 BS)

    action_getter = ActionGetter(atari.env.action_space.n)

    with tf.Session() as sess:
        sess.run(init)
        if MODEL_STEP:
            saver.restore(sess, "./models/model{}.ckpt".format(MODEL_STEP))
            print("MODEL RESTORED")

        frame_number = 0


        rewards = []
        loss_list = []

        while frame_number < MAX_FRAMES:
            atari.reset()
            episode_reward_sum = 0

            for _ in range(MAX_EPISODE_LENGTH):
                action = action_getter.get_action(sess, frame_number, atari.state, MAIN_DQN)

                processed_new_frame, reward, terminal, terminal_life_lost = atari.step(action)
                frame_number += 1

                episode_reward_sum += reward

                replay_memory.add_experience(action,
                                             reward,
                                             terminal_life_lost,
                                             processed_new_frame)
                if frame_number > REPLAY_MEMORY_START_SIZE:
                    if frame_number % UPDATE_FREQ:
                        loss = learn(sess, replay_memory, MAIN_DQN, TARGET_DQN, BS, DISCOUNT_FACTOR)
                        loss_list.append(loss)

                    if frame_number % NETW_UPDATE_FREQ == 0:
                        update_target_network(sess)

                if frame_number % MODEL_SAVE_FREQ == 0:
                    if MODEL_STEP:
                        print("/models/model{}.ckpt".format(frame_number + MODEL_STEP))
                        save_model(saver, sess, "./models/model{}.ckpt".format(frame_number + MODEL_STEP))
                    else:
                        print("/models/model{}.ckpt".format(frame_number))
                        save_model(saver, sess, "./models/model{}.ckpt".format(frame_number))

                if terminal:
                    break


            rewards.append(episode_reward_sum)


            if len(rewards) % 10 == 0:
                print(len(rewards), frame_number, np.mean(rewards[-100:]))
                with open('rewards.dat', 'a') as reward_file:
                    print(len(rewards), frame_number,
                          np.mean(rewards[-100:]), file=reward_file)


class StuckDetector:
    def __init__(self, frame):
        self.prev_frame = frame

    def unstuck(self, new_frame, env):
        if np.array_equal(new_frame, self.prev_frame):
            env.step(1)
        self.prev_frame = new_frame

def test():
    action_getter = ActionGetter(atari.env.action_space.n, test=True)

    with tf.Session() as sess:
        saver.restore(sess, "./models/model{}.ckpt".format(MODEL_STEP))
        while True:
            ep_reward = 0
            processed_frame = atari.reset(test=True)
            stuck_detector = StuckDetector(processed_frame)
            while True:
                atari.env.render()
                action = action_getter.get_action(sess, None, atari.state, MAIN_DQN, test=True)
                processed_new_frame, reward, terminal, _ = atari.step(action)
                stuck_detector.unstuck(processed_new_frame, atari.env)
                ep_reward += reward
                # time.sleep(0.02)
                if terminal:
                    break
            print("EPISODE TOTAL REWARD: {}".format(ep_reward))

if METHOD == 'train':
    train()
elif METHOD == 'test':
    test()
