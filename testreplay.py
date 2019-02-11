import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import os
import timeit
import random
from collections import deque




class ReplayMemory:
    """Replay Memory that stores the last size=1,000,000 transitions"""

    def __init__(self, size=15, frame_height=2, frame_width=2,
                 agent_history_length=4, batch_size=1):
        """
            Args:
                size: Integer, Number of stored transitions
                frame_height: Integer, Height of a frame of an Atari game
                frame_width: Integer, Width of a frame of an Atari game
                agent_history_length: Integer, Number of frames stacked together to create a state
                batch_size: Integer, Number if transitions returned in a minibatch
            """
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, frame_height, frame_width), dtype=np.uint8)
        self.dones = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.frame_height,
                                self.frame_width, self.agent_history_length), dtype=np.uint8)
        self.next_states = np.empty((self.batch_size, self.frame_height,
                                     self.frame_width, self.agent_history_length), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, reward, done, next_frame):
        self.actions[self.count] = action
        self.frames[self.count] = next_frame
        self.rewards[self.count] = reward
        self.dones[self.count] = done
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
            else :
                frames_to_stack.appendleft(self.frames[frame_number - i])
                last_not_done = frame_number - i

        stacked_state = np.stack(frames_to_stack, axis=2)
        return stacked_state

    def get_mini_batch(self):
        indexes = []

        batch_size_count = 0
        while batch_size_count < self.batch_size:
            index = random.randint(4 ,self.count - 1)
            if not self.dones[index - 1]:
                indexes.append(index)
                batch_size_count += 1

        print(indexes)

        count = 0
        for i in indexes:
            self.states[count] = self.get_stacked_state(i - 1)
            self.next_states[count] = self.get_stacked_state(i)
            count += 1

        return self.states, self.actions[indexes], self.rewards[indexes], self.next_states, self.dones[indexes]



env = gym.make('Breakout-v0')
rm = ReplayMemory()
rm.add_experience(0, 0, False, [[0,1],[2,3]])
rm.add_experience(11, 12, True, [[4,5],[6,7]])

rm.add_experience(21, 22, False, [[8,9],[10,11]])
rm.add_experience(31, 33, False, [[12,13],[14,15]])
rm.add_experience(41, 42, True, [[16,17],[18,19]])

rm.add_experience(51, 52, False, [[20,21],[22,23]])
rm.add_experience(61, 63, False, [[24,25],[26,27]])

states, actions, rewards, next_states, dones = rm.get_mini_batch()
print("states \n", states)
print("actions \n", actions)
print("next_states \n", next_states)
print("dones \n", dones)