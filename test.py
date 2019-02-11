import gym
import time
import numpy as np
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imresize
import sys
np.set_printoptions(threshold=sys.maxsize)
from collections import deque


env = gym.make('Breakout-v0')
frame = env.reset()
def preprocess_frame(frame):
    # 80x80
    frame = frame[33:193, :]  # Crop the image
    frame = frame[::2, ::2]  # Reduce dimensions by taking every 2nd pixel

    frame = np.mean(frame, axis=2).astype(np.uint8)  # Grayscale - change (R,G,B) to (COLOUR)
    return frame
for i in range(200):
    frame, _,_,_ = env.step(1)
    n = preprocess_frame(frame)
    plt.imshow(n)
    plt.show()
    time.sleep(0.2)