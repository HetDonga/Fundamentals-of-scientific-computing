#include <stdio.h>

import numpy as np
np.random.seed(0)
N = 12 # number of particles
L = 10.0 # box side length
dt = 0.01 # time step
T = 1000 # number of steps to simulate

def initialize():
  pos = np.random.rand(N,2)
  vel = (np.random.rand(N,2)-0.5)*2
  mass = np.ones(N,1)

def walls_reflect(pos, vel, L):
  # Left wall (x < 0)
  left_hits = pos[:, 0] < 0
  vel[left_hits, 0] *= -1 # flip x velocity
  pos[left_hits, 0] = 0.0 # push back inside
  # Right wall (x > L)
  right_hits = pos[:, 0] > L
  vel[right_hits, 0] *= -1
  pos[right_hits, 0] = L
  # Bottom wall (y < 0)
  bottom_hits = pos[:, 1] < 0
  vel[bottom_hits, 1] *= -1
  pos[bottom_hits, 1] = 0.0
  # Top wall (y > L)
  top_hits = pos[:, 1] > L
  vel[top_hits, 1] *= -1
  pos[top_hits, 1] = L
  return pos, vel
