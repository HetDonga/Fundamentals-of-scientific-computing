import numpy as np

np.random.seed(0)
N = 12 # number of particles
L = 10.0 # box side length
dt = 0.01 # time step
T = 1000 # number of steps to simulate

pos = np.random.rand(N, 2)
vel = (np.random.rand(N, 2) - 0.5) * 2
# Note: fixed a minor syntax error here (was np.ones(N,1))
mass = np.ones(N) 

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



def pairwise_distances(pos, rc):
    # Broadcast to get (N, N, 2) displacement vectors without for loops
    d = pos[:, None, :] - pos[None, :, :]
    
    # Calculate lengths of displacement vectors to get (N, N) distance matrix
    dist = np.linalg.norm(d, axis=-1)
    
    # Boolean mask: True where 0 < dist <= rc
    neighbors = (dist > 0) & (dist <= rc)
    
    return d, dist, neighbors

