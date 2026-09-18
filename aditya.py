#include <stdio.h>

import numpy as np 



def initialize(N, L):
    pos = np.random.rand(N, 2) * L
    vel = (np.random.rand(N, 2) - 0.5) * 2
    mass = np.ones(N)
    return pos, vel, mass

def walls_reflect(pos, vel, L):
    for i in range(len(pos)):
        if pos[i, 0] < 0:
            pos[i, 0] = 0
            vel[i, 0] *= -1
        elif pos[i, 0] > L:
            pos[i, 0] = L
            vel[i, 0] *= -1

        if pos[i, 1] < 0:
            pos[i, 1] = 0
            vel[i, 1] *= -1
        elif pos[i, 1] > L:
            pos[i, 1] = L
            vel[i, 1] *= -1


def main():
    print("hello world")

if __name__ == "__main__":
    main()
