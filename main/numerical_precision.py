import time
import numpy as np
import matplotlib.pyplot as plt

# 1. PARAMETERS
np.random.seed(0)

N = 45
L = 10.0
dt = 0.005
T = 2000

m = 1.0
k = 10.0
r0 = 0.8
rc = 2.0

gamma = 0  # Conservative dynamics

# 2. IDENTICAL INITIAL CONDITIONS
# Generate once in float64, then convert for both simulations.

initial_pos = np.random.rand(N, 2) * L
initial_vel = (np.random.rand(N, 2) - 0.5) * 2.0 + 10.0

# 3. WALL REFLECTIONS

def walls_reflect(pos, vel, L):
    hit = pos[:, 0] < 0
    pos[hit, 0] = -pos[hit, 0]
    vel[hit, 0] = np.abs(vel[hit, 0])

    hit = pos[:, 0] > L
    pos[hit, 0] = 2 * L - pos[hit, 0]
    vel[hit, 0] = -np.abs(vel[hit, 0])

    hit = pos[:, 1] < 0
    pos[hit, 1] = -pos[hit, 1]
    vel[hit, 1] = np.abs(vel[hit, 1])

    hit = pos[:, 1] > L
    pos[hit, 1] = 2 * L - pos[hit, 1]
    vel[hit, 1] = -np.abs(vel[hit, 1])

    return pos, vel


# 4. PAIRWISE DISTANCES AND NEIGHBOURS

def pairwise_distances(pos, rc):
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(d, axis=-1)
    neighbors = (dist > 0) & (dist <= rc)

    return d, dist, neighbors

# 5. SPRING FORCES

def pair_forces(pos):
    d, dist, neighbors = pairwise_distances(pos, rc)

    unit = d / np.maximum(dist[:, :, None], 1e-12)
    fmag = -k * (dist - r0) * neighbors

    return np.sum(fmag[:, :, None] * unit, axis=1)

# 6. DRAG AND TOTAL FORCES

def drag_forces(vel, gamma):
    return -gamma * vel


def total_forces(pos, vel, gamma):
    return pair_forces(pos) + drag_forces(vel, gamma)

# 7. ENERGY CALCULATIONS

def kinetic_energy(vel, mass):
    speed_squared = np.sum(vel**2, axis=1)
    return 0.5 * np.sum(mass * speed_squared)


def potential_energy(pos):
    d, dist, neighbors = pairwise_distances(pos, rc)

    pair_energy = (
        0.5 * k * ((dist - r0)**2 - (rc - r0)**2)
    ) * neighbors

    return 0.5 * np.sum(pair_energy)


def total_energy(pos, vel, mass):
    return kinetic_energy(vel, mass) + potential_energy(pos)

# 8. INTEGRATION

def integrate_step(pos, vel, mass, dt, gamma):
    F0 = pair_forces(pos)
    a0 = (F0 - gamma * vel) / mass[:, None]

    new_pos = pos + vel * dt + 0.5 * a0 * dt**2

    F1 = pair_forces(new_pos)

    numerator = vel + 0.5 * dt * (
        a0 + F1 / mass[:, None]
    )

    denominator = (
        1.0 + 0.5 * dt * gamma / mass[:, None]
    )

    new_vel = numerator / denominator

    walls_reflect(new_pos, new_vel, L)

    pos[:] = new_pos
    vel[:] = new_vel

    return pos, vel

# 9. RUN ONE PRECISION

def run_simulation(dtype):
    pos = initial_pos.astype(dtype).copy()
    vel = initial_vel.astype(dtype).copy()
    mass = np.full(N, m, dtype=dtype)

    # Memory for the main simulation state arrays.
    memory_bytes = pos.nbytes + vel.nbytes + mass.nbytes

    initial_energy = float(total_energy(pos, vel, mass))

    kinetic_history = np.empty(T, dtype=np.float64)
    potential_history = np.empty(T, dtype=np.float64)
    total_history = np.empty(T, dtype=np.float64)

    start_time = time.perf_counter()

    for step in range(T):
        integrate_step(pos, vel, mass, dt, gamma)

        kinetic_history[step] = kinetic_energy(vel, mass)
        potential_history[step] = potential_energy(pos)
        total_history[step] = (
            kinetic_history[step] + potential_history[step]
        )

    execution_time = time.perf_counter() - start_time

    # Save positions at every step for trajectory comparison.
    # This array is excluded from the state-memory measurement.
    # To avoid changing the simulation's memory use, only final
    # positions are retained for the main comparison.
    final_positions = pos.astype(np.float64).copy()

    max_energy_drift = (
        np.max(np.abs(total_history - initial_energy))
        / max(abs(initial_energy), 1e-15)
    )

    return {
        "dtype": np.dtype(dtype).name,
        "positions": final_positions,
        "kinetic": kinetic_history,
        "potential": potential_history,
        "total": total_history,
        "initial_energy": initial_energy,
        "final_energy": float(total_history[-1]),
        "max_energy_drift": float(max_energy_drift),
        "runtime": execution_time,
        "memory_bytes": memory_bytes,
    }

# 10. RUN FLOAT32 AND FLOAT64

print("Running float32 simulation...")
result32 = run_simulation(np.float32)

print("Running float64 simulation...")
result64 = run_simulation(np.float64)

# 11. MEMORY, TIME, ENERGY AND TRAJECTORY COMPARISON

memory32 = result32["memory_bytes"]
memory64 = result64["memory_bytes"]

relative_position_error = (
    np.linalg.norm(
        result32["positions"] - result64["positions"]
    )
    / max(np.linalg.norm(result64["positions"]), 1e-15)
)

print("\n========== TASK 5 RESULTS ==========")

print("\nMemory usage of positions, velocities and masses:")
print(f"float32: {memory32} bytes")
print(f"float64: {memory64} bytes")
print(f"float64 / float32 memory ratio: {memory64 / memory32:.2f}")

print("\nExecution time:")
print(f"float32: {result32['runtime']:.6f} seconds")
print(f"float64: {result64['runtime']:.6f} seconds")

if result32["runtime"] > 0:
    print(
        "Runtime ratio (float64 / float32): "
        f"{result64['runtime'] / result32['runtime']:.3f}"
    )

print("\nEnergy comparison:")
for result in (result32, result64):
    print(f"\n{result['dtype']}:")
    print(f"Initial energy: {result['initial_energy']:.10f}")
    print(f"Final energy:   {result['final_energy']:.10f}")
    print(
        f"Maximum relative energy drift: "
        f"{result['max_energy_drift']:.6e}"
    )

print("\nTrajectory comparison:")
print(
    "Relative final-position error: "
    f"{relative_position_error:.8e}"
)

print("====================================")

# 12. PLOT ENERGY COMPARISON

steps = np.arange(1, T + 1)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

axes[0, 0].plot(steps, result32["kinetic"], label="float32")
axes[0, 0].plot(steps, result64["kinetic"], label="float64")
axes[0, 0].set_title("Kinetic Energy")
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("Energy")
axes[0, 0].legend()

axes[0, 1].plot(steps, result32["potential"], label="float32")
axes[0, 1].plot(steps, result64["potential"], label="float64")
axes[0, 1].set_title("Potential Energy")
axes[0, 1].set_xlabel("Step")
axes[0, 1].set_ylabel("Energy")
axes[0, 1].legend()

axes[1, 0].plot(steps, result32["total"], label="float32")
axes[1, 0].plot(steps, result64["total"], label="float64")
axes[1, 0].set_title("Total Energy")
axes[1, 0].set_xlabel("Step")
axes[1, 0].set_ylabel("Energy")
axes[1, 0].legend()

axes[1, 1].scatter(
    result32["positions"][:, 0],
    result32["positions"][:, 1],
    marker="x",
    label="float32",
    s=35
)

axes[1, 1].scatter(
    result64["positions"][:, 0],
    result64["positions"][:, 1],
    facecolors="none",
    edgecolors="blue",
    label="float64",
    s=35
)

axes[1, 1].set_title(
    f"Final positions\nRelative error = {relative_position_error:.3e}"
)
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
axes[1, 1].set_xlim(0, L)
axes[1, 1].set_ylim(0, L)
axes[1, 1].set_aspect("equal")
axes[1, 1].legend()

plt.tight_layout()
plt.show()
if __name__ == "__main__":
    main()
