
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# NUMERICAL PRECISION EXPERIMENT
# ============================================================

STEPS = 500
TEST_DT = 0.001

# Generate one shared initial configuration in float64.
rng = np.random.default_rng(42)
initial_pos = rng.random((N, 2)) * L
initial_vel = (rng.random((N, 2)) - 0.5) * 2.0

def precision_run(dtype):

    # Identical initial conditions, converted to chosen precision.
    pos = initial_pos.astype(dtype)
    vel = initial_vel.astype(dtype)
    masses = np.full(N, m, dtype=dtype)

    k_local = dtype(k)
    r0_local = dtype(r0)
    rc_local = dtype(rc)
    gamma_local = dtype(0.0)  # Conservative comparison

    def force_function(pos, vel):

        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=-1))

        neighbours = (dist > 0) & (dist <= rc_local)

        unit = d / np.maximum(
            dist[:, :, None],
            dtype(1e-12)
        )

        fmag = -k_local * (dist - r0_local) * neighbours
        spring = np.sum(fmag[:, :, None] * unit, axis=1)

        drag = -gamma_local * vel

        return spring + drag

    def energy_function(pos, vel):

        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=-1))

        neighbours = (dist > 0) & (dist <= rc_local)

        KE = dtype(0.5) * np.sum(
            masses * np.sum(vel * vel, axis=1)
        )

        # Shifted potential, consistent between both runs.
        pair_U = (
            dtype(0.5) * k_local
            * ((dist - r0_local)**2 - (rc_local - r0_local)**2)
            * neighbours
        )
        PE = np.sum(pair_U) / dtype(2.0)

        return float(KE + PE)

    energy_history = [energy_function(pos, vel)]

    # Measure execution time for the same number of steps.
    start = time.perf_counter()

    for step in range(STEPS):

        # Semi-implicit Euler: identical integrator for both dtypes.
        force = force_function(pos, vel)
        vel += (force / masses[:, None]) * dtype(TEST_DT)
        pos += vel * dtype(TEST_DT)

        # Reflect at the walls.
        for axis in range(2):
            low = pos[:, axis] < 0
            pos[low, axis] *= -1
            vel[low, axis] = np.abs(vel[low, axis])

            high = pos[:, axis] > L
            pos[high, axis] = dtype(2 * L) - pos[high, axis]
            vel[high, axis] = -np.abs(vel[high, axis])

        energy_history.append(energy_function(pos, vel))

    elapsed = time.perf_counter() - start

    # Report the main arrays' memory, not temporary arrays.
    array_memory = pos.nbytes + vel.nbytes + masses.nbytes

    return {
        "dtype": np.dtype(dtype).name,
        "position": pos.copy().astype(np.float64),
        "energy": np.array(energy_history, dtype=np.float64),
        "time_seconds": elapsed,
        "array_memory_bytes": array_memory,
        "final_energy": energy_history[-1],
        "initial_energy": energy_history[0],
        "energy_history": np.array(energy_history, dtype=np.float64)
    }

# Run both simulations.
result32 = precision_run(np.float32)
result64 = precision_run(np.float64)

# ============================================================
# COMPARE RESULTS
# ============================================================

pos32 = result32["position"]
pos64 = result64["position"]

position_error = (
    np.linalg.norm(pos32 - pos64)
    / max(np.linalg.norm(pos64), 1e-15)
)

energy_drift32 = (
    abs(result32["final_energy"] - result32["initial_energy"])
    / max(abs(result32["initial_energy"]), 1e-15)
)

energy_drift64 = (
    abs(result64["final_energy"] - result64["initial_energy"])
    / max(abs(result64["initial_energy"]), 1e-15)
)

comparison = pd.DataFrame({
    "Metric": [
        "Execution time (seconds)",
        "Main array memory (bytes)",
        "Initial total energy",
        "Final total energy",
        "Maximum relative energy drift",
    ],
    "float32": [
        result32["time_seconds"],
        result32["array_memory_bytes"],
        result32["initial_energy"],
        result32["final_energy"],
        np.max(np.abs(
            result32["energy_history"] - result32["initial_energy"]
        )) / max(abs(result32["initial_energy"]), 1e-15),
    ],
    "float64": [
        result64["time_seconds"],
        result64["array_memory_bytes"],
        result64["initial_energy"],
        result64["final_energy"],
        np.max(np.abs(
            result64["energy_history"] - result64["initial_energy"]
        )) / max(abs(result64["initial_energy"]), 1e-15),
    ]
})

display(comparison)

print(f"Relative final-position error: {position_error:.6e}")
print(f"Final-position difference norm: {np.linalg.norm(pos32 - pos64):.6e}")

# ============================================================
# PLOT ENERGY HISTORIES
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(result32["energy_history"], label="float32")
plt.plot(result64["energy_history"], label="float64")

plt.xlabel("Simulation step")
plt.ylabel("Total mechanical energy")
plt.title("Numerical precision: energy comparison")
plt.legend()
plt.tight_layout()
plt.show()
