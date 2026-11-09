import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# 1. PARAMETERS
# ============================================================

np.random.seed(0)

N = 45
L = 10.0
dt = 0.005
T = 2000

m = 1.0
mass = np.full(N, m, dtype=float)

k = 10.0
r0 = 0.8
rc = 2.0

gamma = 0  # Set to 0 for conservative dynamics

# ============================================================
# 2. INITIAL POSITIONS AND VELOCITIES
# ============================================================

pos = np.random.rand(N, 2) * L
vel = (np.random.rand(N, 2) - 0.5) * 2.0+10

# ============================================================
# 3. WALL REFLECTIONS
# ============================================================


def walls_reflect(pos, vel, L):

    # Left wall
    hit = pos[:, 0] < 0
    pos[hit, 0] = -pos[hit, 0]
    vel[hit, 0] = np.abs(vel[hit, 0])

    # Right wall
    hit = pos[:, 0] > L
    pos[hit, 0] = 2 * L - pos[hit, 0]
    vel[hit, 0] = -np.abs(vel[hit, 0])

    # Bottom wall
    hit = pos[:, 1] < 0
    pos[hit, 1] = -pos[hit, 1]
    vel[hit, 1] = np.abs(vel[hit, 1])

    # Top wall
    hit = pos[:, 1] > L
    pos[hit, 1] = 2 * L - pos[hit, 1]
    vel[hit, 1] = -np.abs(vel[hit, 1])

    return pos, vel


# ============================================================
# 4. PAIRWISE DISTANCES AND NEIGHBOURS
# ============================================================

def pairwise_distances(pos, rc):
    d = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(d, axis=-1)
    neighbors = (dist > 0) & (dist <= rc)

    return d, dist, neighbors

# ============================================================
# 5. SPRING FORCES
# ============================================================

def pair_forces(pos):
    d, dist, neighbors = pairwise_distances(pos, rc)

    unit = d / np.maximum(dist[:, :, None], 1e-12)
    fmag = -k * (dist - r0) * neighbors

    return np.sum(fmag[:, :, None] * unit, axis=1)

# ============================================================
# 6. DRAG AND TOTAL FORCES
# ============================================================

def drag_forces(vel, gamma):
    return -gamma * vel

def total_forces(pos, vel, gamma):
    return pair_forces(pos) + drag_forces(vel, gamma)

# ============================================================
# 7. ENERGY CALCULATIONS
# ============================================================

def kinetic_energy(vel, mass):
    speed_squared = np.sum(vel**2, axis=1)
    return 0.5 * np.sum(mass * speed_squared)


def potential_energy(pos):
    d, dist, neighbors = pairwise_distances(pos, rc)

    # Harmonic spring potential, shifted to zero at the cutoff.
    # Each interacting pair is counted only once.
    pair_energy = (
        0.5 * k * ((dist - r0)**2 - (rc - r0)**2)
    ) * neighbors

    return 0.5 * np.sum(pair_energy)

def total_energy(pos, vel, mass):
    return kinetic_energy(vel, mass) + potential_energy(pos)

# ============================================================
# 8. NUMERICAL VERIFICATION
# ============================================================

def verify_simulation(pos):
    d, dist, neighbors = pairwise_distances(pos, rc)
    F = pair_forces(pos)

    print("\n========== VERIFICATION ==========")
    print("Distance symmetry error:",
          np.max(np.abs(dist - dist.T)))
    print("Maximum diagonal distance:",
          np.max(np.abs(np.diag(dist))))
    print("Self-neighbour count:",
          np.sum(np.diag(neighbors)))
    print("All particles inside box:",
          bool(np.all((pos >= 0) & (pos <= L))))
    print("Net internal force magnitude:",
          np.linalg.norm(np.sum(F, axis=0)))
    print("==================================\n")

verify_simulation(pos)

# ============================================================
# 9. INTEGRATION: SPRING VERLET WITH IMPLICIT LINEAR DRAG
# ============================================================


def integrate_step(pos, vel, mass, dt, gamma):

    # Calculate spring force and acceleration
    F0 = pair_forces(pos)
    a0 = (F0 - gamma * vel) / mass[:, None]

    # Predict new position
    new_pos = pos + vel * dt + 0.5 * a0 * dt**2

    # Calculate spring force at predicted position
    F1 = pair_forces(new_pos)

    # Update velocity with implicit linear drag
    numerator = vel + 0.5 * dt * (
        a0 + F1 / mass[:, None]
    )
    denominator = 1.0 + 0.5 * dt * gamma / mass[:, None]

    new_vel = numerator / denominator

    # Reflect positions and velocities together
    walls_reflect(new_pos, new_vel, L)

    pos[:] = new_pos
    vel[:] = new_vel

    return pos, vel

# ============================================================
# 10. SET UP ANIMATION AND ENERGY PLOTS
# ============================================================

plt.style.use("ggplot")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.set_xlim(0, L)
ax1.set_ylim(0, L)
ax1.set_aspect("equal")
ax1.set_title(f"Particle simulation (gamma={gamma})")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

scatter = ax1.scatter(pos[:, 0], pos[:, 1], s=35)

ax2.set_title("Energy vs simulation step")
ax2.set_xlabel("Step")
ax2.set_ylabel("Energy")
ax2.set_xlim(0, T)

line_ke, = ax2.plot([], [], label="Kinetic energy")
line_pe, = ax2.plot([], [], label="Potential energy")
line_total, = ax2.plot([], [], label="Total energy")
ax2.legend()

KE_history = []
PE_history = []
E_history = []

E_initial = total_energy(pos, vel, mass)

# ============================================================
# 11. ANIMATION UPDATE
# ============================================================

def update(frame):

    integrate_step(pos, vel, mass, dt, gamma)

    scatter.set_offsets(pos)

    KE_history.append(kinetic_energy(vel, mass))
    PE_history.append(potential_energy(pos))
    E_history.append(KE_history[-1] + PE_history[-1])

    steps = np.arange(1, len(E_history) + 1)

    line_ke.set_data(steps, KE_history)
    line_pe.set_data(steps, PE_history)
    line_total.set_data(steps, E_history)

    # Automatically adjust the vertical energy range.
    all_energy = np.concatenate([
        np.asarray(KE_history),
        np.asarray(PE_history),
        np.asarray(E_history),
        np.array([0.0])
    ])

    ymin = np.min(all_energy)
    ymax = np.max(all_energy)
    margin = max(0.1 * (ymax - ymin), 1e-6)

    ax2.set_ylim(ymin - margin, ymax + margin)

    return scatter, line_ke, line_pe, line_total

# ============================================================
# 12. RUN ANIMATION
# ============================================================

animation = FuncAnimation(
    fig,
    update,
    frames=T,
    interval=20,
    blit=False,
    repeat=False
)

plt.tight_layout()
plt.show()

# ============================================================
# 13. QUANTITATIVE RESULTS
# ============================================================

KE_history = np.asarray(KE_history)
PE_history = np.asarray(PE_history)
E_history = np.asarray(E_history)

if len(E_history) > 0:

    if abs(E_initial) > 1e-12:
        max_energy_drift = (
            np.max(np.abs(E_history - E_initial))
            / abs(E_initial)
        )
    else:
        max_energy_drift = np.nan

    print("\n========== RESULTS ==========")
    print(f"Number of particles: {N}")
    print(f"Timestep: {dt}")
    print(f"Drag coefficient: {gamma}")
    print(f"Initial total energy: {E_initial:.8f}")
    print(f"Final kinetic energy: {KE_history[-1]:.8f}")
    print(f"Final potential energy: {PE_history[-1]:.8f}")
    print(f"Final total energy: {E_history[-1]:.8f}")
    print(f"Maximum relative energy drift: {max_energy_drift:.6e}")
    print(f"Final average particle speed: "
          f"{np.mean(np.linalg.norm(vel, axis=1)):.6e}")
    print("=============================\n")
