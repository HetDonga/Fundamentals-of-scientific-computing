
import numpy as np
import matplotlib.pyplot as plt
import statistics
import csv
from time import perf_counter
from collections import defaultdict


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

N_VALUES = [50, 200, 500, 1000]

L = 10.0
k = 10.0
r0 = 0.8
rc = 2.0

REPEATS = 7


# --------------------------------------------------
# 1. BASELINE: CHECK EVERY PAIR
# --------------------------------------------------

def force_baseline(pos):

    N = len(pos)
    forces = np.zeros((N, 2))

    pairs_checked = 0

    for i in range(N):

        for j in range(i + 1, N):

            pairs_checked += 1

            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]

            r = np.sqrt(dx*dx + dy*dy)

            if 0 < r <= rc:

                magnitude = -k * (r - r0)

                fx = magnitude * dx / r
                fy = magnitude * dy / r

                forces[i, 0] += fx
                forces[i, 1] += fy

                forces[j, 0] -= fx
                forces[j, 1] -= fy

    return forces, pairs_checked


# --------------------------------------------------
# 2. CELL-LINKED NEIGHBOR METHOD
# --------------------------------------------------

def force_cell_linked(pos):

    N = len(pos)
    forces = np.zeros((N, 2))

    # Divide box into cells
    number_of_cells = max(1, int(L // rc))
    cell_width = L / number_of_cells

    cells = defaultdict(list)

    # Assign every particle to a cell
    for i in range(N):

        cx = min(int(pos[i, 0] / cell_width),
                 number_of_cells - 1)

        cy = min(int(pos[i, 1] / cell_width),
                 number_of_cells - 1)

        cells[(cx, cy)].append(i)

    pairs_checked = 0

    # Examine the particle's own and adjacent cells
    for i in range(N):

        cx = min(int(pos[i, 0] / cell_width),
                 number_of_cells - 1)

        cy = min(int(pos[i, 1] / cell_width),
                 number_of_cells - 1)

        for ox in (-1, 0, 1):

            for oy in (-1, 0, 1):

                nx = cx + ox
                ny = cy + oy

                if not (0 <= nx < number_of_cells):
                    continue

                if not (0 <= ny < number_of_cells):
                    continue

                for j in cells[(nx, ny)]:

                    # Avoid checking the same pair twice
                    if j <= i:
                        continue

                    pairs_checked += 1

                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]

                    r = np.sqrt(dx*dx + dy*dy)

                    if 0 < r <= rc:

                        magnitude = -k * (r - r0)

                        fx = magnitude * dx / r
                        fy = magnitude * dy / r

                        forces[i, 0] += fx
                        forces[i, 1] += fy

                        forces[j, 0] -= fx
                        forces[j, 1] -= fy

    return forces, pairs_checked


# --------------------------------------------------
# 3. RUN EXPERIMENT
# --------------------------------------------------

results = []

print("\nTASK 8: CELL-LINKED NEIGHBOR SEARCH")
print("------------------------------------")

for N in N_VALUES:

    print(f"\nTesting N = {N}")

    # Same positions for both methods
    rng = np.random.default_rng(42 + N)

    pos = rng.random((N, 2)) * L

    # Calculate forces and count checked pairs
    f_baseline, baseline_pairs = force_baseline(pos)

    f_cell, cell_pairs = force_cell_linked(pos)

    # CORRECTNESS VERIFICATION
    max_error = np.max(np.abs(f_baseline - f_cell))

    relative_error = (
        np.linalg.norm(f_baseline - f_cell)
        / max(np.linalg.norm(f_baseline), 1e-12)
    )

    correct = np.allclose(
        f_baseline,
        f_cell,
        rtol=1e-9,
        atol=1e-9
    )

    print("Correctness:", "PASS" if correct else "FAIL")
    print(f"Maximum force error: {max_error:.12e}")
    print(f"Relative force error: {relative_error:.12e}")

    if not correct:
        raise ValueError("Force calculations do not match!")

    # TIMING: warm-up
    force_baseline(pos)
    force_cell_linked(pos)

    baseline_times = []
    cell_times = []

    for repeat in range(REPEATS):

        start = perf_counter()
        force_baseline(pos)
        baseline_times.append(perf_counter() - start)

        start = perf_counter()
        force_cell_linked(pos)
        cell_times.append(perf_counter() - start)

    # Median wall-clock time
    baseline_median = statistics.median(baseline_times)
    cell_median = statistics.median(cell_times)

    # PERFORMANCE METRICS
    speedup = baseline_median / cell_median

    pair_reduction = (
        (baseline_pairs - cell_pairs) / baseline_pairs
    ) * 100

    results.append({

        "N": N,
        "Baseline time (s)": baseline_median,
        "Cell time (s)": cell_median,
        "Speedup": speedup,
        "Baseline pairs": baseline_pairs,
        "Cell pairs": cell_pairs,
        "Pair reduction (%)": pair_reduction,
        "Maximum force error": max_error,
        "Relative force error": relative_error

    })


# --------------------------------------------------
# 4. PRINT RESULTS TABLE
# --------------------------------------------------

print("\n\n========== FINAL RESULTS ==========\n")

print(
    f"{'N':<8}"
    f"{'Baseline(s)':<15}"
    f"{'Cell(s)':<15}"
    f"{'Speedup':<12}"
    f"{'Pairs saved %':<15}"
)

for row in results:

    print(
        f"{row['N']:<8}"
        f"{row['Baseline time (s)']:<15.6f}"
        f"{row['Cell time (s)']:<15.6f}"
        f"{row['Speedup']:<12.2f}"
        f"{row['Pair reduction (%)']:<15.2f}"
    )


# --------------------------------------------------
# 5. SAVE CSV
# --------------------------------------------------

with open("task8_cell_results.csv", "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to task8_cell_results.csv")


# --------------------------------------------------
# 6. EXECUTION TIME GRAPH
# --------------------------------------------------

N = [row["N"] for row in results]

baseline_times = [
    row["Baseline time (s)"] for row in results
]

cell_times = [
    row["Cell time (s)"] for row in results
]

plt.figure(figsize=(9, 6))

plt.plot(N, baseline_times, marker="o",
         label="Baseline all-pairs")

plt.plot(N, cell_times, marker="o",
         label="Cell-linked")

plt.xlabel("Number of particles (N)")
plt.ylabel("Median execution time (seconds)")
plt.title("Execution Time Comparison")

plt.xticks(N)
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("task8_cell_timing.png", dpi=300)
plt.show()


# --------------------------------------------------
# 7. SPEEDUP GRAPH
# --------------------------------------------------

speedups = [row["Speedup"] for row in results]

plt.figure(figsize=(9, 6))

plt.plot(N, speedups, marker="o")
plt.axhline(y=1, linestyle="--")

plt.xlabel("Number of particles (N)")
plt.ylabel("Speedup")
plt.title("Cell-Linked Method Speedup")

plt.xticks(N)
plt.grid()
plt.tight_layout()

plt.savefig("task8_cell_speedup.png", dpi=300)
plt.show()


# --------------------------------------------------
# 8. PAIR REDUCTION GRAPH
# --------------------------------------------------

reductions = [
    row["Pair reduction (%)"] for row in results
]

plt.figure(figsize=(9, 6))

plt.plot(N, reductions, marker="o")

plt.xlabel("Number of particles (N)")
plt.ylabel("Reduction in pair checks (%)")
plt.title("Reduction in Distance Calculations")

plt.xticks(N)
plt.grid()
plt.tight_layout()

plt.savefig("task8_cell_pair_reduction.png", dpi=300)
plt.show()


print("\nAll Task 8 experiments completed!")
