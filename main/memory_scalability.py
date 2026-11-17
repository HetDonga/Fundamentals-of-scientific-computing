
import numpy as np
import matplotlib.pyplot as plt

# 1. PARAMETERS

N_VALUES = [100, 1000, 5000, 10000]

DTYPES = [np.float32, np.float64]

# 2. MEMORY CALCULATION

def theoretical_memory(N, dtype):
    itemsize = np.dtype(dtype).itemsize
    return N**2 * 2 * itemsize


def format_memory(byte_count):
    if byte_count >= 1024**3:
        return f"{byte_count / 1024**3:.2f} GiB"
    elif byte_count >= 1024**2:
        return f"{byte_count / 1024**2:.2f} MiB"
    else:
        return f"{byte_count / 1024:.2f} KiB"


# 3. ACTUAL NUMPY MEMORY

def actual_numpy_memory(N, dtype):
    displacement = np.empty((N, N, 2), dtype=dtype)
    memory = displacement.nbytes
    del displacement
    return memory

# 4. PRINT MEMORY TABLE

def main():

    results = {}

    print("\n========== TASK 7: MEMORY SCALABILITY ==========\n")

    print(
        f"{'N':>8} {'float32':>18} "
        f"{'float64':>18} {'Memory ratio':>16}"
    )

    print("-" * 65)

    for N in N_VALUES:

        memory32 = theoretical_memory(N, np.float32)
        memory64 = theoretical_memory(N, np.float64)

        results[N] = {
            "float32": memory32,
            "float64": memory64
        }

        print(
            f"{N:8d} "
            f"{format_memory(memory32):>18} "
            f"{format_memory(memory64):>18} "
            f"{memory64 / memory32:15.2f}x"
        )

    # 5. VERIFY WITH ACTUAL NUMPY NBYTES

    print("\n========== NUMPY NBYTES VERIFICATION ==========\n")

    for N in [100, 1000]:

        for dtype in DTYPES:

            theoretical = theoretical_memory(N, dtype)
            actual = actual_numpy_memory(N, dtype)

            print(
                f"N={N:5d} | "
                f"{np.dtype(dtype).name:7s} | "
                f"Theoretical: {theoretical:,} bytes | "
                f"Actual: {actual:,} bytes | "
                f"Match: {theoretical == actual}"
            )

    # 6. MEMORY GRAPH

    memory32 = [
        results[N]["float32"] / (1024**2)
        for N in N_VALUES
    ]

    memory64 = [
        results[N]["float64"] / (1024**2)
        for N in N_VALUES
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        N_VALUES,
        memory32,
        marker="o",
        linewidth=2,
        markersize=8,
        label="float32"
    )

    plt.plot(
        N_VALUES,
        memory64,
        marker="s",
        linewidth=2,
        markersize=8,
        label="float64"
    )

    # Add labels at every point.
    for i, N in enumerate(N_VALUES):

        plt.annotate(
            f"{memory32[i]:.2f} MiB",
            (N, memory32[i]),
            xytext=(8, -16),
            textcoords="offset points",
            fontsize=9
        )

        plt.annotate(
            f"{memory64[i]:.2f} MiB",
            (N, memory64[i]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9
        )

    # 1 GiB reference line.
    plt.axhline(
        1024,
        linestyle="--",
        linewidth=1.5,
        label="1 GiB reference"
    )

    plt.xlabel("Number of particles (N)")
    plt.ylabel("Displacement array memory (MiB)")

    plt.title(
        "Memory Scalability of Fully Vectorized Pairwise Calculation"
    )

    plt.yscale("log")
    plt.xticks(N_VALUES)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig("task7_memory_scalability.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
