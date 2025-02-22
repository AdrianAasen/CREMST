import numpy as np
import time
from itertools import combinations


def generate_random_hash(n_qubits, k_hash_symbols):
    """
    Generate a single random hash (function) mapping n_qubits to [0..k_hash_symbols-1].
    """
    return np.random.randint(low=0, high=k_hash_symbols, size=n_qubits)

def is_hash_family_perfect(hash_list, k_hash_symbols):
    """
    Checks if a collection of hash functions (hash_list) forms a perfect hash family
    for subsets of size k_hash_symbols.
    """
    n_qubits = hash_list.shape[1]

    # Generate all subsets of size k_hash_symbols
    subset_indices = list(combinations(range(n_qubits), k_hash_symbols))

    # For each subset of size k_hash_symbols, we need at least one hash function
    # that assigns distinct labels to those elements in the subset.
    for subset in subset_indices:
        found_distinguishing_hash = False
        for hash_func in hash_list:
            labels = hash_func[list(subset)]
            # If these k qubits get k distinct labels in 'hash_func':
            if len(np.unique(labels)) == k_hash_symbols:
                found_distinguishing_hash = True
                break
        if not found_distinguishing_hash:
            return False
    return True

def generate_perfect_hash_family(
    n_qubits, k_hash_symbols, max_iter=10_000, limit_size=None
):
    """
    Brute-force algorithm to create a perfect hash family for 'n_qubits' such that
    every k_hash_symbols-subset is distinctly mapped. Stops early if we exceed 'limit_size',
    which can be the size of the best known family so far.

    Parameters
    ----------
    n_qubits : int
        Number of elements (qubits) to hash.
    k_hash_symbols : int
        Size of subsets (t) that must be collision-free.
    max_iter : int
        Safety cap to avoid infinite loops.
    limit_size : int or None
        If not None, stop the search early when the current family's size
        exceeds 'limit_size'.

    Returns
    -------
    np.ndarray of shape (m, n_qubits)
        The (partial) hash family. Could be perfect or partial if we stopped early.
    """
    # Start with a simple mod-hash as the first function
    hash_list = [np.arange(n_qubits) % k_hash_symbols]

    iteration = 0
    while iteration < max_iter:
        # If the current family is already larger than limit_size (best known), stop
        if limit_size is not None and len(hash_list) > limit_size:
            print(f"Stopping early, as we exceeded the best known family size = {limit_size}.")
            break

        # Check if the current family is perfect
        if is_hash_family_perfect(np.array(hash_list), k_hash_symbols):
            break

        # Otherwise, generate and add a new random hash
        new_hash = generate_random_hash(n_qubits, k_hash_symbols)
        hash_list.append(new_hash)
        iteration += 1

    # Final check (in case we exited by iteration or limit_size)
    if not is_hash_family_perfect(np.array(hash_list), k_hash_symbols):
        print(f"Warning: Not perfect after {iteration} iterations (limit_size={limit_size}). Returning None.")
        return None

    return np.array(hash_list)

def generate_best_perfect_hash_family(
    n_qubits, k_hash_symbols, num_runs=5, max_iter=10_000
):
    """
    Outer loop: repeats perfect-hash generation multiple times, returning
    the family with the fewest hash functions. Also passes in the current
    best size as 'limit_size' to skip wasted effort in the inner loop.

    Parameters
    ----------
    n_qubits : int
        Number of qubits/elements.
    k_hash_symbols : int
        Subset size t we need to distinguish.
    num_runs : int
        How many times to attempt generation.
    max_iter : int
        Safety cap for each generation call.

    Returns
    -------
    best_family : np.ndarray
        The best (smallest) family found across all runs.
    """
    best_family = None

    for run_index in range(num_runs):
        print(f"\n--- Run {run_index + 1} / {num_runs} ---")
        # Pass in the current best family's size as limit_size:
        current_limit = best_family.shape[0] if best_family is not None else None

        candidate_family = generate_perfect_hash_family(
            n_qubits, k_hash_symbols,
            max_iter=max_iter,
            limit_size=current_limit
        )

        # Compare lengths:
        if candidate_family is None:
            pass           # print("No family better than previous best family found.")
        elif best_family is None or  candidate_family.shape[0] < best_family.shape[0]:
            best_family = candidate_family
            print(f"New best family found with {best_family.shape[0]} functions.")
        else:
            print(f"Family of size {candidate_family.shape[0]} is not better than current best ({best_family.shape[0]}).")

    return best_family

if __name__ == "__main__":
    n_qubits = 16 # Number of qubits
    k_hash_symbols = 6 # Size of subsets to distinguish
    timeout_start = time.time()
    num_runs = 1000
    family = generate_best_perfect_hash_family(n_qubits, k_hash_symbols, num_runs = num_runs, max_iter=500)
    size=len(family)
    print(f"Time spent: {time.time() - timeout_start}.")
    print(f"Best hash found: {len(family)}")
    path = "EMQST_lib/hash_family/"
    print(family)
    with open(f'{path}perfect_hash({size},{n_qubits},{k_hash_symbols}).npy', 'wb') as f:
        np.save(f, family)
    

