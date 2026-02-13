import itertools
import random
import re


def shuffle_substrings(input_str: str) -> list[str]:
    # Ensure only alphanumeric, dash, or underscore characters
    assert re.fullmatch(
        r"[a-zA-Z0-9_+\-]+", input_str
    ), "Invalid character in string"

    # If no special characters, return input as is
    if not any(c in input_str for c in "-_+"):
        return [input_str]

    # Split input on any allowed delimiter
    substrings = re.split(r"[-_+]", input_str)

    # Generate all permutations of substrings
    perms = list(itertools.permutations(substrings))

    # Generate all possible delimiter combinations of length (n - 1)
    delimiters = ["-", "_", "+"]
    all_results = []
    for perm in perms:
        if len(perm) == 1:
            all_results.append(perm[0])
            continue
        for delim_combo in itertools.product(delimiters, repeat=len(perm) - 1):
            combined = "".join(
                [a + b for a, b in zip(perm, delim_combo + ("",))]
            )
            all_results.append(combined)

    return all_results
