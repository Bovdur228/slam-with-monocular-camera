import os
import sys
from pathlib import Path

import numpy as np


# =================================================================================================
# Paths
# =================================================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BINDINGS_DIR = (
    PROJECT_ROOT
    / "cpp"
    / "build"
    / "Release"
)

DBOW3_DLL_DIR = Path(
    r"D:\work\DBow3\build\bin\Release"
)

VCPKG_DLL_DIR = Path(
    r"D:\work\vcpkg\installed\x64-windows\bin"
)


# =================================================================================================
# Make compiled module and DLL dependencies visible to Python
# =================================================================================================

sys.path.insert(
    0,
    str(BINDINGS_DIR)
)

if os.name == "nt":
    os.add_dll_directory(
        str(DBOW3_DLL_DIR)
    )

    os.add_dll_directory(
        str(VCPKG_DLL_DIR)
    )


import dbow3_bindings


# =================================================================================================
# Test data
# =================================================================================================

rng = np.random.default_rng(
    42
)

NUM_TRAINING_IMAGES = 20
DESCRIPTORS_PER_IMAGE = 300
ORB_DESCRIPTOR_BYTES = 32


training_sets = []

for _ in range(NUM_TRAINING_IMAGES):

    descriptors = rng.integers(
        0,
        256,
        size=(
            DESCRIPTORS_PER_IMAGE,
            ORB_DESCRIPTOR_BYTES
        ),
        dtype=np.uint8
    )

    training_sets.append(
        descriptors
    )


# =================================================================================================
# Create vocabulary
# =================================================================================================

vocabulary = dbow3_bindings.Vocabulary()

vocabulary.create(
    training_sets,
    branching_factor=6,
    depth=4
)

print(
    f"Vocabulary size: {vocabulary.size()}"
)

assert vocabulary.size() > 0


# =================================================================================================
# Create descriptors for similarity test
# =================================================================================================

reference_descriptors = training_sets[0].copy()

similar_descriptors = reference_descriptors.copy()

# Немного портим часть descriptors,
# сохраняя большую часть изображения похожей.
num_modified = 40

similar_descriptors[:num_modified] = rng.integers(
    0,
    256,
    size=(
        num_modified,
        ORB_DESCRIPTOR_BYTES
    ),
    dtype=np.uint8
)

different_descriptors = rng.integers(
    0,
    256,
    size=(
        DESCRIPTORS_PER_IMAGE,
        ORB_DESCRIPTOR_BYTES
    ),
    dtype=np.uint8
)


# =================================================================================================
# Vocabulary score test
# =================================================================================================

score_same = vocabulary.score(
    reference_descriptors,
    reference_descriptors
)

score_similar = vocabulary.score(
    reference_descriptors,
    similar_descriptors
)

score_different = vocabulary.score(
    reference_descriptors,
    different_descriptors
)


print(
    f"Score same:      {score_same:.6f}"
)

print(
    f"Score similar:   {score_similar:.6f}"
)

print(
    f"Score different: {score_different:.6f}"
)


assert score_same >= score_similar
assert score_similar >= score_different


# =================================================================================================
# Database test
# =================================================================================================

database = dbow3_bindings.Database(
    vocabulary
)


entry_reference = database.add(
    reference_descriptors
)

entry_different_1 = database.add(
    training_sets[5]
)

entry_different_2 = database.add(
    training_sets[10]
)

entry_similar = database.add(
    similar_descriptors
)


print(
    f"Database size: {database.size()}"
)

print(
    "Entry IDs:",
    entry_reference,
    entry_different_1,
    entry_different_2,
    entry_similar
)


assert database.size() == 4


# =================================================================================================
# Query test
# =================================================================================================

results = database.query(
    reference_descriptors,
    max_results=4
)


print(
    "Query results:"
)

for entry_id, score in results:

    print(
        f"  entry_id={entry_id}, "
        f"score={score:.6f}"
    )


assert len(results) > 0

best_entry_id, best_score = results[0]

assert best_entry_id == entry_reference


print()
print(
    "DBoW3 Python bindings test PASSED."
)