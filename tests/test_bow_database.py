import tempfile
from pathlib import Path

import numpy as np

from slam.bow import BoWDatabase


# =================================================================================================
# Simple fake KeyFrame for testing
# =================================================================================================

class FakeKeyFrame:

    def __init__(
        self,
        keyframe_id,
        descriptors
    ):
        self.id = keyframe_id
        self.des = descriptors


# =================================================================================================
# Test data
# =================================================================================================

rng = np.random.default_rng(
    123
)

NUM_TRAINING_IMAGES = 25
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
# Create BoW database
# =================================================================================================

bow_database = BoWDatabase()

bow_database.create_vocabulary(
    training_sets,
    branching_factor=6,
    depth=4
)


print(
    "Initial stats:",
    bow_database.get_stats()
)


assert bow_database.vocabulary_size() > 0

assert bow_database.size() == 0


# =================================================================================================
# Prepare several KeyFrames
# =================================================================================================

reference_descriptors = training_sets[0].copy()

similar_descriptors = reference_descriptors.copy()

similar_descriptors[:40] = rng.integers(
    0,
    256,
    size=(
        40,
        ORB_DESCRIPTOR_BYTES
    ),
    dtype=np.uint8
)

different_descriptors = training_sets[10].copy()


kf_10 = FakeKeyFrame(
    10,
    reference_descriptors
)

kf_25 = FakeKeyFrame(
    25,
    different_descriptors
)

kf_100 = FakeKeyFrame(
    100,
    similar_descriptors
)


# =================================================================================================
# Add KeyFrames
# =================================================================================================

entry_10 = bow_database.add_keyframe(
    kf_10
)

entry_25 = bow_database.add_keyframe(
    kf_25
)

entry_100 = bow_database.add_keyframe(
    kf_100
)


print(
    "Entry mapping:"
)

print(
    f"  KF 10  -> entry {entry_10}"
)

print(
    f"  KF 25  -> entry {entry_25}"
)

print(
    f"  KF 100 -> entry {entry_100}"
)


assert bow_database.size() == 3

assert bow_database.contains_keyframe(
    10
)

assert bow_database.contains_keyframe(
    25
)

assert bow_database.contains_keyframe(
    100
)


# =================================================================================================
# Verify ID mapping
# =================================================================================================

assert bow_database.get_entry_id(
    10
) == entry_10

assert bow_database.get_entry_id(
    25
) == entry_25

assert bow_database.get_entry_id(
    100
) == entry_100


assert bow_database.get_keyframe_id(
    entry_10
) == 10

assert bow_database.get_keyframe_id(
    entry_25
) == 25

assert bow_database.get_keyframe_id(
    entry_100
) == 100


# =================================================================================================
# Similarity test
# =================================================================================================

score_same = bow_database.score(
    reference_descriptors,
    reference_descriptors
)

score_similar = bow_database.score(
    reference_descriptors,
    similar_descriptors
)

score_different = bow_database.score(
    reference_descriptors,
    different_descriptors
)


print()
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
# Query test
# =================================================================================================

results = bow_database.query_keyframe(
    kf_10,
    max_results=3
)


print()
print(
    "Query results:"
)

for result in results:

    print(
        f"  keyframe_id={result.keyframe_id}, "
        f"entry_id={result.entry_id}, "
        f"score={result.score:.6f}"
    )


assert len(results) > 0

assert results[0].keyframe_id == 10

assert results[0].entry_id == entry_10


# =================================================================================================
# Duplicate protection
# =================================================================================================

duplicate_rejected = False

try:

    bow_database.add_keyframe(
        kf_10
    )

except ValueError:

    duplicate_rejected = True


assert duplicate_rejected


print()
print(
    "Duplicate KeyFrame protection: PASSED"
)


# =================================================================================================
# Vocabulary save/load test
# =================================================================================================

with tempfile.TemporaryDirectory() as temp_dir:

    vocabulary_path = (
        Path(temp_dir)
        / "test_vocabulary.yml.gz"
    )

    bow_database.save_vocabulary(
        vocabulary_path
    )

    assert vocabulary_path.exists()

    loaded_database = BoWDatabase()

    loaded_database.load_vocabulary(
        vocabulary_path
    )

    assert (
        loaded_database.vocabulary_size()
        ==
        bow_database.vocabulary_size()
    )

    assert loaded_database.size() == 0

    loaded_database.add_keyframe(
        kf_10
    )

    loaded_results = loaded_database.query_keyframe(
        kf_10,
        max_results=1
    )

    assert len(loaded_results) == 1

    assert loaded_results[0].keyframe_id == 10


print(
    "Vocabulary save/load: PASSED"
)


# =================================================================================================
# Final stats
# =================================================================================================

stats = bow_database.get_stats()


print()
print(
    "Final stats:",
    stats
)


assert stats[
    "database_size"
] == 3

assert stats[
    "mapped_entries"
] == 3

assert stats[
    "mapped_keyframes"
] == 3


print()
print(
    "BoWDatabase test PASSED."
)