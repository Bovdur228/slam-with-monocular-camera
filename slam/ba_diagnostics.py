import numpy as np

from collections import Counter

from slam.ba import (
    collect_ba_observations,
    collect_and_filter_observations,
)


def print_count_distribution(
    counts,
    title
):
    """
    Печатает распределение количества наблюдений.

    counts — список чисел, например:
        [len(mp.observations) for mp in map_points]

    Используется для диагностики:
        - сколько MapPoints имеют 0 observations
        - сколько имеют 1 observation
        - сколько имеют 2
        - сколько имеют 3+
    """

    print("-" * 100)
    print(title)

    if len(counts) == 0:
        print("No items.")
        return

    counter = Counter(counts)

    total = len(counts)

    zero = counter.get(0, 0)
    one = counter.get(1, 0)
    two = counter.get(2, 0)
    three = counter.get(3, 0)

    four_to_five = sum(
        value
        for count, value in counter.items()
        if 4 <= count <= 5
    )

    six_to_ten = sum(
        value
        for count, value in counter.items()
        if 6 <= count <= 10
    )

    more_than_ten = sum(
        value
        for count, value in counter.items()
        if count > 10
    )

    print(f"Total items:        {total}")
    print(f"Mean count:         {np.mean(counts):.2f}")
    print(f"Median count:       {np.median(counts):.2f}")
    print(f"Max count:          {np.max(counts)}")

    print("-" * 100)

    print(f"0 observations:     {zero}")
    print(f"1 observation:      {one}")
    print(f"2 observations:     {two}")
    print(f"3 observations:     {three}")
    print(f"4-5 observations:   {four_to_five}")
    print(f"6-10 observations:  {six_to_ten}")
    print(f">10 observations:   {more_than_ten}")

# ---------------------------------------------------------------------------------------------------------------------------------
def print_ba_graph_diagnostics(
    map_points,
    keyframes,
    K,
    max_initial_residual=50.0,
    title="BA Graph Diagnostics"
):
    """
    Диагностирует качество BA-графа.

    Показывает:
        1. Сколько raw observations есть у MapPoints.
        2. Сколько observations приходится на KeyFrames.
        3. Сколько observations проходят projection check.
        4. Сколько observations проходят reprojection error filter.
        5. Распределение clean observations по MapPoints и KeyFrames.
    """

    print("=" * 100)
    print(title)
    print("=" * 100)

    print(f"MapPoints: {len(map_points)}")
    print(f"KeyFrames: {len(keyframes)}")

    # ------------------------------------------------------------------
    # 1. Raw observations per MapPoint
    # ------------------------------------------------------------------

    raw_mp_counts = [
        len(mp.observations)
        for mp in map_points
    ]

    print_count_distribution(
        raw_mp_counts,
        "Raw observations per MapPoint"
    )

    # ------------------------------------------------------------------
    # 2. Raw observations per selected KeyFrame
    # ------------------------------------------------------------------

    keyframe_ids = {
        kf.id
        for kf in keyframes
    }

    raw_kf_counts = {
        kf.id: 0
        for kf in keyframes
    }

    for mp in map_points:
        for obs in mp.observations:

            kf_id = obs.keyframe.id

            if kf_id in keyframe_ids:
                raw_kf_counts[kf_id] += 1

    print_count_distribution(
        list(raw_kf_counts.values()),
        "Raw observations per selected KeyFrame"
    )

    # ------------------------------------------------------------------
    # 3. Observations after projection check
    # ------------------------------------------------------------------

    ba_observations = collect_ba_observations(
        map_points,
        K
    )

    print("-" * 100)
    print(f"collect_ba_observations: {len(ba_observations)}")

    # ------------------------------------------------------------------
    # 4. Observations after reprojection error filter
    # ------------------------------------------------------------------

    clean_observations = collect_and_filter_observations(
        map_points,
        K,
        max_initial_residual
    )

    print(
        f"collect_and_filter_observations "
        f"(max residual {max_initial_residual}): "
        f"{len(clean_observations)}"
    )

    # ------------------------------------------------------------------
    # 5. Clean observations per MapPoint
    # ------------------------------------------------------------------

    clean_mp_counts = {
        mp.id: 0
        for mp in map_points
    }

    for mp_id, _, _ in clean_observations:
        if mp_id in clean_mp_counts:
            clean_mp_counts[mp_id] += 1

    print_count_distribution(
        list(clean_mp_counts.values()),
        "Clean BA observations per MapPoint"
    )

    # ------------------------------------------------------------------
    # 6. Clean observations per selected KeyFrame
    # ------------------------------------------------------------------

    clean_kf_counts = {
        kf.id: 0
        for kf in keyframes
    }

    for _, kf_id, _ in clean_observations:
        if kf_id in clean_kf_counts:
            clean_kf_counts[kf_id] += 1

    print_count_distribution(
        list(clean_kf_counts.values()),
        "Clean BA observations per selected KeyFrame"
    )

    print("=" * 100)