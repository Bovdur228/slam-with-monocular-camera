import cv2 as cv
import numpy as np

from slam.map import MapPoint, Observation
from slam.tracking import project_point
from slam.triangulation import triangulate_points_world_from_poses


def get_keyframe_used_keypoint_indices(kf):
    """
    Возвращает индексы keypoints, которые уже связаны с MapPoints в данном KeyFrame.
    """

    used_keypoint_indices = set()

    for mp in kf.map_points:

        for obs in mp.observations:

            if obs.keyframe is kf:
                used_keypoint_indices.add(obs.keypoint_idx)

    return used_keypoint_indices

# --------------------------------------------------------------------------------------------------------------------------------------------
def compute_two_view_reprojection_error(
    point_world,
    reference_kf,
    current_kf,
    reference_kp_idx,
    current_kp_idx,
    K
):
    """
    Считает среднюю reprojection error новой 3D-точки
    в reference KeyFrame и current KeyFrame.
    """

    reference_uv = project_point(
        point_world,
        reference_kf.R,
        reference_kf.t,
        K
    )

    current_uv = project_point(
        point_world,
        current_kf.R,
        current_kf.t,
        K
    )

    if reference_uv is None or current_uv is None:
        return None

    reference_observed_uv = np.array(
        reference_kf.kp[reference_kp_idx].pt,
        dtype=np.float64
    )

    current_observed_uv = np.array(
        current_kf.kp[current_kp_idx].pt,
        dtype=np.float64
    )

    reference_error = np.linalg.norm(
        reference_uv - reference_observed_uv
    )

    current_error = np.linalg.norm(
        current_uv - current_observed_uv
    )

    mean_error = 0.5 * (
        reference_error + current_error
    )

    return mean_error

# --------------------------------------------------------------------------------------------------------------------------------------------
def triangulate_new_mappoints_between_keyframes(
    reference_kf,
    current_kf,
    map_points,
    map_point_id,
    K,
    bf,
    ransac_threshold=1.0,
    min_ransac_inliers=20,
    max_depth=100,
    max_descriptor_distance=50,
    max_reprojection_error=10.0,
    max_new_points=500
):
    """
    Создаёт новые MapPoints только между двумя KeyFrames.

    Возвращает:
        updated_map_point_id
        stats
    """

    stats = {
        "matches": 0,
        "candidate_matches": 0,
        "triangulated": 0,
        "created": 0,
        "skipped_used_keypoint": 0,
        "skipped_descriptor_distance": 0,
        "skipped_reprojection": 0,
        "skipped_invalid_triangulation": 0
    }

    if reference_kf is None or current_kf is None:
        return map_point_id, stats

    if reference_kf.des is None or current_kf.des is None:
        return map_point_id, stats

    matches = bf.match(
        reference_kf.des,
        current_kf.des
    )

    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    stats["matches"] = len(matches)

    reference_used_kp_idxes = get_keyframe_used_keypoint_indices(
        reference_kf
    )

    current_used_kp_idxes = get_keyframe_used_keypoint_indices(
        current_kf
    )

    pts_reference = []
    pts_current = []
    reference_kp_idxes = []
    current_kp_idxes = []
    descriptors = []

    for match in matches:

        if match.distance > max_descriptor_distance:
            stats["skipped_descriptor_distance"] += 1
            continue

        if match.queryIdx in reference_used_kp_idxes:
            stats["skipped_used_keypoint"] += 1
            continue

        if match.trainIdx in current_used_kp_idxes:
            stats["skipped_used_keypoint"] += 1
            continue

        pts_reference.append(
            reference_kf.kp[match.queryIdx].pt
        )

        pts_current.append(
            current_kf.kp[match.trainIdx].pt
        )

        reference_kp_idxes.append(
            match.queryIdx
        )

        current_kp_idxes.append(
            match.trainIdx
        )

        descriptors.append(
            current_kf.des[match.trainIdx]
        )

        if len(pts_reference) >= max_new_points:
            break

    stats["candidate_matches"] = len(pts_reference)

    if len(pts_reference) < 8:
        return map_point_id, stats

    pts_reference = np.array(
        pts_reference,
        dtype=np.float32
    )

    pts_current = np.array(
        pts_current,
        dtype=np.float32
    )

    E, mask = cv.findEssentialMat(
        pts_reference,
        pts_current,
        K,
        method=cv.RANSAC,
        prob=0.999,
        threshold=ransac_threshold
    )

    if E is None or mask is None:
        stats["skipped_geometric_ransac"] = stats.get("skipped_geometric_ransac", 0) + len(pts_reference)
        return map_point_id, stats

    ransac_mask = mask.ravel() == 1

    if np.count_nonzero(ransac_mask) < min_ransac_inliers:
        stats["skipped_geometric_ransac"] = stats.get("skipped_geometric_ransac", 0) + len(pts_reference)
        return map_point_id, stats

    pts_reference = pts_reference[ransac_mask]
    pts_current = pts_current[ransac_mask]

    reference_kp_idxes = np.array(reference_kp_idxes)[ransac_mask]
    current_kp_idxes = np.array(current_kp_idxes)[ransac_mask]
    descriptors = np.array(descriptors)[ransac_mask]

    stats["ransac_inliers"] = int(np.count_nonzero(ransac_mask))

    points_world, valid_mask = triangulate_points_world_from_poses(
        pts_reference,
        pts_current,
        reference_kf.R,
        reference_kf.t,
        current_kf.R,
        current_kf.t,
        K,
        max_depth=max_depth
    )

    valid_reference_kp_idxes = np.array(
        reference_kp_idxes
    )[valid_mask]

    valid_current_kp_idxes = np.array(
        current_kp_idxes
    )[valid_mask]

    valid_descriptors = np.array(
        descriptors
    )[valid_mask]

    stats["triangulated"] = len(points_world)

    for point_world, reference_kp_idx, current_kp_idx, descriptor in zip(
        points_world,
        valid_reference_kp_idxes,
        valid_current_kp_idxes,
        valid_descriptors
    ):

        if reference_kp_idx in reference_used_kp_idxes:
            stats["skipped_used_keypoint"] += 1
            continue

        if current_kp_idx in current_used_kp_idxes:
            stats["skipped_used_keypoint"] += 1
            continue

        reprojection_error = compute_two_view_reprojection_error(
            point_world,
            reference_kf,
            current_kf,
            reference_kp_idx,
            current_kp_idx,
            K
        )

        if reprojection_error is None:
            stats["skipped_invalid_triangulation"] += 1
            continue

        if reprojection_error > max_reprojection_error:
            stats["skipped_reprojection"] += 1
            continue

        mp = MapPoint(
            point_world,
            descriptor,
            created_keyframe_id=current_kf.id
        )

        mp.id = map_point_id
        map_point_id += 1

        mp.observations.append(
            Observation(
                reference_kf,
                int(reference_kp_idx)
            )
        )

        mp.observations.append(
            Observation(
                current_kf,
                int(current_kp_idx)
            )
        )

        mp.num_observations = len(mp.observations)

        map_points.append(mp)

        reference_kf.map_points.append(mp)
        current_kf.map_points.append(mp)

        reference_used_kp_idxes.add(
            int(reference_kp_idx)
        )

        current_used_kp_idxes.add(
            int(current_kp_idx)
        )

        stats["created"] += 1

    return map_point_id, stats