import cv2 as cv
import numpy as np


def projection_matrix_from_pose(
    Rwc,
    twc,
    K
):
    """
    Создаёт projection matrix P = K @ [Rcw | tcw]
    из нашей pose convention Rwc/twc.

    У нас:
        Pw = Rwc @ Pc + twc

    Для projection нужно:
        Pc = Rcw @ Pw + tcw
    """

    Rcw = Rwc.T
    tcw = -Rcw @ twc

    P = K @ np.hstack(
        (
            Rcw,
            tcw
        )
    )

    return P

# -----------------------------------------------------------------------------------------------------------------------------------------
def triangulate_points_world_from_poses(
    pts1,
    pts2,
    prev_Rwc,
    prev_twc,
    curr_Rwc,
    curr_twc,
    K,
    max_depth=100
):
    """
    Триангулирует точки сразу в world coordinates,
    используя global pose предыдущего и текущего кадра.

    pts1 — points на предыдущем кадре
    pts2 — points на текущем кадре
    """

    P1 = projection_matrix_from_pose(
        prev_Rwc,
        prev_twc,
        K
    )

    P2 = projection_matrix_from_pose(
        curr_Rwc,
        curr_twc,
        K
    )

    points_4d = cv.triangulatePoints(
        P1,
        P2,
        pts1.T,
        pts2.T
    )

    w = points_4d[3]

    valid_w_mask = np.abs(w) > 1e-8

    points_world = np.zeros(
        (points_4d.shape[1], 3),
        dtype=np.float64
    )

    points_world[valid_w_mask] = (
        points_4d[:3, valid_w_mask] / w[valid_w_mask]
    ).T

    points_cam_prev = (
        prev_Rwc.T @ (points_world.T - prev_twc)
    ).T

    points_cam_curr = (
        curr_Rwc.T @ (points_world.T - curr_twc)
    ).T

    valid_mask = (
        valid_w_mask
        &
        np.isfinite(points_world).all(axis=1)
        &
        np.isfinite(points_cam_prev).all(axis=1)
        &
        np.isfinite(points_cam_curr).all(axis=1)
        &
        (points_cam_prev[:, 2] > 0)
        &
        (points_cam_curr[:, 2] > 0)
        &
        (points_cam_prev[:, 2] < max_depth)
        &
        (points_cam_curr[:, 2] < max_depth)
    )

    return points_world[valid_mask], valid_mask