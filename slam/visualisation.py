import cv2 as cv
import numpy as np


def create_trajectory_image(
    size=800,
    origin=(400, 400)
):
    """
    Создаёт изображение для 2D-визуализации траектории камеры.
    """

    traj_img = np.zeros(
        (size, size, 3),
        dtype=np.uint8
    )

    cv.circle(
        traj_img,
        origin,
        5,
        (0, 0, 255),
        -1
    )

    return traj_img


def create_map_image(
    size=800
):
    """
    Создаёт изображение для 2D-визуализации MapPoints.
    """

    map_img = np.zeros(
        (size, size, 3),
        dtype=np.uint8
    )

    return map_img


def draw_camera_trajectory(
    traj_img,
    global_t,
    origin=(400, 400),
    color=(0, 255, 0)
):
    """
    Рисует текущую позицию камеры на trajectory image.
    """

    x_cam = int(global_t[0, 0]) + origin[0]
    z_cam = int(global_t[2, 0]) + origin[1]

    cv.circle(
        traj_img,
        (x_cam, z_cam),
        2,
        color,
        -1
    )


def draw_new_mappoints(
    map_img,
    map_points,
    last_drawn_map_point_idx,
    origin=(400, 400),
    color=(255, 255, 255)
):
    """
    Рисует только новые MapPoints, которые ещё не были нарисованы.
    Возвращает обновлённый last_drawn_map_point_idx.
    """

    for p in map_points[last_drawn_map_point_idx:]:

        map_point_x = int(p.position[0]) + origin[0]
        map_point_z = int(p.position[2]) + origin[1]

        cv.circle(
            map_img,
            (map_point_x, map_point_z),
            1,
            color,
            -1
        )

    return len(map_points)


def draw_debug_overlay(
    draw_img,
    matches,
    pose_source,
    local_map_points,
    tracking_stats
):
    """
    Рисует debug-информацию поверх feature matching изображения.
    """

    debug_lines = [
        f"Matches: {len(matches)}",
        f"Pose: {pose_source}",
        f"Local MPs: {len(local_map_points)}",
        f"Tracked: {tracking_stats['tracked_used']}/{tracking_stats['tracked_before_pnp']}",
        f"Projected/Inside: {tracking_stats['projected']}/{tracking_stats['inside']}",
        f"PnP inliers: {tracking_stats['pnp_inliers']}/{tracking_stats['pnp_points']}",
    ]

    x = 10
    y_start = 30
    line_height = 35

    for i, text in enumerate(debug_lines):

        y = y_start + i * line_height

        cv.putText(
            draw_img,
            text,
            (x, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


def draw_feature_matching_view(
    prev_gray,
    prev_kp,
    gray,
    kp,
    matches,
    pose_source,
    local_map_points,
    tracking_stats,
    max_matches_to_draw=50
):
    """
    Создаёт изображение feature matching и добавляет debug overlay.
    """

    draw_img = cv.drawMatches(
        prev_gray,
        prev_kp,
        gray,
        kp,
        matches[:max_matches_to_draw],
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    draw_debug_overlay(
        draw_img,
        matches,
        pose_source,
        local_map_points,
        tracking_stats
    )

    return draw_img


def show_slam_windows(
    draw_img,
    traj_img,
    map_img
):
    """
    Показывает основные окна SLAM pipeline.
    """

    cv.imshow(
        "Feature Matching",
        draw_img
    )

    cv.imshow(
        "Trajectory",
        traj_img
    )

    cv.imshow(
        "Point Cloud",
        map_img
    )