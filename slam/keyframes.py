from slam.map import KeyFrame, Observation


def create_KeyFrame(
        global_R,
        global_t,
        kp,
        des,
        current_frame_map_points,
        keyframes,
        keyframe_id,
):
    """
    создаёт новый KeyFrame.
    ничего не возвращает.
    """

    kf = KeyFrame(
        global_R.copy(),
        global_t.copy(),
        kp,
        des
    )
    kf.id = keyframe_id

    seen_mappoint_ids = set()

    # сохраняем в текущий keyframe все точки, которые он увидел в кадре
    for mp, valid_kp_idx in current_frame_map_points:

        if mp.id in seen_mappoint_ids:
            continue

        seen_mappoint_ids.add(mp.id)

        obs = Observation(
            kf,
            valid_kp_idx
        )

        mp.observations.append(obs)

        kf.map_points.append(mp)

    keyframes.append(kf)