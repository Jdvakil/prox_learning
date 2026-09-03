#!/usr/bin/env python3
"""v12 copy of the table/wrist camera helpers."""

from __future__ import annotations

import numpy as np

THIRD_PERSON_FOV = 58.0
WRIST_FOV = 56.74
WRIST_CAMERA_MJCF = "robot_0/gripper/wrist_camera"
CAMERA_REFERENCE_BODY = "robot_0/fr3_link0"
CAMERA_OFFSET = np.asarray([-1.05, -0.55, 1.30], dtype=np.float64)
LOOKAT_OFFSET = np.asarray([0.55, 0.0, 0.45], dtype=np.float64)


def third_person_pose(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from molmo_spaces.env.data_views import create_mlspaces_body

    body = create_mlspaces_body(env.current_data, CAMERA_REFERENCE_BODY)
    rotation = np.asarray(body.pose[:3, :3], dtype=np.float64)
    translation = np.asarray(body.pose[:3, 3], dtype=np.float64)
    position = rotation @ CAMERA_OFFSET + translation
    target = rotation @ LOOKAT_OFFSET + translation
    forward = target - position
    forward /= np.linalg.norm(forward)
    desired_up = rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, desired_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return position, forward, up


def wrist_camera_pose(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model, data = env.current_model, env.current_data
    camera_id = int(model.camera(WRIST_CAMERA_MJCF).id)
    position = np.asarray(data.cam_xpos[camera_id], dtype=np.float64)
    rotation = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
    forward = -rotation[:, 2]
    up = rotation[:, 1]
    forward = forward / np.linalg.norm(forward)
    up = up / np.linalg.norm(up)
    return position, forward, up
