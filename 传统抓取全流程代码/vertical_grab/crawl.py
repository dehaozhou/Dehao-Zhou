#!/usr/bin/env python
# !coding=utf-8
"""

在 机械臂 零位状态 通过设置 相机坐标系下物体得 位置 x y z 来验证 计算出来得 其次变换矩阵是否准确

"""

import numpy as np


def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    )

    Ry = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )

    Rz = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )

    R = Rz @ Ry @ Rx  # 先z轴再y轴最后x轴
    return R


def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]

    return H


def chage_pose(pose, num):
    """
    根据物体和基座的其次变换矩阵 求得 物体z轴 0 0 num 所在位置对应 基座标系的位姿
    y轴补偿6cm
    Args:
        pose:
        nums:

    Returns:
    pose:

    """

    matrix = pose_to_homogeneous_matrix(pose)

    obj_init = np.array([0, 0, num])

    obj_init = np.append(obj_init, [1])  # 将物体坐标转换为齐次坐标

    obj_base_init = matrix.dot(obj_init)

    return [i for i in obj_base_init[:3]] + pose[3:]
