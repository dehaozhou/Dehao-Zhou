import numpy as np
from scipy.spatial.transform import Rotation as R


def convert(x, y, z, x1, y1, z1, rx, ry, rz, rotation_matrix, translation_vector):
    """
    接收单位 m

    我们需要将旋转向量和平移向量转换为齐次变换矩阵，然后使用深度相机识别到的物体坐标（x, y, z）和
    机械臂末端的位姿（x1,y1,z1,rx,ry,rz）来计算物体相对于机械臂基座的位姿（x, y, z, rx, ry, rz）

    """

    rotation_matrix = rotation_matrix
    translation_vector = translation_vector
    # 深度相机识别物体返回的坐标
    obj_camera_coordinates = np.array([x, y, z])

    # 机械臂末端的位姿，单位为弧度
    end_effector_pose = np.array([x1, y1, z1, rx, ry, rz])

    # 将旋转矩阵和平移向量转换为齐次变换矩阵
    T_camera_to_end_effector = np.eye(4)
    T_camera_to_end_effector[:3, :3] = rotation_matrix
    T_camera_to_end_effector[:3, 3] = translation_vector

    # 机械臂末端的位姿转换为齐次变换矩阵
    position = end_effector_pose[:3]
    orientation = R.from_euler("xyz", end_effector_pose[3:], degrees=False).as_matrix()

    T_base_to_end_effector = np.eye(4)
    T_base_to_end_effector[:3, :3] = orientation
    T_base_to_end_effector[:3, 3] = position

    # 计算物体相对于机械臂基座的位姿
    obj_camera_coordinates_homo = np.append(
        obj_camera_coordinates, [1]
    )  # 将物体坐标转换为齐次坐标
    # obj_end_effector_coordinates_homo = np.linalg.inv(T_camera_to_end_effector).dot(obj_camera_coordinates_homo)

    obj_end_effector_coordinates_homo = T_camera_to_end_effector.dot(
        obj_camera_coordinates_homo
    )

    obj_base_coordinates_homo = T_base_to_end_effector.dot(
        obj_end_effector_coordinates_homo
    )

    obj_base_coordinates = obj_base_coordinates_homo[
        :3
    ]  # 从齐次坐标中提取物体的x, y, z坐标

    # 计算物体的旋转
    obj_orientation_matrix = T_base_to_end_effector[:3, :3].dot(rotation_matrix)
    obj_orientation_euler = R.from_matrix(obj_orientation_matrix).as_euler(
        "xyz", degrees=False
    )

    # 组合结果
    obj_base_pose = np.hstack((obj_base_coordinates, obj_orientation_euler))

    obj_base_pose[3:] = rx, ry, rz

    return obj_base_pose
