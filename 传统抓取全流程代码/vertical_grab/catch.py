import cv2
import numpy as np
from numpy import ndarray
from typing import Tuple

from vertical_grab.convert import convert
from vertical_grab.crawl import chage_pose


def vertical_catch_main(
        mask: ndarray,
        depth_frame: ndarray,
        color_intr: dict,
        current_pose: list,
        arm_gripper_length: float,
        vertical_rx_ry_rz: list,
        rotation_matrix: list,
        translation_vector: list,
        use_point_depth_or_mean: bool = True,
) -> Tuple[list, list, list]:
    """
    :param center:  抓取的中心点位
    :param mask:    抓取物体的轮廓信息
    :param depth_frame:     物体的深度值信息
    :param color_intr:      相机的内参
    :param current_pose:    当前的位姿信息
    :param arm_gripper_length:      夹爪的长度
    :param vertical_rx_ry_rz:       正确的夹爪偏移角度
    :param rotation_matrix:         手眼标定的旋转矩阵
    :param translation_vector:      手眼标定的平移矩阵
    :param use_point_depth_or_mean:     使用一个点位的深度信息还是整个物体的平均深度

    :return:
    above_object_pose：      垂直抓取物体上方的位姿
    correct_angle_pose：     垂直抓取物体正确的角度位姿
    finally_pose：           垂直抓取最终下爪的抓取位姿
    """
    # 开始凭着mask中心点位抓取``
    _, center = compute_angle_with_mask(mask)
    real_x, real_y = center[0], center[1]

    # 修改对抓取点位深度信息的获取方式由单点改为整个mask的深度信息
    if not use_point_depth_or_mean:
        dis = depth_frame[real_y][real_x]
    else:
        depth_mask = depth_frame[mask == 255]
        non_zero_values = depth_mask[depth_mask != 0]
        sorted_values = np.sort(non_zero_values)
        top_20_percent_index = int(0.2 * len(sorted_values))
        top_20_percent_values = sorted_values[:top_20_percent_index]
        dis = np.mean(top_20_percent_values)

    x = int(dis * (real_x - color_intr["ppx"]) / color_intr["fx"])
    y = int(dis * (real_y - color_intr["ppy"]) / color_intr["fy"])
    dis = int(dis)
    x, y, z = (
        (x) * 0.001,
        (y) * 0.001,
        (dis) * 0.001,
    )  # 夹爪刚好碰到 -180  前面加针 -200

    # 计算物体位置，位置是物体中心点正上方10公分
    obj_pose = convert(x, y, z, *current_pose, rotation_matrix, translation_vector)
    obj_pose = [i for i in obj_pose]

    # 下潜距离
    _z = min(obj_pose[2] * 0.8 + 0.10, 0.1 + 0.03)

    # 最终位置为物体上方 + 夹爪 + 10cm的距离
    obj_pose[2] = obj_pose.copy()[2] + 0.10 + arm_gripper_length * 0.001

    # 修改为垂直于桌面的RX,RY,RZ``
    obj_pose[3:] = vertical_rx_ry_rz

    above_object_pose = obj_pose.copy()

    # 计算偏转角度
    _angle = obj_pose[5] - vertical_rx_ry_rz[2]
    angle_joint, _ = compute_angle_with_mask(mask)
    angle = (angle_joint / 180) * 3.14 - _angle
    catch_pose = obj_pose.copy()

    # 移动到上方和旋转角度决定了抓取的流畅性

    # 计算偏转角
    if obj_pose[5] - angle > 0:
        catch_pose[5] = obj_pose[5] - angle
    else:
        catch_pose[5] = obj_pose[5] - angle

    # 记录一下
    correct_angle_pose = catch_pose.copy()

    # 计算最终位姿
    finally_pose = chage_pose(list(catch_pose), _z)

    finally_pose = finally_pose.copy()

    return above_object_pose, correct_angle_pose, finally_pose


def compute_angle_with_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 初始化最大外接矩形
    min_rect = None
    max_area = 0

    for contour in contours:
        # 计算最小外接矩形
        center, (w, h), angle = cv2.minAreaRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            min_rect = center, (w, h), angle

    # 获取最小外接矩形的信息
    center, (width, height), angle = min_rect

    if width > height:
        angle = -(90 - angle)
    else:
        angle = angle
    return angle, center