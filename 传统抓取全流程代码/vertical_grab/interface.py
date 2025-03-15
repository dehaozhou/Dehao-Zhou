from numpy import ndarray
from .catch import vertical_catch_main


def vertical_catch(
        mask: ndarray,
        depth_frame: ndarray,
        color_intr: dict,
        current_pose: list,
        arm_gripper_length: float,
        vertical_rx_ry_rz: list,
        rotation_matrix: list,
        translation_vector: list,
        use_point_depth_or_mean: bool = True
):
    """
    垂直抓取的主要函数
    Args:
        mask: 物体轮廓信息
        depth_frame: 深度帧
        color_intr: 相机内参
        current_pose: 当前位姿
        arm_gripper_length: 夹爪长度
        vertical_rx_ry_rz: 预设的夹爪垂直的位姿
        rotation_matrix: 手眼标定结果的转换矩阵
        translation_vector: 手眼标定结果的平移矩阵
        use_point_depth_or_mean: 是否使用平均深度还是单点深度，True：使用单点深度

    Returns:
        垂直抓取的过程中的三个关键位姿点，按照顺序调用并加入夹爪开关闭合可以实现整套垂直抓取
    """
    return vertical_catch_main(
        mask,
        depth_frame,
        color_intr,
        current_pose,
        arm_gripper_length,
        vertical_rx_ry_rz,
        rotation_matrix,
        translation_vector,
        use_point_depth_or_mean
    )
