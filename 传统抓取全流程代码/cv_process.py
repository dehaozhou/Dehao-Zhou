import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

def choose_model():
    """Initialize SAM predictor with proper parameters"""
    model_weight = 'sam_b.pt'
    overrides = dict(
        task='segment',
        mode='predict',
        #imgsz=1024,
        model=model_weight,
        conf=0.01,
        save=False
    )
    return SAMPredictor(overrides=overrides)


def set_classes(model, target_class):
    """Set YOLO-World model to detect specific class"""
    model.set_classes([target_class])


def detect_objects(image_or_path, target_class=None):
    """
    Detect objects with YOLO-World
    image_or_path: can be a file path (str) or a numpy array (image).
    Returns: (list of bboxes in xyxy format, detected classes list, visualization image)
    """
    model = YOLO("yolov8s-world.pt")
    if target_class:
        set_classes(model, target_class)

    # YOLOv8 的 predict 可同时处理 文件路径(str) 或 图像数组(np.ndarray)
    results = model.predict(image_or_path)

    boxes = results[0].boxes
    vis_img = results[0].plot()  # Get visualized detection results

    # Extract valid detections
    valid_boxes = []
    for box in boxes:
        if box.conf.item() > 0.25:  # Confidence threshold
            valid_boxes.append({
                "xyxy": box.xyxy[0].tolist(),
                "conf": box.conf.item(),
                "cls": results[0].names[box.cls.item()]
            })

    return valid_boxes, vis_img


def process_sam_results(results):
    """Process SAM results to get mask and center point"""
    if not results or not results[0].masks:
        return None, None

    # Get first mask (assuming single object segmentation)
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    # Find contour and center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


def segment_image(image_path, output_mask='mask1.png'):
    """
    image_path: can be either a file path (str) or a numpy array (BGR image).
    output_mask: output mask file name.
    1) 用户可决定是否检测特定类别
    2) 调用 detect_objects 做初步检测
    3) 若 detections 存在，自动选最高分；否则让用户点击选择
    4) 用 SAM 分割并保存结果掩码
    5) 返回分割后的 mask (np.ndarray or None)
    """
    # 1) 用户输入 - 是否检测特定类别
    use_target_class = input("Detect specific class? (yes/no): ").lower() == 'yes'
    target_class = input("Enter class name: ").strip() if use_target_class else None

    # 2) 初步检测 - YOLO
    detections, vis_img = detect_objects(image_path, target_class)

    # 保存检测可视化结果
    cv2.imwrite('detection_visualization.jpg', vis_img)

    # 3) 准备给 SAM 的图像 (RGB 格式)
    if isinstance(image_path, str):
        # 如果是字符串，说明是图像路径
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        image_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    else:
        # 否则假设 image_path 就是一个 BGR 的 numpy 数组
        image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)

    # 4) 初始化 SAM predictor
    predictor = choose_model()
    predictor.set_image(image_rgb)

    # 5) 判断是否有目标检测结果
    if detections:
        # 自动选最高置信度
        best_det = max(detections, key=lambda x: x["conf"])
        results = predictor(bboxes=[best_det["xyxy"]])
        center, mask = process_sam_results(results)
        print(f"Auto-selected {best_det['cls']} with confidence {best_det['conf']:.2f}")
    else:
        # 手动点击
        print("No detections - click on target object")
        cv2.imshow('Select Object', vis_img)
        point = []

        def click_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point.extend([x, y])
                cv2.destroyAllWindows()

        cv2.setMouseCallback('Select Object', click_handler)
        cv2.waitKey(0)

        if len(point) == 2:
            results = predictor(points=[point], labels=[1])
            center, mask = process_sam_results(results)
        else:
            raise ValueError("No selection made")

    # 6) 保存 mask
    if mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        print(f"Segmentation saved to {output_mask}")
    else:
        print("[WARNING] Could not generate mask")

    return mask


if __name__ == '__main__':
    seg_mask = segment_image('color.png')
    print("Segmentation result mask shape:", seg_mask.shape if seg_mask is not None else None)
