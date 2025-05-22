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


from listen import voice_command_to_keyword  # ✅ 导入语音识别函数

def segment_image(image_path, output_mask='mask1.png'):
    """
    image_path: 图像路径或 BGR 图像数组
    output_mask: 掩码保存路径
    自动语音获取检测目标 → YOLO 检测 → SAM 分割 → 保存掩码
    检测不到时支持手动点击目标。
    """

    # ✅ 使用语音获取目标类别
    print("🎙️ 请通过语音输入目标物体名称...")
    target_class = voice_command_to_keyword()
    if not target_class:
        print("⚠️ 未识别出目标类别，请重试。")
        return None
    print(f"✅ 语音识别目标类别为：{target_class}")

    # 2) 使用 YOLO 检测
    detections, vis_img = detect_objects(image_path, target_class)

    # 保存检测图
    cv2.imwrite('detection_visualization.jpg', vis_img)

    # 3) 准备 RGB 图像用于 SAM
    if isinstance(image_path, str):
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            raise ValueError(f"无法读取图像路径: {image_path}")
        image_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)

    # 4) 初始化 SAM
    predictor = choose_model()
    predictor.set_image(image_rgb)

    # 5) 判断是否检测到目标
    if detections:
        best_det = max(detections, key=lambda x: x["conf"])
        results = predictor(bboxes=[best_det["xyxy"]])
        center, mask = process_sam_results(results)
        print(f"✅ 自动检测到目标：{best_det['cls']}，置信度：{best_det['conf']:.2f}")
    else:
        # 🖱️ 手动点击选择
        print("⚠️ 未检测到目标，请点击图像选择对象")

        point = []

        def click_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point.extend([x, y])
                print(f"🖱️ 点击坐标：{x}, {y}")
                cv2.setMouseCallback('Select Object', lambda *args: None)

        cv2.namedWindow('Select Object', cv2.WINDOW_NORMAL)
        cv2.imshow('Select Object', vis_img)
        cv2.setMouseCallback('Select Object', click_handler)

        while True:
            key = cv2.waitKey(100)
            if point:
                break
            if cv2.getWindowProperty('Select Object', cv2.WND_PROP_VISIBLE) < 1:
                print("❌ 窗口被关闭，未进行点击")
                return None

        cv2.destroyAllWindows()

        # 使用点击点提示 SAM
        results = predictor(points=[point], labels=[1])
        center, mask = process_sam_results(results)

    # 6) 保存掩码
    if mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        print(f"✅ 分割掩码已保存：{output_mask}")
    else:
        print("⚠️ 分割失败，未生成掩码")

    return mask



if __name__ == '__main__':
    seg_mask = segment_image('color.png')
    print("Segmentation result mask shape:", seg_mask.shape if seg_mask is not None else None)
