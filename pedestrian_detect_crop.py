import cv2
import os
import sys
import random
from ultralytics import YOLO


class ObjectTracker:
    """简单的目标跟踪器，使用IOU和距离进行匹配"""

    def __init__(self, max_distance=80, max_age=5):
        self.next_id = 1
        self.tracked_objects = {}  # id -> (center_x, center_y, bbox, color, age)
        self.max_distance = max_distance
        self.max_age = max_age

    def get_color(self):
        """生成随机颜色"""
        return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    def calculate_iou(self, box1, box2):
        """计算两个框的IOU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections):
        """
        更新跟踪器
        detections: list of (x1, y1, x2, y2)
        返回: list of (x1, y1, x2, y2, obj_id, color)
        """
        new_tracked = {}
        result = []

        # 准备检测框
        detections_with_info = []
        for (x1, y1, x2, y2) in detections:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            detections_with_info.append({
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy),
                'matched': False
            })

        matched_detections = set()
        matched_tracks = set()

        # IOU匹配
        for obj_id, (old_cx, old_cy, old_bbox, old_color, age) in self.tracked_objects.items():
            best_match = None
            best_iou = 0.3

            for i, det in enumerate(detections_with_info):
                if det['matched']:
                    continue

                iou = self.calculate_iou(old_bbox, det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = i

            if best_match is not None:
                det = detections_with_info[best_match]
                x1, y1, x2, y2 = det['bbox']
                cx, cy = det['center']
                new_tracked[obj_id] = (cx, cy, det['bbox'], old_color, 0)
                result.append((x1, y1, x2, y2, obj_id, old_color))
                det['matched'] = True
                matched_detections.add(best_match)
                matched_tracks.add(obj_id)

        # 距离匹配（未匹配的跟踪）
        for obj_id, (old_cx, old_cy, old_bbox, old_color, age) in self.tracked_objects.items():
            if obj_id in matched_tracks:
                continue

            best_match = None
            best_dist = float('inf')

            for i, det in enumerate(detections_with_info):
                if det['matched']:
                    continue

                dist = ((det['center'][0] - old_cx) ** 2 +
                        (det['center'][1] - old_cy) ** 2) ** 0.5
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_match = i

            if best_match is not None:
                det = detections_with_info[best_match]
                x1, y1, x2, y2 = det['bbox']
                cx, cy = det['center']
                new_tracked[obj_id] = (cx, cy, det['bbox'], old_color, 0)
                result.append((x1, y1, x2, y2, obj_id, old_color))
                det['matched'] = True
                matched_detections.add(best_match)
                matched_tracks.add(obj_id)

        # 新物体分配ID
        for det in detections_with_info:
            if det['matched']:
                continue

            new_id = self.next_id
            self.next_id += 1
            new_color = self.get_color()
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center']
            new_tracked[new_id] = (cx, cy, det['bbox'], new_color, 0)
            result.append((x1, y1, x2, y2, new_id, new_color))

        # 保留未匹配但年龄未超期的跟踪
        for obj_id, (old_cx, old_cy, old_bbox, old_color, age) in self.tracked_objects.items():
            if obj_id not in matched_tracks:
                if age < self.max_age:
                    new_tracked[obj_id] = (old_cx, old_cy, old_bbox, old_color, age + 1)

        self.tracked_objects = new_tracked
        return result


def parse_arguments():
    """解析命令行参数"""
    model_path = "./checkpoints/yolo26x.pt"
    video_path = "./output_sr/super_resolution_output.mp4"

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]

    return model_path, video_path


def create_directories(base_output_dir):
    """创建基础输出目录"""
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"行人检测框将保存到: {base_output_dir}")
    print(f"每个行人将创建单独的文件夹 (person_1, person_2, ...)")


def create_person_directory(base_output_dir, person_id):
    """为每个行人创建单独的文件夹"""
    person_dir = os.path.join(base_output_dir, f"person_{person_id}")
    os.makedirs(person_dir, exist_ok=True)
    return person_dir


def load_model(model_path):
    """加载YOLO模型"""
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None
    print(f"加载模型: {model_path}")
    return YOLO(model_path)


def open_video(video_path):
    """打开视频文件"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频: {video_path}")
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
    return cap, {'fps': fps, 'width': width, 'height': height, 'total_frames': total_frames}


def detect_persons(model, frame, conf_threshold=0.15):
    """
    检测行人
    返回: list of (x1, y1, x2, y2)
    """
    results = model(frame, classes=[0], verbose=False, conf=conf_threshold)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2))

    return detections


def draw_detections(frame, tracked_objects):
    """在图像上绘制检测框和标签"""
    for (x1, y1, x2, y2, person_id, color) in tracked_objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"person{person_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10

        cv2.rectangle(frame,
                     (x1, label_y - label_size[1] - 5),
                     (x1 + label_size[0], label_y + 5),
                     color, -1)
        cv2.putText(frame, label, (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_statistics(frame, frame_count, person_count, total_unique_persons):
    """在左上角显示统计信息"""
    stat_lines = [
        f"Frame: {frame_count}",
        f"Current: {person_count}",
        f"Total IDs: {total_unique_persons}"
    ]

    max_text_width = 0
    line_height = 25
    for line in stat_lines:
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        max_text_width = max(max_text_width, text_size[0])

    overlay = frame.copy()
    bg_height = len(stat_lines) * line_height + 10
    cv2.rectangle(overlay, (5, 5), (max_text_width + 15, bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    y_offset = 25
    colors = [(0, 255, 0), (0, 255, 255), (255, 255, 0)]
    for i, line in enumerate(stat_lines):
        cv2.putText(frame, line, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
        y_offset += line_height


def save_person_crops(frame, tracked_objects, base_output_dir, frame_count):
    """
    保存每个行人的检测框为单独的图片
    每个行人有独立的文件夹: person_crops/person_{id}/frame_{帧号}.jpg
    """
    saved_count = 0
    for (x1, y1, x2, y2, person_id, color) in tracked_objects:
        # 裁剪行人区域
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            continue

        # 为该行人创建文件夹
        person_dir = create_person_directory(base_output_dir, person_id)

        # 保存图片 (命名: frame_000001.jpg)
        filename = f"frame_{frame_count:06d}.jpg"
        save_path = os.path.join(person_dir, filename)
        cv2.imwrite(save_path, person_crop)
        saved_count += 1

    return saved_count


def process_video(model, cap, base_output_dir):
    """处理视频主流程"""
    frame_count = 0
    total_crops_saved = 0

    # 初始化跟踪器
    tracker = ObjectTracker(max_distance=100, max_age=10)

    print("\n开始处理视频...")
    print("按 'q' 键退出")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 检测行人
        detections = detect_persons(model, frame)

        # 跟踪 (分配稳定ID)
        tracked_objects = tracker.update(detections)

        # 当前帧行人数
        person_count = len(tracked_objects)
        total_unique_persons = tracker.next_id - 1

        # 保存行人检测框 (按ID分文件夹)
        saved = save_person_crops(frame, tracked_objects, base_output_dir, frame_count)
        total_crops_saved += saved

        # 绘制
        draw_detections(frame, tracked_objects)
        draw_statistics(frame, frame_count, person_count, total_unique_persons)

        # 显示
        cv2.imshow("Pedestrian Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n用户中断")
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    return frame_count, total_crops_saved, tracker.next_id - 1


def print_summary(frame_count, total_crops_saved, total_unique_persons, base_output_dir):
    """输出统计报告"""
    print("\n" + "=" * 60)
    print("行人检测统计报告")
    print("=" * 60)
    print(f"总帧数: {frame_count}")
    print(f"检测到唯一行人数: {total_unique_persons}")
    print(f"保存的图片总数: {total_crops_saved}")
    print(f"平均每帧保存: {total_crops_saved / frame_count:.2f} 张")
    print(f"\n行人文件夹保存在: {base_output_dir}")
    print(f"  - person_1/ 到 person_{total_unique_persons}/")
    print("=" * 60)


def main():
    """主函数"""
    # ========== 核心配置 ==========
    config = {
        'model_path': "./checkpoints/yolo26m.pt",
        'video_path': "./videos/20260325150818_40b9302e_chunk_007_5fps.mp4",
        'base_output_dir': "./person_crops",
        'conf_threshold': 0.15
    }

    # 命令行参数覆盖
    if len(sys.argv) > 1:
        config['video_path'] = sys.argv[1]
    if len(sys.argv) > 2:
        config['model_path'] = sys.argv[2]
    # ========== 配置结束 ==========

    print("=" * 60)
    print("行人检测工具 - 每人单独文件夹")
    print("=" * 60)

    # 创建基础输出目录
    create_directories(config['base_output_dir'])

    # 加载模型
    model = load_model(config['model_path'])
    if model is None:
        return

    # 打开视频
    cap, video_info = open_video(config['video_path'])
    if cap is None:
        return

    # 处理视频
    frame_count, total_crops_saved, total_unique_persons = process_video(
        model, cap, config['base_output_dir']
    )

    # 输出统计
    print_summary(frame_count, total_crops_saved, total_unique_persons, config['base_output_dir'])

    print("\n处理完成!")


if __name__ == "__main__":
    main()