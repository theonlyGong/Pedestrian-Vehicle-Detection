'''
测试yolov26对输入视频的性能效果，视频分辨率为1080p；
主要观察是否有人和车的漏检问题（连续视频帧）
'''


import cv2
import os
import random
import sys
from collections import defaultdict
from ultralytics import YOLO


class ObjectTracker:
    """改进的目标跟踪器，使用IOU和距离进行匹配"""

    def __init__(self, max_distance=150, max_age=15):
        self.tracked_objects = {}
        self.class_counters = defaultdict(int)
        self.global_id_counter = 0
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

    def calculate_distance(self, box1, box2):
        """计算两个框中心点的距离"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        cx1 = (x1_1 + x2_1) / 2
        cy1 = (y1_1 + y2_1) / 2
        cx2 = (x1_2 + x2_2) / 2
        cy2 = (y1_2 + y2_2) / 2

        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    def update(self, detections):
        """更新跟踪器"""
        new_tracked = {}
        result = []

        detections_with_info = []
        for (x1, y1, x2, y2, class_name) in detections:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            detections_with_info.append({
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy),
                'class_name': class_name,
                'matched': False
            })

        matched_detections = set()
        matched_tracks = set()

        # IOU匹配
        for obj_id, (old_cx, old_cy, old_bbox, old_class, old_color, age, old_display_id) in self.tracked_objects.items():
            best_match = None
            best_iou = 0.2

            for i, det in enumerate(detections_with_info):
                if det['matched'] or det['class_name'] != old_class:
                    continue

                iou = self.calculate_iou(old_bbox, det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = i

            if best_match is not None:
                det = detections_with_info[best_match]
                x1, y1, x2, y2 = det['bbox']
                cx, cy = det['center']
                new_tracked[obj_id] = (cx, cy, det['bbox'], old_class, old_color, 0, old_display_id)
                result.append((x1, y1, x2, y2, old_class, old_display_id, old_color))
                det['matched'] = True
                matched_detections.add(best_match)
                matched_tracks.add(obj_id)

        # 距离匹配
        for obj_id, (old_cx, old_cy, old_bbox, old_class, old_color, age, old_display_id) in self.tracked_objects.items():
            if obj_id in matched_tracks:
                continue

            best_match = None
            best_dist = float('inf')

            for i, det in enumerate(detections_with_info):
                if det['matched'] or det['class_name'] != old_class:
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
                new_tracked[obj_id] = (cx, cy, det['bbox'], old_class, old_color, 0, old_display_id)
                result.append((x1, y1, x2, y2, old_class, old_display_id, old_color))
                det['matched'] = True
                matched_detections.add(best_match)
                matched_tracks.add(obj_id)

        # 新物体分配ID
        for det in detections_with_info:
            if det['matched']:
                continue

            class_name = det['class_name']
            self.class_counters[class_name] += 1
            display_id = self.class_counters[class_name]
            self.global_id_counter += 1
            global_id = self.global_id_counter

            new_color = self.get_color()
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center']
            new_tracked[global_id] = (cx, cy, det['bbox'], class_name, new_color, 0, display_id)
            result.append((x1, y1, x2, y2, class_name, display_id, new_color))

        # 保留未匹配但年龄未超期的跟踪
        for obj_id, (old_cx, old_cy, old_bbox, old_class, old_color, age, old_display_id) in self.tracked_objects.items():
            if obj_id not in matched_tracks:
                if age < self.max_age:
                    new_tracked[obj_id] = (old_cx, old_cy, old_bbox, old_class, old_color, age + 1, old_display_id)

        self.tracked_objects = new_tracked
        return result

    def get_active_tracks(self):
        """获取当前活跃的跟踪"""
        active = defaultdict(int)
        for obj_id, (cx, cy, bbox, class_name, color, age, display_id) in self.tracked_objects.items():
            if age == 0:
                active[class_name] += 1
        return active


def parse_arguments():
    """解析命令行参数"""
    model_path = "./checkpoints/yolo26m.pt"
    video_path = "./videos/20260325145929_195f3111_chunk_005_5fps.mp4"

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]

    return model_path, video_path


def create_directories(output_dir, output_video_dir):
    """创建输出目录"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_video_dir, exist_ok=True)


def load_model(model_path):
    """加载YOLO模型"""
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return None
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


def create_video_writer(output_video_dir, video_info):
    """创建视频写入器"""
    output_video_path = os.path.join(output_video_dir, "detected_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, video_info['fps'],
                                (video_info['width'], video_info['height']))

    if not out_video.isOpened():
        print(f"错误: 无法创建视频输出文件: {output_video_path}")
        return None, None

    print(f"将保存检测视频到: {output_video_path}")
    return out_video, output_video_path


def calculate_frames_to_save(total_frames, save_frames_count):
    """计算要保存的帧索引"""
    frames_to_save = set()
    if total_frames > 0:
        for i in range(save_frames_count):
            frame_idx = int((i + 1) * total_frames / (save_frames_count + 1))
            frames_to_save.add(frame_idx)
    else:
        frames_to_save = set(range(30, 30 * save_frames_count + 1, 30))

    print(f"将保存以下帧: {sorted(frames_to_save)}")
    return frames_to_save


def detect_objects(model, frame, target_classes, class_names, conf_threshold=0.1, min_conf=0.15):
    """检测目标"""
    results = model(frame, classes=target_classes, verbose=False, conf=conf_threshold)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = class_names.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if conf > min_conf:
                detections.append((x1, y1, x2, y2, class_name))

    return detections


def draw_detections(frame, tracked_objects):
    """绘制检测框和标签"""
    for (x1, y1, x2, y2, class_name, obj_id, color) in tracked_objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}{obj_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10

        cv2.rectangle(frame,
                     (x1, label_y - label_size[1] - 5),
                     (x1 + label_size[0], label_y + 5),
                     color, -1)
        cv2.putText(frame, label, (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_statistics(frame, frame_count, current_stats, class_names):
    """在左上角显示统计信息"""
    stat_lines = [f"Frame: {frame_count}"]
    for class_name in sorted(class_names.values()):
        count = current_stats.get(class_name, 0)
        stat_lines.append(f"{class_name}: {count}")

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
    for i, line in enumerate(stat_lines):
        color = (0, 255, 0) if i == 0 else (0, 255, 255)
        cv2.putText(frame, line, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += line_height


def save_frame(frame, frame_count, frames_to_save, saved_count, output_dir, save_frames_count):
    """保存指定帧"""
    if frame_count in frames_to_save and saved_count < save_frames_count:
        save_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(save_path, frame)
        saved_count += 1
        print(f"保存帧: {save_path} ({saved_count}/{save_frames_count})")
        return saved_count
    return saved_count


def process_frame(model, frame, frame_count, tracker, target_classes, class_names):
    """处理单帧"""
    # 检测
    detections = detect_objects(model, frame, target_classes, class_names)

    # 跟踪
    tracked_objects = tracker.update(detections)

    # 统计
    current_frame_objects = defaultdict(set)
    for (x1, y1, x2, y2, class_name, obj_id, color) in tracked_objects:
        current_frame_objects[class_name].add(obj_id)

    current_stats = {class_name: len(obj_set)
                     for class_name, obj_set in current_frame_objects.items()}

    # 绘制
    draw_detections(frame, tracked_objects)
    draw_statistics(frame, frame_count, current_stats, class_names)

    return tracked_objects, current_stats


def print_statistics(all_object_ids, max_concurrent, frame_stats_history, frame_count, output_dir, saved_count):
    """输出最终统计报告"""
    print("\n" + "=" * 60)
    print("检测统计报告（已去重）")
    print("=" * 60)

    # 唯一物体数
    total_unique = 0
    print("\n【视频总唯一物体数】:")
    for class_name in sorted(all_object_ids.keys()):
        unique_count = len(all_object_ids[class_name])
        total_unique += unique_count
        print(f"  {class_name}: {unique_count} 个")
    print(f"  总计: {total_unique} 个")

    # 最大并发数
    print("\n【单帧最大并发数】:")
    total_max = 0
    for class_name in sorted(max_concurrent.keys()):
        total_max += max_concurrent[class_name]
        print(f"  {class_name}: {max_concurrent[class_name]} 个")
    print(f"  总计: {total_max} 个")

    # 平均数量
    print("\n【平均每帧数量】:")
    avg_counts = defaultdict(float)
    for stats in frame_stats_history:
        for class_name, count in stats.items():
            avg_counts[class_name] += count

    total_avg = 0
    for class_name in sorted(avg_counts.keys()):
        avg = avg_counts[class_name] / frame_count
        total_avg += avg
        print(f"  {class_name}: {avg:.2f} 个/帧")
    print(f"  总计: {total_avg:.2f} 个/帧")

    # ID列表
    print("\n【各类别检测到的唯一物体ID列表】:")
    for class_name in sorted(all_object_ids.keys()):
        ids = sorted(all_object_ids[class_name])
        print(f"  {class_name}: {ids}")

    print(f"\n已保存 {saved_count} 帧到: {output_dir}")
    print("=" * 60)


def process_video(model, cap, video_info, tracker, config):
    """处理视频主流程"""
    output_dir = config['output_dir']
    output_video_dir = config['output_video_dir']
    save_frames_count = config['save_frames_count']
    target_classes = config['target_classes']
    class_names = config['class_names']

    # 创建视频写入器
    out_video, _ = create_video_writer(output_video_dir, video_info)
    if out_video is None:
        return False

    # 计算要保存的帧
    frames_to_save = calculate_frames_to_save(video_info['total_frames'], save_frames_count)

    # 统计变量
    frame_count = 0
    saved_count = 0
    all_object_ids = defaultdict(set)
    max_concurrent = defaultdict(int)
    frame_stats_history = []

    print("\n开始处理视频...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 处理帧
        tracked_objects, current_stats = process_frame(
            model, frame, frame_count, tracker, target_classes, class_names
        )

        # 更新统计
        for (x1, y1, x2, y2, class_name, obj_id, color) in tracked_objects:
            all_object_ids[class_name].add(obj_id)
        frame_stats_history.append(current_stats)

        for class_name, count in current_stats.items():
            if count > max_concurrent[class_name]:
                max_concurrent[class_name] = count

        # 保存帧
        saved_count = save_frame(frame, frame_count, frames_to_save, saved_count, output_dir, save_frames_count)

        # 显示和写入
        cv2.imshow("YOLOv26 Detection", frame)
        out_video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    # 输出统计
    print_statistics(all_object_ids, max_concurrent, frame_stats_history, frame_count, output_dir, saved_count)

    return True


def main():
    """主函数：核心配置和调用"""
    # ========== 核心配置 ==========
    config = {
        'model_path': "./checkpoints/yolo26m.pt",
        'video_path': "./output_sr/super_resolution_output.mp4",
        'output_dir': "./output_frames",
        'output_video_dir': "./output_video",
        'save_frames_count': 10,
        'target_classes': [0, 2],  # 人，私家车
        'class_names': {0: "person", 2: "car"},
        'tracker_params': {'max_distance': 150, 'max_age': 15}
    }

    # 命令行参数覆盖
    if len(sys.argv) > 1:
        config['video_path'] = sys.argv[1]
    if len(sys.argv) > 2:
        config['model_path'] = sys.argv[2]
    # ========== 配置结束 ==========

    print("=" * 60)
    print("YOLOv26 目标检测与跟踪")
    print("=" * 60)

    # 创建目录
    create_directories(config['output_dir'], config['output_video_dir'])

    # 加载模型
    model = load_model(config['model_path'])
    if model is None:
        return

    # 打开视频
    cap, video_info = open_video(config['video_path'])
    if cap is None:
        return

    # 初始化跟踪器
    tracker = ObjectTracker(**config['tracker_params'])

    # 处理视频
    success = process_video(model, cap, video_info, tracker, config)

    if success:
        print("\n处理完成!")
    else:
        print("\n处理失败!")


if __name__ == "__main__":
    main()