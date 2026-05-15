"""
车辆-行人联合检测跟踪系统
功能：
1. 检测置信度最高的前5辆且在画面中央的car
2. 实时跟踪这5辆车并标注
3. 检测车附近的人，标记并分配ID
4. 人截取保存功能
5. 余弦相似度 + IoU+中心点距离进行ID去重
"""

import argparse
import cv2
import os
import sys
import random
import numpy as np
from collections import deque
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='车辆-行人联合检测跟踪系统')

    # 路径参数
    parser.add_argument('--video', '-v', type=str, default = './videos/20260325150945_448767d6_chunk_012_5fps.mp4',
                        help='输入视频路径')
    parser.add_argument('--model', '-m', type=str, default='./checkpoints/yolo26x.pt',
                        help='YOLO模型路径 (默认: ./checkpoints/yolo26x.pt)')
    parser.add_argument('--output', '-o', type=str, default='./output_vehicle_v2.mp4',
                        help='输出视频路径')
    parser.add_argument('--save-crops', action='store_true',
                        help='保存行人截图到单独文件夹')
    parser.add_argument('--crops-dir', type=str, default='./person_crops_v2',
                        help='行人截图保存目录')

    # 车辆检测参数
    parser.add_argument('--vehicle-conf', type=float, default=0.3,
                        help='车辆检测置信度阈值 (默认: 0.3)')
    parser.add_argument('--top-k-vehicles', type=int, default=10,
                        help='选择关注车辆数量 (默认: 10)')
    parser.add_argument('--center-region-ratio', type=float, default=0.6,
                        help='中心检测区域占画面比例 (0-1, 默认0.6, 如0.6表示画面中心60%%区域)')
    parser.add_argument('--center-region-ratio-w', type=float, default=0.9,
                        help='中心检测区域宽度占画面比例 (默认None, 设置后覆盖center-region-ratio的宽度)')
    parser.add_argument('--center-region-ratio-h', type=float, default=0.6,
                        help='中心检测区域高度占画面比例 (默认None, 设置后覆盖center-region-ratio的高度)')

    # 行人检测参数
    parser.add_argument('--person-conf', type=float, default=0.15,
                        help='行人检测置信度阈值 (默认: 0.15)')
    parser.add_argument('--person-vehicle-dist', type=int, default=150,
                        help='人车距离阈值，单位像素 (默认: 150)')

    # 行人跟踪参数
    parser.add_argument('--person-track-dist', type=int, default=100,
                        help='行人跟踪距离阈值 (默认: 100)')
    parser.add_argument('--person-max-age', type=int, default=20,
                        help='行人最大丢失帧数 (默认: 10)')
    parser.add_argument('--iou-threshold', type=float, default=0.15,
                        help='IOU匹配阈值 (默认: 0.15)')

    # 重识别参数
    parser.add_argument('--reid-threshold', type=float, default=0.65,
                        help='余弦相似度阈值 (默认: 0.75)')
    parser.add_argument('--reid-history', type=int, default=40,
                        help='重识别历史帧数 (默认: 30)')

    # 车辆重选参数
    parser.add_argument('--reselect-cooldown', type=int, default=30,
                        help='车辆重选冷却帧数 (默认: 30)')
    parser.add_argument('--reselect-threshold', type=float, default=0.5,
                        help='触发重选的最小车辆比例 (0-1, 默认0.5表示少于一半时触发)')

    # 其他
    parser.add_argument('--no-display', action='store_true',
                        help='不显示实时窗口')
    parser.add_argument('--save-video', action='store_true',
                        help='保存输出视频')

    return parser.parse_args()


class PersonReidentifier:
    """人物重识别器 - 使用HSV直方图和余弦相似度"""

    def __init__(self, similarity_threshold=0.7, history_frames=30):
        self.saved_persons = {}
        self.similarity_threshold = similarity_threshold
        self.history_frames = history_frames

    def extract_features(self, crop):
        """提取HSV颜色直方图特征"""
        if crop is None or crop.size == 0:
            return None

        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])

            for hist in [h_hist, s_hist, v_hist]:
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            features = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])
            return features
        except Exception:
            return None

    def calculate_similarity(self, features1, features2):
        """计算余弦相似度"""
        if features1 is None or features2 is None:
            return 0.0

        dot_product = np.dot(features1, features2)
        norm1, norm2 = np.linalg.norm(features1), np.linalg.norm(features2)

        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    def add_person(self, person_id, crop, frame_count):
        """添加或更新人物特征"""
        if crop is None or crop.size == 0:
            return

        features = self.extract_features(crop)
        if features is not None and features.size > 0:
            if person_id not in self.saved_persons:
                self.saved_persons[person_id] = {
                    'features': features,
                    'last_frame': frame_count,
                    'history': deque(maxlen=self.history_frames)
                }
            self.saved_persons[person_id]['history'].append(features)
            self.saved_persons[person_id]['last_frame'] = frame_count

    def find_match(self, crop, exclude_ids=None):
        """查找匹配的人物"""
        if exclude_ids is None:
            exclude_ids = set()

        if crop is None or crop.size == 0:
            return None, 0.0

        features = self.extract_features(crop)
        if features is None or features.size == 0:
            return None, 0.0

        best_match, best_similarity = None, self.similarity_threshold

        for person_id, data in self.saved_persons.items():
            if person_id in exclude_ids:
                continue

            similarities = [self.calculate_similarity(features, hist) for hist in data['history']]
            avg_similarity = np.mean(similarities) if similarities else 0

            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match = person_id

        return best_match, best_similarity


class VehicleTracker:
    """车辆跟踪器 - IOU + 距离匹配"""

    def __init__(self, max_distance=100, iou_threshold=0.3):
        self.next_id = 1
        self.tracked_vehicles = {}
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold

    @staticmethod
    def calculate_iou(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections):
        """
        更新车辆跟踪
        detections: list of (x1, y1, x2, y2, conf)
        返回: list of (x1, y1, x2, y2, vehicle_id, conf)
        """
        new_tracked = {}
        result = []

        detections_with_info = []
        for (x1, y1, x2, y2, conf) in detections:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections_with_info.append({
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy),
                'conf': conf,
                'matched': False
            })

        matched_tracks = set()

        # IOU匹配
        for vid, (old_center, old_bbox, old_conf) in self.tracked_vehicles.items():
            best_match, best_iou = None, self.iou_threshold

            for i, det in enumerate(detections_with_info):
                if det['matched']:
                    continue
                iou = self.calculate_iou(old_bbox, det['bbox'])
                if iou > best_iou:
                    best_iou, best_match = iou, i

            if best_match is not None:
                det = detections_with_info[best_match]
                new_tracked[vid] = (det['center'], det['bbox'], det['conf'])
                result.append((*det['bbox'], vid, det['conf']))
                det['matched'] = True
                matched_tracks.add(vid)

        # 距离匹配
        for vid, (old_center, old_bbox, old_conf) in self.tracked_vehicles.items():
            if vid in matched_tracks:
                continue

            best_match, best_dist = None, float('inf')
            for i, det in enumerate(detections_with_info):
                if det['matched']:
                    continue
                dist = ((det['center'][0] - old_center[0]) ** 2 +
                       (det['center'][1] - old_center[1]) ** 2) ** 0.5
                if dist < best_dist and dist < self.max_distance:
                    best_dist, best_match = dist, i

            if best_match is not None:
                det = detections_with_info[best_match]
                new_tracked[vid] = (det['center'], det['bbox'], det['conf'])
                result.append((*det['bbox'], vid, det['conf']))
                det['matched'] = True
                matched_tracks.add(vid)

        # 新车
        for det in detections_with_info:
            if det['matched']:
                continue
            new_id = self.next_id
            self.next_id += 1
            new_tracked[new_id] = (det['center'], det['bbox'], det['conf'])
            result.append((*det['bbox'], new_id, det['conf']))

        self.tracked_vehicles = new_tracked
        return result


class PersonTracker:
    """行人跟踪器 - IOU + 中心点距离 + 重识别"""

    def __init__(self, max_distance=80, max_age=5, iou_threshold=0.3):
        self.next_id = 1
        self.tracked_persons = {}
        self.max_distance = max_distance
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    @staticmethod
    def get_color():
        return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    @staticmethod
    def calculate_iou(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    @staticmethod
    def is_person_nearby_vehicles(person_bbox, vehicle_bboxes, distance_threshold=150):
        px1, py1, px2, py2 = person_bbox
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2

        for vx1, vy1, vx2, vy2 in vehicle_bboxes:
            vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
            dist = ((pcx - vcx) ** 2 + (pcy - vcy) ** 2) ** 0.5
            if dist < distance_threshold:
                return True
        return False

    def update(self, detections, reidentifier=None, frame=None, frame_count=None,
               focus_vehicle_bboxes=None, distance_threshold=150):
        """更新行人跟踪"""
        new_tracked, result = {}, []

        if focus_vehicle_bboxes is None:
            focus_vehicle_bboxes = []

        # 筛选车辆附近的行人
        detections_with_info = []
        for (x1, y1, x2, y2) in detections:
            if not self.is_person_nearby_vehicles((x1, y1, x2, y2), focus_vehicle_bboxes, distance_threshold):
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections_with_info.append({
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy),
                'matched': False,
                'crop': frame[y1:y2, x1:x2] if frame is not None else None
            })

        matched_tracks = set()

        # IOU匹配
        for pid, (old_center, old_bbox, old_color, age, _) in self.tracked_persons.items():
            best_match, best_iou = None, self.iou_threshold

            for i, det in enumerate(detections_with_info):
                if det['matched']:
                    continue
                iou = self.calculate_iou(old_bbox, det['bbox'])
                if iou > best_iou:
                    best_iou, best_match = iou, i

            if best_match is not None:
                det = detections_with_info[best_match]
                x1, y1, x2, y2 = det['bbox']
                cx, cy = det['center']

                new_features = None
                if reidentifier is not None and det['crop'] is not None and det['crop'].size > 0:
                    new_features = reidentifier.extract_features(det['crop'])
                    reidentifier.add_person(pid, det['crop'], frame_count)

                new_tracked[pid] = ((cx, cy), det['bbox'], old_color, 0, new_features)
                result.append((x1, y1, x2, y2, pid, old_color, det['crop']))
                det['matched'] = True
                matched_tracks.add(pid)

        # 距离匹配
        for pid, (old_center, old_bbox, old_color, age, _) in self.tracked_persons.items():
            if pid in matched_tracks:
                continue

            best_match, best_dist = None, float('inf')
            for i, det in enumerate(detections_with_info):
                if det['matched']:
                    continue
                dist = ((det['center'][0] - old_center[0]) ** 2 +
                       (det['center'][1] - old_center[1]) ** 2) ** 0.5
                if dist < best_dist and dist < self.max_distance:
                    best_dist, best_match = dist, i

            if best_match is not None:
                det = detections_with_info[best_match]
                x1, y1, x2, y2 = det['bbox']
                cx, cy = det['center']

                new_features = None
                if reidentifier is not None and det['crop'] is not None and det['crop'].size > 0:
                    new_features = reidentifier.extract_features(det['crop'])
                    reidentifier.add_person(pid, det['crop'], frame_count)

                new_tracked[pid] = ((cx, cy), det['bbox'], old_color, 0, new_features)
                result.append((x1, y1, x2, y2, pid, old_color, det['crop']))
                det['matched'] = True
                matched_tracks.add(pid)

        # 重识别匹配
        if reidentifier is not None and frame is not None:
            active_ids = set(new_tracked.keys())
            for i, det in enumerate(detections_with_info):
                if det['matched']:
                    continue
                if det['crop'] is not None and det['crop'].size > 0:
                    matched_id, _ = reidentifier.find_match(det['crop'], exclude_ids=active_ids)
                    if matched_id is not None and matched_id not in active_ids:
                        x1, y1, x2, y2 = det['bbox']
                        cx, cy = det['center']
                        old_color = self.tracked_persons.get(matched_id, (None, None, self.get_color(), 0, None))[2]

                        new_features = reidentifier.extract_features(det['crop'])
                        reidentifier.add_person(matched_id, det['crop'], frame_count)

                        new_tracked[matched_id] = ((cx, cy), det['bbox'], old_color, 0, new_features)
                        result.append((x1, y1, x2, y2, matched_id, old_color, det['crop']))
                        det['matched'] = True
                        active_ids.add(matched_id)

        # 新行人
        for det in detections_with_info:
            if det['matched']:
                continue

            new_id = self.next_id
            self.next_id += 1
            new_color = self.get_color()
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center']

            new_features = None
            if reidentifier is not None and det['crop'] is not None and det['crop'].size > 0:
                new_features = reidentifier.extract_features(det['crop'])
                reidentifier.add_person(new_id, det['crop'], frame_count)

            new_tracked[new_id] = ((cx, cy), det['bbox'], new_color, 0, new_features)
            result.append((x1, y1, x2, y2, new_id, new_color, det['crop']))

        # 保留未匹配但年龄未超期的
        for pid, (old_center, old_bbox, old_color, age, old_features) in self.tracked_persons.items():
            if pid not in matched_tracks:
                if age < self.max_age:
                    new_tracked[pid] = (old_center, old_bbox, old_color, age + 1, old_features)

        self.tracked_persons = new_tracked
        return result


class VehiclePersonDetector:
    """车辆-行人联合检测器"""

    def __init__(self, args):
        self.model = YOLO(args.model)
        self.args = args

        self.vehicle_tracker = VehicleTracker(
            max_distance=100,
            iou_threshold=args.iou_threshold
        )
        self.person_tracker = PersonTracker(
            max_distance=args.person_track_dist,
            max_age=args.person_max_age,
            iou_threshold=args.iou_threshold
        )
        self.reidentifier = PersonReidentifier(
            similarity_threshold=args.reid_threshold,
            history_frames=args.reid_history
        )

        self.frame_width = None
        self.frame_height = None
        self.focus_vehicle_ids = set()
        self.center_roi = None  # (x1, y1, x2, y2) 中心区域
        self.last_reselect_frame = 0  # 上次重新选择车辆的帧号
        self.reselect_cooldown = args.reselect_cooldown  # 重新选择的冷却帧数
        self.reselect_threshold = args.reselect_threshold  # 触发重选的最小车辆比例

    def get_center_roi(self):
        """计算画面中心区域坐标"""
        if self.center_roi is None and self.frame_width is not None:
            # 使用单独的宽高比例，如果未设置则使用统一的ratio
            ratio_w = self.args.center_region_ratio_w if self.args.center_region_ratio_w is not None else self.args.center_region_ratio
            ratio_h = self.args.center_region_ratio_h if self.args.center_region_ratio_h is not None else self.args.center_region_ratio
            w, h = self.frame_width, self.frame_height
            # 中心区域的宽高
            roi_w, roi_h = w * ratio_w, h * ratio_h
            # 中心区域左上角坐标
            x1 = int((w - roi_w) / 2)
            y1 = int((h - roi_h) / 2)
            x2 = int(x1 + roi_w)
            y2 = int(y1 + roi_h)
            self.center_roi = (x1, y1, x2, y2)
        return self.center_roi

    def is_in_center_roi(self, x1, y1, x2, y2):
        """判断目标是否在中心区域内（中心点在区域内即可）"""
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rx1, ry1, rx2, ry2 = self.get_center_roi()
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def count_vehicles_in_roi(self):
        """统计当前有多少辆关注车辆在中心区域内"""
        count = 0
        for vid, (center, bbox, conf) in self.vehicle_tracker.tracked_vehicles.items():
            if vid in self.focus_vehicle_ids:
                cx, cy = center
                rx1, ry1, rx2, ry2 = self.get_center_roi()
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    count += 1
        return count

    def select_top_vehicles(self, frame, existing_vehicles=None):
        """在中心区域内选择置信度最高的前N辆车

        Args:
            frame: 当前帧
            existing_vehicles: 已有的车辆列表 [(vid, bbox), ...]，用于保留现有车辆
        """
        if self.frame_width is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            self.get_center_roi()  # 初始化中心区域

        rx1, ry1, rx2, ry2 = self.get_center_roi()

        # 裁剪中心区域进行检测
        center_frame = frame[ry1:ry2, rx1:rx2]
        results = self.model(center_frame, verbose=False, conf=self.args.vehicle_conf)

        vehicles = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls, conf = int(box.cls[0]), float(box.conf[0])
                if cls == 2 and conf >= self.args.vehicle_conf:  # car class
                    # 坐标是相对于中心区域的，需要转换回全局坐标
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    x1, y1, x2, y2 = rx1 + bx1, ry1 + by1, rx1 + bx2, ry1 + by2
                    vehicles.append((x1, y1, x2, y2, conf))

        if len(vehicles) == 0:
            return []

        # 按置信度排序，取top_k
        vehicles.sort(key=lambda x: x[4], reverse=True)

        # 保留现有车辆，添加新车辆
        if existing_vehicles:
            # 保留现有的车辆到跟踪器
            for vid, bbox in existing_vehicles:
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                self.vehicle_tracker.tracked_vehicles[vid] = (
                    (cx, cy), bbox, 0.5
                )
                self.focus_vehicle_ids.add(vid)

            # 添加新车辆直到达到top_k
            existing_count = len(existing_vehicles)
            needed = self.args.top_k_vehicles - existing_count
            new_vehicles = []

            # 找到下一个可用的ID
            next_id = max([vid for vid, _ in existing_vehicles] + [0]) + 1 if existing_vehicles else 1

            # 过滤掉与现有车辆重叠的检测
            filtered_vehicles = []
            for v in vehicles:
                vx1, vy1, vx2, vy2, vconf = v
                vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
                # 检查是否与现有车辆重叠
                overlap = False
                for _, (ex1, ey1, ex2, ey2) in existing_vehicles:
                    ecx, ecy = (ex1 + ex2) / 2, (ey1 + ey2) / 2
                    dist = ((vcx - ecx) ** 2 + (vcy - ecy) ** 2) ** 0.5
                    if dist < 50:  # 中心点距离小于50视为同一辆车
                        overlap = True
                        break
                if not overlap:
                    filtered_vehicles.append(v)

            for i, (x1, y1, x2, y2, conf) in enumerate(filtered_vehicles[:needed]):
                vid = next_id + i
                self.vehicle_tracker.tracked_vehicles[vid] = (
                    ((x1 + x2) / 2, (y1 + y2) / 2),
                    (x1, y1, x2, y2),
                    conf
                )
                self.focus_vehicle_ids.add(vid)
                new_vehicles.append((x1, y1, x2, y2, vid, conf))

            if new_vehicles:
                print(f"重新选择: 保留 {existing_count} 辆现有车辆，新增 {len(new_vehicles)} 辆")
            return [(v[0], v[1], v[2], v[3], v[4], v[5]) for v in new_vehicles]
        else:
            # 初始化：选择top_k车辆
            selected = vehicles[:self.args.top_k_vehicles]

            # 初始化跟踪
            self.vehicle_tracker.tracked_vehicles.clear()
            self.focus_vehicle_ids.clear()
            for i, (x1, y1, x2, y2, conf) in enumerate(selected):
                vid = i + 1
                self.vehicle_tracker.tracked_vehicles[vid] = (
                    ((x1 + x2) / 2, (y1 + y2) / 2),
                    (x1, y1, x2, y2),
                    conf
                )
                self.focus_vehicle_ids.add(vid)

            print(f"选定 {len(selected)} 辆中心区域车辆，ROI: {self.center_roi}")
            return [(v[0], v[1], v[2], v[3], i+1, v[4]) for i, v in enumerate(selected)]

    def detect_and_track(self, frame, frame_count):
        """在中心区域内检测并跟踪车辆和行人"""
        rx1, ry1, rx2, ry2 = self.get_center_roi()

        # 裁剪中心区域
        center_frame = frame[ry1:ry2, rx1:rx2]
        results = self.model(center_frame, verbose=False,
                            conf=min(self.args.vehicle_conf, self.args.person_conf))

        vehicle_detections = []
        person_detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls, conf = int(box.cls[0]), float(box.conf[0])
                # 坐标转换回全局
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = rx1 + bx1, ry1 + by1, rx1 + bx2, ry1 + by2

                if cls == 2 and conf >= self.args.vehicle_conf:  # car
                    vehicle_detections.append((x1, y1, x2, y2, conf))
                elif cls == 0 and conf >= self.args.person_conf:  # person
                    person_detections.append((x1, y1, x2, y2))

        # 更新车辆跟踪
        self.vehicle_tracker.update(vehicle_detections)

        # 只返回选定的关注车辆
        tracked_vehicles = [
            (*bbox, vid, conf)
            for vid, (center, bbox, conf) in self.vehicle_tracker.tracked_vehicles.items()
            if vid in self.focus_vehicle_ids
        ]

        # 检查是否需要重新选择车辆（当ROI内车辆数量少于阈值时）
        min_vehicle_threshold = max(1, int(self.args.top_k_vehicles * self.reselect_threshold))
        vehicles_in_roi = self.count_vehicles_in_roi()

        if vehicles_in_roi < min_vehicle_threshold:
            # 检查冷却期
            if frame_count - self.last_reselect_frame > self.reselect_cooldown:
                print(f"\n帧 {frame_count}: ROI内车辆数({vehicles_in_roi})低于阈值({min_vehicle_threshold})，重新选择车辆...")

                # 收集仍在ROI内的现有车辆
                existing_vehicles = []
                for vid, (center, bbox, conf) in self.vehicle_tracker.tracked_vehicles.items():
                    if vid in self.focus_vehicle_ids:
                        cx, cy = center
                        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                            existing_vehicles.append((vid, bbox))

                # 清除旧的focus_vehicle_ids，保留ROI内的
                self.focus_vehicle_ids.clear()

                # 重新选择车辆
                new_vehicles = self.select_top_vehicles(frame, existing_vehicles=existing_vehicles)
                self.last_reselect_frame = frame_count

                if new_vehicles:
                    print(f"重新选择完成，当前关注车辆: {list(self.focus_vehicle_ids)}")

                # 重新获取跟踪车辆列表
                tracked_vehicles = [
                    (*bbox, vid, conf)
                    for vid, (center, bbox, conf) in self.vehicle_tracker.tracked_vehicles.items()
                    if vid in self.focus_vehicle_ids
                ]

        # 获取关注车辆的位置用于筛选行人
        vehicle_bboxes = [(v[0], v[1], v[2], v[3]) for v in tracked_vehicles]

        # 更新行人跟踪
        tracked_persons = self.person_tracker.update(
            person_detections,
            self.reidentifier,
            frame,
            frame_count,
            focus_vehicle_bboxes=vehicle_bboxes,
            distance_threshold=self.args.person_vehicle_dist
        )

        return tracked_vehicles, tracked_persons

    def draw_annotations(self, frame, tracked_vehicles, tracked_persons, frame_count):
        """绘制标注"""
        # 绘制中心区域（黄色虚线框）
        if self.center_roi:
            rx1, ry1, rx2, ry2 = self.center_roi
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
            # 计算实际使用的宽高比例
            ratio_w = self.args.center_region_ratio_w if self.args.center_region_ratio_w is not None else self.args.center_region_ratio
            ratio_h = self.args.center_region_ratio_h if self.args.center_region_ratio_h is not None else self.args.center_region_ratio
            cv2.putText(frame, f"Center ROI ({ratio_w:.0%}x{ratio_h:.0%})",
                       (rx1, ry1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 绘制车辆 - 细框（线宽1）
        for x1, y1, x2, y2, vid, conf in tracked_vehicles:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label = f"car id{vid} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 绘制行人 - 细框（线宽1）
        for x1, y1, x2, y2, pid, color, crop in tracked_persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            label = f"person id{pid}"
            cv2.putText(frame, label, (x1, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 统计信息
        stats = [
            f"Frame: {frame_count}",
            f"Center Cars: {len(tracked_vehicles)}",
            f"Center Persons: {len(tracked_persons)}"
        ]
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (10, 25 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame


def save_person_crop(crop, person_id, frame_count, base_dir):
    """保存行人截图"""
    if crop is None or crop.size == 0:
        return False

    person_dir = Path(base_dir) / f"person_{person_id}"
    person_dir.mkdir(parents=True, exist_ok=True)

    filename = f"frame_{frame_count:06d}.jpg"
    save_path = person_dir / filename

    try:
        cv2.imwrite(str(save_path), crop)
        return True
    except Exception:
        return False


def process_video(args):
    """处理视频主流程"""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"错误: 无法打开视频: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频: {width}x{height}@{fps:.1f}fps, {total_frames}帧")

    detector = VehiclePersonDetector(args)

    # 准备输出
    out = None
    if args.save_video:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        print(f"输出: {args.output}")

    # 准备截图保存
    if args.save_crops:
        Path(args.crops_dir).mkdir(parents=True, exist_ok=True)
        print(f"截图保存到: {args.crops_dir}")

    # 第一帧：选择关注车辆
    ret, first_frame = cap.read()
    if not ret:
        print("错误: 无法读取第一帧")
        return

    top_vehicles = detector.select_top_vehicles(first_frame)
    if not top_vehicles:
        print("错误: 未检测到足够车辆")
        return

    print(f"选定 {len(top_vehicles)} 辆关注车辆:")
    for x1, y1, x2, y2, vid, conf in top_vehicles:
        print(f"  car id{vid}: conf={conf:.3f}, bbox=({x1},{y1},{x2},{y2})")

    # 处理第一帧
    tracked_vehicles, tracked_persons = detector.detect_and_track(first_frame, 1)
    annotated = detector.draw_annotations(first_frame.copy(), tracked_vehicles, tracked_persons, 1)

    # 保存截图
    if args.save_crops:
        for x1, y1, x2, y2, pid, color, crop in tracked_persons:
            save_person_crop(crop, pid, 1, args.crops_dir)

    if out:
        out.write(annotated)
    if not args.no_display:
        cv2.imshow("Vehicle-Person Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            return

    frame_count = 1
    total_crops_saved = 0
    print(f"开始处理 (按 'q' 退出)...")

    # 处理后续帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        tracked_vehicles, tracked_persons = detector.detect_and_track(frame, frame_count)
        annotated = detector.draw_annotations(frame.copy(), tracked_vehicles, tracked_persons, frame_count)

        # 保存截图
        if args.save_crops:
            for x1, y1, x2, y2, pid, color, crop in tracked_persons:
                if save_person_crop(crop, pid, frame_count, args.crops_dir):
                    total_crops_saved += 1

        if out:
            out.write(annotated)
        if not args.no_display:
            cv2.imshow("Vehicle-Person Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户中断")
                break

        if frame_count % 30 == 0:
            print(f"\r进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)  截图: {total_crops_saved}", end='', flush=True)

    print(f"\n完成: {frame_count}帧, 总截图: {total_crops_saved}")
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"错误: 模型不存在: {args.model}")
        return
    if not os.path.exists(args.video):
        print(f"错误: 视频不存在: {args.video}")
        return

    process_video(args)


if __name__ == "__main__":
    main()