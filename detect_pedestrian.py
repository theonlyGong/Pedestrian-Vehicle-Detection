"""
行人-车辆联合检测跟踪系统
功能：
1. 检测ROI区域内的所有行人
2. 实时跟踪这些行人并标注
3. 检测行人附近的车辆，标记
4. 行人截取保存功能
5. 余弦相似度 + IoU + 中心点距离进行ID去重
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
    parser = argparse.ArgumentParser(description='行人-车辆联合检测跟踪系统')

    # 路径参数
    parser.add_argument('--video', '-v', type=str, default='./videos/20260325150945_448767d6_chunk_012_5fps.mp4',
                        help='输入视频路径')
    parser.add_argument('--model', '-m', type=str, default='./checkpoints/yolo26x.pt',
                        help='YOLO模型路径 (默认: ./checkpoints/yolo26x.pt)')
    parser.add_argument('--output', '-o', type=str, default='./output_pedestrian.mp4',
                        help='输出视频路径')
    # 截图保存参数（默认开启）
    parser.add_argument('--crops-dir', type=str, default='./person_crops',
                        help='行人截图保存目录 (默认: ./person_crops)')
    parser.add_argument('--no-save-crops', action='store_true',
                        help='禁用行人截图保存')

    # 行人检测参数（主要目标）
    parser.add_argument('--person-conf', type=float, default=0.25,
                        help='行人检测置信度阈值 (默认: 0.25)')

    # 车辆检测参数（次要目标）
    parser.add_argument('--vehicle-conf', type=float, default=0.3,
                        help='车辆检测置信度阈值 (默认: 0.3)')
    parser.add_argument('--vehicle-person-dist', type=int, default=100,
                        help='车人距离阈值，单位像素 (默认: 100)')

    # ROI区域参数
    parser.add_argument('--center-region-ratio', type=float, default=0.6,
                        help='中心检测区域占画面比例 (0-1, 默认0.6)')
    parser.add_argument('--center-region-ratio-w', type=float, default=0.9,
                        help='中心检测区域宽度占画面比例 (默认0.9)')
    parser.add_argument('--center-region-ratio-h', type=float, default=0.6,
                        help='中心检测区域高度占画面比例 (默认0.6)')

    # 行人跟踪参数
    parser.add_argument('--person-track-dist', type=int, default=100,
                        help='行人跟踪距离阈值 (默认: 100)')
    parser.add_argument('--person-max-age', type=int, default=20,
                        help='行人最大丢失帧数 (默认: 20)')
    parser.add_argument('--iou-threshold', type=float, default=0.15,
                        help='IOU匹配阈值 (默认: 0.15)')

    # 重识别参数
    parser.add_argument('--reid-threshold', type=float, default=0.75,
                        help='余弦相似度阈值 (默认: 0.75)')
    parser.add_argument('--reid-history', type=int, default=40,
                        help='重识别历史帧数 (默认: 40)')
    parser.add_argument('--reid-min-absent-frames', type=int, default=None,
                        help='重识别最小消失帧数，默认3秒对应的帧数 (默认: None=自动计算)')

    # 其他
    parser.add_argument('--no-display', action='store_true',
                        help='不显示实时窗口')
    parser.add_argument('--save-video', action='store_true',
                        help='保存输出视频')

    return parser.parse_args()


class PersonReidentifier:
    """人物重识别器 - 使用HSV直方图和余弦相似度"""

    def __init__(self, similarity_threshold=0.7, history_frames=40):
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

    def find_match(self, crop, frame_count, min_absent_frames=0, active_ids=None):
        """
        查找匹配的人物
        min_absent_frames: 最小消失帧数，只有消失超过这个帧数的历史人物才允许匹配
        active_ids: 当前活跃的ID集合，这些ID会被排除
        """
        if active_ids is None:
            active_ids = set()

        if crop is None or crop.size == 0:
            return None, 0.0

        features = self.extract_features(crop)
        if features is None or features.size == 0:
            return None, 0.0

        best_match, best_similarity = None, self.similarity_threshold

        for person_id, data in self.saved_persons.items():
            # 排除活跃ID
            if person_id in active_ids:
                continue
            # 检查是否消失足够长时间
            absent_frames = frame_count - data['last_frame']
            if absent_frames < min_absent_frames:
                continue

            similarities = [self.calculate_similarity(features, hist) for hist in data['history']]
            avg_similarity = np.mean(similarities) if similarities else 0

            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match = person_id

        return best_match, best_similarity


class VehicleTracker:
    """车辆跟踪器 - 简单的IOU + 距离匹配"""

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

    def __init__(self, max_distance=100, max_age=20, iou_threshold=0.15):
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

    def update(self, detections, reidentifier=None, frame=None, frame_count=None, min_absent_frames=0):
        """更新行人跟踪 - 跟踪ROI内所有行人"""
        new_tracked, result = {}, []

        detections_with_info = []
        for (x1, y1, x2, y2) in detections:
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

        # 重识别匹配（中间帧丢失后找回）
        if reidentifier is not None and frame is not None:
            active_ids = set(new_tracked.keys())
            for i, det in enumerate(detections_with_info):
                if det['matched']:
                    continue
                if det['crop'] is not None and det['crop'].size > 0:
                    # 只有消失超过min_absent_frames的历史人物才允许匹配
                    matched_id, similarity = reidentifier.find_match(
                        det['crop'], frame_count, min_absent_frames, active_ids
                    )
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

        # 新行人 - 在分配新ID前，先检查与所有历史人物的相似度（必须消失超过min_absent_frames）
        for det in detections_with_info:
            if det['matched']:
                continue

            new_id = None
            new_color = None
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center']

            # 检查与所有历史人物的相似度（只有消失超过min_absent_frames才允许匹配）
            if reidentifier is not None and det['crop'] is not None and det['crop'].size > 0:
                active_ids = set(new_tracked.keys()) | matched_tracks
                matched_id, similarity = reidentifier.find_match(
                    det['crop'], frame_count, min_absent_frames, active_ids
                )
                if matched_id is not None and similarity >= 0.7:
                    # 复用历史ID
                    new_id = matched_id
                    # 如果该ID在活跃跟踪中，使用其颜色；否则生成新颜色
                    if matched_id in self.tracked_persons:
                        _, _, new_color, _, _ = self.tracked_persons[matched_id]
                    elif matched_id in new_tracked:
                        _, _, new_color, _, _ = new_tracked[matched_id]
                    else:
                        new_color = self.get_color()
                    print(f"[重识别] 新检测人物匹配到历史ID {matched_id} (相似度: {similarity:.3f}, 消失{min_absent_frames}+帧)")

            # 如果没有匹配到历史ID，分配新ID
            if new_id is None:
                new_id = self.next_id
                self.next_id += 1
                new_color = self.get_color()

            new_features = None
            if reidentifier is not None and det['crop'] is not None and det['crop'].size > 0:
                new_features = reidentifier.extract_features(det['crop'])
                reidentifier.add_person(new_id, det['crop'], frame_count)

            new_tracked[new_id] = ((cx, cy), det['bbox'], new_color, 0, new_features)
            result.append((x1, y1, x2, y2, new_id, new_color, det['crop']))

        # 保留未匹配但年龄未超期的
        for pid, (old_center, old_bbox, old_color, age, old_features) in self.tracked_persons.items():
            if pid not in matched_tracks and pid not in new_tracked:
                if age < self.max_age:
                    new_tracked[pid] = (old_center, old_bbox, old_color, age + 1, old_features)

        self.tracked_persons = new_tracked
        return result


class PedestrianVehicleDetector:
    """行人-车辆联合检测器"""

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
        self.center_roi = None  # (x1, y1, x2, y2) 中心区域
        self.fps = None  # 视频帧率，用于计算3秒对应的帧数

    def set_fps(self, fps):
        """设置视频帧率，用于计算重识别的最小消失帧数"""
        self.fps = fps

    def get_center_roi(self):
        """计算画面中心区域坐标"""
        if self.center_roi is None and self.frame_width is not None:
            ratio_w = self.args.center_region_ratio_w if self.args.center_region_ratio_w is not None else self.args.center_region_ratio
            ratio_h = self.args.center_region_ratio_h if self.args.center_region_ratio_h is not None else self.args.center_region_ratio
            w, h = self.frame_width, self.frame_height
            roi_w, roi_h = w * ratio_w, h * ratio_h
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

    def detect_and_track(self, frame, frame_count):
        """在中心区域内检测并跟踪行人和车辆"""
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
        tracked_vehicles = self.vehicle_tracker.update(vehicle_detections)

        # 计算3秒对应的帧数（用于重识别限制）
        if self.args.reid_min_absent_frames is not None:
            min_absent_frames = self.args.reid_min_absent_frames
        elif self.fps is not None:
            min_absent_frames = int(self.fps * 3)  # 默认3秒
        else:
            min_absent_frames = 0  # 如果不知道fps，则不限制

        # 更新行人跟踪（ROI内所有行人）
        tracked_persons = self.person_tracker.update(
            person_detections,
            self.reidentifier,
            frame,
            frame_count,
            min_absent_frames
        )

        # 筛选行人附近的车辆
        nearby_vehicles = self.filter_vehicles_near_persons(tracked_vehicles, tracked_persons)

        return tracked_persons, nearby_vehicles

    def filter_vehicles_near_persons(self, all_vehicles, tracked_persons):
        """筛选出靠近行人的车辆"""
        if not tracked_persons:
            return []

        # 获取所有行人的位置
        person_centers = []
        for x1, y1, x2, y2, pid, color, crop in tracked_persons:
            pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
            person_centers.append((pcx, pcy))

        # 筛选靠近任意行人的车辆
        nearby_vehicles = []
        for x1, y1, x2, y2, vid, conf in all_vehicles:
            vcx, vcy = (x1 + x2) / 2, (y1 + y2) / 2

            # 检查是否靠近任意行人
            is_nearby = False
            for pcx, pcy in person_centers:
                dist = ((vcx - pcx) ** 2 + (vcy - pcy) ** 2) ** 0.5
                if dist < self.args.vehicle_person_dist:
                    is_nearby = True
                    break

            if is_nearby:
                nearby_vehicles.append((x1, y1, x2, y2, vid, conf))

        return nearby_vehicles

    def draw_annotations(self, frame, tracked_persons, nearby_vehicles, frame_count):
        """绘制标注"""
        # 绘制中心区域（黄色粗线框）
        if self.center_roi:
            rx1, ry1, rx2, ry2 = self.center_roi
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 3)
            ratio_w = self.args.center_region_ratio_w if self.args.center_region_ratio_w is not None else self.args.center_region_ratio
            ratio_h = self.args.center_region_ratio_h if self.args.center_region_ratio_h is not None else self.args.center_region_ratio
            cv2.putText(frame, f"Center ROI ({ratio_w:.0%}x{ratio_h:.0%})",
                       (rx1, ry1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 绘制行人 - 细框（线宽1）
        for x1, y1, x2, y2, pid, color, crop in tracked_persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            label = f"person id{pid}"
            cv2.putText(frame, label, (x1, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 绘制行人附近的车辆 - 细框（线宽1），绿色
        for x1, y1, x2, y2, vid, conf in nearby_vehicles:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label = f"car id{vid} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 统计信息
        stats = [
            f"Frame: {frame_count}",
            f"Persons: {len(tracked_persons)}",
            f"Nearby Cars: {len(nearby_vehicles)}"
        ]
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (10, 25 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame


def save_person_crop(crop, person_id, frame_count, base_dir):
    """保存行人截图 - 每个行人独立文件夹"""
    if crop is None or crop.size == 0:
        return False

    # 为该行人创建独立文件夹
    person_dir = Path(base_dir) / f"person_{person_id}"
    person_dir.mkdir(parents=True, exist_ok=True)

    # 保存图片: frame_{帧号:06d}.jpg
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

    detector = PedestrianVehicleDetector(args)
    detector.set_fps(fps)  # 设置fps用于计算3秒消失限制

    # 计算并显示重识别参数
    if args.reid_min_absent_frames is not None:
        print(f"重识别最小消失帧数: {args.reid_min_absent_frames} (手动设置)")
    else:
        print(f"重识别最小消失帧数: {int(fps * 3)} (3秒对应帧数)")

    # 准备输出
    out = None
    if args.save_video:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        print(f"输出: {args.output}")

    # 准备截图保存（默认开启）
    save_crops = not args.no_save_crops
    if save_crops:
        os.makedirs(args.crops_dir, exist_ok=True)
        print(f"截图保存到: {args.crops_dir}")
        print(f"每个行人将创建单独的文件夹 (person_1, person_2, ...)")

    # 第一帧：初始化
    ret, first_frame = cap.read()
    if not ret:
        print("错误: 无法读取第一帧")
        return

    # 初始化ROI
    detector.frame_height, detector.frame_width = first_frame.shape[:2]
    detector.get_center_roi()
    print(f"ROI区域: {detector.center_roi}")

    # 处理第一帧
    tracked_persons, nearby_vehicles = detector.detect_and_track(first_frame, 1)
    annotated = detector.draw_annotations(first_frame.copy(), tracked_persons, nearby_vehicles, 1)

    frame_count = 1
    total_crops_saved = 0

    # 保存截图
    if save_crops:
        for x1, y1, x2, y2, pid, color, crop in tracked_persons:
            if save_person_crop(crop, pid, 1, args.crops_dir):
                total_crops_saved += 1

    if out:
        out.write(annotated)
    if not args.no_display:
        cv2.imshow("Pedestrian-Vehicle Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            return

    print(f"开始处理 (按 'q' 退出)...")

    # 处理后续帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        tracked_persons, nearby_vehicles = detector.detect_and_track(frame, frame_count)
        annotated = detector.draw_annotations(frame.copy(), tracked_persons, nearby_vehicles, frame_count)

        # 保存截图
        if save_crops:
            for x1, y1, x2, y2, pid, color, crop in tracked_persons:
                if save_person_crop(crop, pid, frame_count, args.crops_dir):
                    total_crops_saved += 1

        if out:
            out.write(annotated)
        if not args.no_display:
            cv2.imshow("Pedestrian-Vehicle Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户中断")
                break

        if frame_count % 30 == 0:
            print(f"\r进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)  截图: {total_crops_saved}", end='', flush=True)

    print(f"\n完成: {frame_count}帧, 总截图: {total_crops_saved}")

    # 输出统计报告
    if save_crops and total_crops_saved > 0:
        print("\n" + "=" * 60)
        print("行人检测统计报告")
        print("=" * 60)
        print(f"总帧数: {frame_count}")
        print(f"检测到唯一行人数: {detector.person_tracker.next_id - 1}")
        print(f"保存的图片总数: {total_crops_saved}")
        if frame_count > 0:
            print(f"平均每帧保存: {total_crops_saved / frame_count:.2f} 张")
        print(f"\n行人文件夹保存在: {args.crops_dir}")
        print(f"  - person_1/ 到 person_{detector.person_tracker.next_id - 1}/")
        print("=" * 60)

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