"""
乘客上下车统计系统
功能：
1. 基于detect_pedestrian.py的逻辑，在固定ROI区域内跟踪行人和车辆
2. 检测行人消失（上车）：当行人ID从有到无消失时，关联最近车辆
3. 检测行人出现（下车）：当行人凭空从车附近出现时，关联最近车辆
4. 使用logging统计整个视频的上下车情况
"""

import argparse
import cv2
import os
import sys
import logging
import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# 从detect_pedestrian导入检测器类
from detect_pedestrian import (
    PedestrianVehicleDetector, save_person_crop, PersonTracker, PersonReidentifier, VehicleTracker
)
from ultralytics import YOLO

import random


class ConditionalReIDPersonTracker(PersonTracker):
    """条件重识别行人跟踪器 - 只有满足特定条件时才使用相似度匹配"""

    def __init__(self, max_distance=100, max_age=20, iou_threshold=0.15, reid_absent_threshold=8):
        """
        初始化跟踪器
        reid_absent_threshold: 使用重识别所需的消失秒数（默认8秒）
        """
        super().__init__(max_distance, max_age, iou_threshold)
        self.reid_absent_threshold = reid_absent_threshold  # 秒数阈值
        self.fps = None  # 帧率，用于计算帧数
        # 记录行人是否离开过ROI: {person_id: {'left_roi': bool, 'last_seen_frame': int}}
        self.person_roi_status = {}

    def set_fps(self, fps):
        """设置帧率"""
        self.fps = fps

    def mark_person_left_roi(self, person_id):
        """标记行人已离开ROI"""
        if person_id in self.person_roi_status:
            self.person_roi_status[person_id]['left_roi'] = True

    def update(self, detections, reidentifier=None, frame=None, frame_count=None, min_absent_frames=0):
        """
        更新行人跟踪 - 条件重识别版本
        只有行人离开ROI或消失超过8秒才使用相似度匹配
        """
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

        # ========== 步骤1: IOU匹配（无条件执行） ==========
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

                # 更新ROI状态：仍在跟踪中
                if pid in self.person_roi_status:
                    self.person_roi_status[pid]['last_seen_frame'] = frame_count

        # ========== 步骤2: 距离匹配（无条件执行） ==========
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

                # 更新ROI状态：仍在跟踪中
                if pid in self.person_roi_status:
                    self.person_roi_status[pid]['last_seen_frame'] = frame_count

        # ========== 步骤3: 重识别匹配（有条件执行） ==========
        # 计算8秒对应的帧数
        reid_frame_threshold = int(self.fps * self.reid_absent_threshold) if self.fps else 8 * 30

        if reidentifier is not None and frame is not None:
            active_ids = set(new_tracked.keys())
            for i, det in enumerate(detections_with_info):
                if det['matched']:
                    continue
                if det['crop'] is None or det['crop'].size == 0:
                    continue

                # 检查是否应该使用重识别
                should_use_reid = False
                use_min_absent_frames = min_absent_frames  # 默认使用传入的值

                # 查找可能匹配的历史ID
                for hist_pid in list(self.person_roi_status.keys()):
                    if hist_pid in active_ids or hist_pid in matched_tracks:
                        continue

                    status = self.person_roi_status[hist_pid]
                    absent_frames = frame_count - status.get('last_seen_frame', frame_count)

                    # 条件1: 行人离开过ROI且现在重新出现
                    if status.get('left_roi', False):
                        should_use_reid = True
                        logging.debug(f"[重识别触发] 行人 {hist_pid} 离开ROI后重新出现")
                        break

                    # 条件2: 消失超过8秒
                    if absent_frames >= reid_frame_threshold:
                        should_use_reid = True
                        logging.debug(f"[重识别触发] 行人 {hist_pid} 消失 {absent_frames} 帧 (>{reid_frame_threshold})")
                        break

                if should_use_reid:
                    # 使用重识别，但只匹配满足条件的ID
                    matched_id, similarity = reidentifier.find_match(
                        det['crop'], frame_count, use_min_absent_frames, active_ids
                    )

                    # 额外检查：确认该ID是否满足重识别条件
                    if matched_id is not None and matched_id not in active_ids:
                        # 验证该ID是否满足条件
                        if matched_id in self.person_roi_status:
                            status = self.person_roi_status[matched_id]
                            absent_frames = frame_count - status.get('last_seen_frame', frame_count)

                            # 如果不满足任一条件，跳过
                            if not status.get('left_roi', False) and absent_frames < reid_frame_threshold:
                                matched_id = None  # 不满足条件，不重用ID

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

                        # 重置ROI状态
                        self.person_roi_status[matched_id] = {
                            'left_roi': False,
                            'last_seen_frame': frame_count
                        }

                        logging.info(f"[重识别] 行人 {matched_id} 满足条件后重新匹配 (相似度: {similarity:.3f})")

        # ========== 步骤4: 新行人（不使用相似度） ==========
        for det in detections_with_info:
            if det['matched']:
                continue

            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center']

            # 直接分配新ID，不使用相似度匹配
            new_id = self.next_id
            self.next_id += 1
            new_color = self.get_color()

            # 记录特征用于未来重识别
            new_features = None
            if reidentifier is not None and det['crop'] is not None and det['crop'].size > 0:
                new_features = reidentifier.extract_features(det['crop'])
                reidentifier.add_person(new_id, det['crop'], frame_count)

            new_tracked[new_id] = ((cx, cy), det['bbox'], new_color, 0, new_features)
            result.append((x1, y1, x2, y2, new_id, new_color, det['crop']))

            # 初始化ROI状态
            self.person_roi_status[new_id] = {
                'left_roi': False,
                'last_seen_frame': frame_count
            }

        # ========== 步骤5: 保留未匹配但年龄未超期的 ==========
        for pid, (old_center, old_bbox, old_color, age, old_features) in self.tracked_persons.items():
            if pid not in matched_tracks and pid not in new_tracked:
                if age < self.max_age:
                    new_tracked[pid] = (old_center, old_bbox, old_color, age + 1, old_features)
                else:
                    # 年龄超期，标记为离开ROI
                    if pid in self.person_roi_status:
                        self.person_roi_status[pid]['left_roi'] = True

        self.tracked_persons = new_tracked
        return result


class ConditionalReIDDetector(PedestrianVehicleDetector):
    """使用条件重识别的检测器"""

    def __init__(self, args):
        # 不调用父类的__init__，而是手动初始化
        self.model = YOLO(args.model)
        self.args = args

        self.vehicle_tracker = VehicleTracker(
            max_distance=100,
            iou_threshold=args.iou_threshold
        )

        # 使用自定义的条件重识别跟踪器
        self.person_tracker = ConditionalReIDPersonTracker(
            max_distance=args.person_track_dist,
            max_age=args.person_max_age,
            iou_threshold=args.iou_threshold,
            reid_absent_threshold=getattr(args, 'reid_absent_seconds', 8)
        )

        self.reidentifier = PersonReidentifier(
            similarity_threshold=args.reid_threshold,
            history_frames=args.reid_history
        )

        self.frame_width = None
        self.frame_height = None
        self.center_roi = None
        self.fps = None

    def set_fps(self, fps):
        """设置视频帧率，传递给跟踪器"""
        self.fps = fps
        self.person_tracker.set_fps(fps)
        logging.info(f"条件重识别阈值: {self.person_tracker.reid_absent_threshold}秒 = {int(fps * self.person_tracker.reid_absent_threshold)}帧")


def setup_logging(output_dir="./logs"):
    """配置日志系统"""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"passenger_count_{timestamp}.log"

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return log_file


class PassengerCounter:
    """乘客上下车计数器 - 严格条件版本"""

    def __init__(self, distance_threshold=100, min_frames_before_exit=3, strict_boarding_distance=30, alighting_distance=15):
        """
        初始化计数器
        distance_threshold: 人车距离阈值，用于判断关联（检测时）
        min_frames_before_exit: 最少存在帧数，避免误检
        strict_boarding_distance: 严格上车距离阈值（默认30像素）
        alighting_distance: 严格下车距离阈值（默认15像素）
        """
        self.distance_threshold = distance_threshold
        self.min_frames_before_exit = min_frames_before_exit
        self.strict_boarding_distance = strict_boarding_distance  # 上车距离阈值
        self.alighting_distance = alighting_distance  # 下车距离阈值（更严格）

        # 跟踪状态
        self.active_persons = {}  # {person_id: {'vehicle_id': vid, 'frames': [], 'center': (x, y), 'in_roi': bool}}

        # 车辆历史状态 - 记录每辆车上一帧是否有人
        self.vehicle_last_frame_persons = {}  # {vehicle_id: bool}
        self.prev_frame_vehicle_persons = {}  # 前一帧的车辆-行人关联

        # 统计信息 - 使用集合去重
        self.boarded_persons = set()  # 已上车的行人ID（去重）
        self.alighted_persons = set()  # 已下车的行人ID（去重）

        # 按车辆统计
        self.vehicle_boarding = defaultdict(set)  # {vehicle_id: {person_id, ...}}
        self.vehicle_alighting = defaultdict(set)  # {vehicle_id: {person_id, ...}}

        # 事件记录（用于调试）
        self.boarding_events = []  # 上车事件
        self.alighting_events = []  # 下车事件

    def calculate_distance(self, center1, center2):
        """计算两点距离"""
        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5

    def find_nearest_vehicle(self, person_center, vehicles):
        """找到距离行人最近的车辆"""
        min_dist = float('inf')
        nearest_vid = None

        for vx1, vy1, vx2, vy2, vid, vconf in vehicles:
            v_center = ((vx1 + vx2) / 2, (vy1 + vy2) / 2)
            dist = self.calculate_distance(person_center, v_center)
            if dist < min_dist:
                min_dist = dist
                nearest_vid = vid

        return nearest_vid, min_dist

    def get_vehicle_persons_mapping(self, tracked_persons, nearby_vehicles):
        """获取每辆车当前关联的行人列表"""
        vehicle_persons = defaultdict(list)  # {vehicle_id: [(pid, distance), ...]}

        for x1, y1, x2, y2, pid, color, crop in tracked_persons:
            p_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            # 找到最近的车辆
            for vx1, vy1, vx2, vy2, vid, vconf in nearby_vehicles:
                v_center = ((vx1 + vx2) / 2, (vy1 + vy2) / 2)
                dist = self.calculate_distance(p_center, v_center)
                if dist < self.alighting_distance:
                    vehicle_persons[vid].append((pid, dist))

        return vehicle_persons

    def update(self, tracked_persons, nearby_vehicles, roi_persons, frame_count):
        """
        更新跟踪状态并检测上下车事件
        参数：
            tracked_persons: 所有跟踪的行人
            nearby_vehicles: 附近车辆
            roi_persons: ROI内的行人ID集合
            frame_count: 当前帧号
        返回：当前帧的事件列表
        """
        current_events = []
        current_person_ids = set()

        # 获取当前帧所有行人信息
        person_centers = {}
        for x1, y1, x2, y2, pid, color, crop in tracked_persons:
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            person_centers[pid] = center
            current_person_ids.add(pid)

        # 获取当前帧每辆车关联的行人（用于下车检测）
        current_vehicle_persons = self.get_vehicle_persons_mapping(tracked_persons, nearby_vehicles)

        # ========== 检查消失的行人（上车事件）==========
        disappeared_persons = set(self.active_persons.keys()) - current_person_ids
        for pid in disappeared_persons:
            person_data = self.active_persons[pid]

            # 条件1: 必须曾在ROI内
            if not person_data.get('in_roi', False):
                del self.active_persons[pid]
                continue

            # 条件2: 必须存在足够帧数
            if len(person_data['frames']) < self.min_frames_before_exit:
                del self.active_persons[pid]
                continue

            # 条件3: 必须有关联的车辆
            last_vehicle = person_data.get('vehicle_id')
            if last_vehicle is None:
                del self.active_persons[pid]
                continue

            # 条件4: 消失前与车辆的距离必须小于严格阈值（30像素）
            last_distance = person_data.get('vehicle_distance', float('inf'))
            if last_distance > self.strict_boarding_distance:
                del self.active_persons[pid]
                continue

            # 条件5: 该行人ID未统计过上车
            if pid in self.boarded_persons:
                del self.active_persons[pid]
                continue

            # 满足所有条件，记录上车事件
            self.boarded_persons.add(pid)
            self.vehicle_boarding[last_vehicle].add(pid)

            event = {
                'person_id': pid,
                'vehicle_id': last_vehicle,
                'frame': frame_count,
                'type': 'boarding',
                'distance': last_distance,
                'duration_frames': len(person_data['frames'])
            }
            self.boarding_events.append(event)
            current_events.append(event)
            logging.info(f"[上车事件] 行人ID {pid} 上了车辆ID {last_vehicle}，距离: {last_distance:.1f}px，帧号: {frame_count}")

            del self.active_persons[pid]

        # ========== 检查新出现的行人（下车事件）==========
        new_persons = current_person_ids - set(self.active_persons.keys())
        for pid in new_persons:
            center = person_centers[pid]

            # 查找最近的车辆
            nearest_vid, min_dist = self.find_nearest_vehicle(center, nearby_vehicles)

            # 条件1: 必须在ROI内才计算下车
            if pid not in roi_persons:
                # 不在ROI内，只记录活跃状态但不统计下车
                self.active_persons[pid] = {
                    'vehicle_id': None,
                    'frames': [frame_count],
                    'center': center,
                    'in_roi': False,
                    'vehicle_distance': float('inf')
                }
                continue

            # 条件2: 必须出现在车辆附近（距离 < 15像素）
            if nearest_vid is None or min_dist > self.alighting_distance:
                # 在ROI内但距离不够
                self.active_persons[pid] = {
                    'vehicle_id': None,
                    'frames': [frame_count],
                    'center': center,
                    'in_roi': True,
                    'vehicle_distance': float('inf')
                }
                continue

            # 条件3: 前一帧该车辆没有检测到人（凭空出现）
            prev_persons = self.prev_frame_vehicle_persons.get(nearest_vid, [])
            if len(prev_persons) > 0:
                # 前一帧该车有人，不算下车
                self.active_persons[pid] = {
                    'vehicle_id': nearest_vid,
                    'frames': [frame_count],
                    'center': center,
                    'in_roi': True,
                    'vehicle_distance': min_dist
                }
                logging.debug(f"[跳过下车] 行人 {pid} 出现但车辆 {nearest_vid} 前一帧有人: {prev_persons}")
                continue

            # 条件4: 该行人ID未统计过下车
            if pid in self.alighted_persons:
                self.active_persons[pid] = {
                    'vehicle_id': nearest_vid,
                    'frames': [frame_count],
                    'center': center,
                    'in_roi': True,
                    'vehicle_distance': min_dist
                }
                continue

            # 满足所有条件，记录下车事件
            self.alighted_persons.add(pid)
            self.vehicle_alighting[nearest_vid].add(pid)

            event = {
                'person_id': pid,
                'vehicle_id': nearest_vid,
                'frame': frame_count,
                'type': 'alighting',
                'distance': min_dist
            }
            self.alighting_events.append(event)
            current_events.append(event)
            logging.info(f"[下车事件] 行人ID {pid} 从车辆ID {nearest_vid} 下车，距离: {min_dist:.1f}px（前一帧该车无人），帧号: {frame_count}")

            # 添加到活跃行人
            self.active_persons[pid] = {
                'vehicle_id': nearest_vid,
                'frames': [frame_count],
                'center': center,
                'in_roi': True,
                'vehicle_distance': min_dist
            }

        # ========== 更新现有活跃行人的信息 ==========
        for pid in self.active_persons:
            if pid in person_centers:
                center = person_centers[pid]
                self.active_persons[pid]['frames'].append(frame_count)
                self.active_persons[pid]['center'] = center

                # 更新是否在ROI内
                self.active_persons[pid]['in_roi'] = pid in roi_persons

                # 更新关联的车辆和距离
                if pid in roi_persons:  # 只在ROI内更新车辆关联
                    nearest_vid, min_dist = self.find_nearest_vehicle(center, nearby_vehicles)
                    if nearest_vid is not None and min_dist < self.distance_threshold:
                        self.active_persons[pid]['vehicle_id'] = nearest_vid
                        self.active_persons[pid]['vehicle_distance'] = min_dist

        # 更新前一帧车辆状态
        self.prev_frame_vehicle_persons = current_vehicle_persons

        return current_events

    def finalize(self, frame_count):
        """视频结束时处理未完成的跟踪 - 结算上车事件"""
        remaining_events = []
        for pid, person_data in list(self.active_persons.items()):
            # 条件检查
            if not person_data.get('in_roi', False):
                continue
            if len(person_data['frames']) < self.min_frames_before_exit:
                continue

            last_vehicle = person_data.get('vehicle_id')
            if last_vehicle is None:
                continue

            last_distance = person_data.get('vehicle_distance', float('inf'))
            if last_distance > self.strict_boarding_distance:
                continue

            if pid in self.boarded_persons:
                continue

            # 结算上车
            self.boarded_persons.add(pid)
            self.vehicle_boarding[last_vehicle].add(pid)

            event = {
                'person_id': pid,
                'vehicle_id': last_vehicle,
                'frame': frame_count,
                'type': 'boarding',
                'distance': last_distance,
                'duration_frames': len(person_data['frames']),
                'note': '视频结束时结算'
            }
            self.boarding_events.append(event)
            remaining_events.append(event)
            logging.info(f"[上车事件-结算] 行人ID {pid} 上了车辆ID {last_vehicle}（视频结束）")

        return remaining_events

    def get_statistics(self):
        """获取统计报告 - 按车辆ID组织"""
        # 按车辆ID分组统计（使用集合去重后的数据）
        vehicle_stats = {}

        # 处理上车统计（去重后的行人ID）
        for vid, persons in self.vehicle_boarding.items():
            if vid not in vehicle_stats:
                vehicle_stats[vid] = {'boarding': [], 'alighting': []}
            vehicle_stats[vid]['boarding'] = sorted(list(persons))

        # 处理下车统计（去重后的行人ID）
        for vid, persons in self.vehicle_alighting.items():
            if vid not in vehicle_stats:
                vehicle_stats[vid] = {'boarding': [], 'alighting': []}
            vehicle_stats[vid]['alighting'] = sorted(list(persons))

        return vehicle_stats

    def save_statistics(self, output_path):
        """保存统计结果到JSON文件 - 简化格式"""
        vehicle_stats = self.get_statistics()

        # 转换为指定格式
        result = {}
        for vid, data in vehicle_stats.items():
            result[f"车辆ID{vid}"] = {
                "上车行人": data['boarding'],
                "下车行人": data['alighting']
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logging.info(f"统计数据已保存到: {output_path}")
        return result


class PassengerCountDetector(ConditionalReIDDetector):
    """扩展的行人车辆检测器，添加上下车统计功能"""

    def __init__(self, args):
        super().__init__(args)
        self.passenger_counter = PassengerCounter(
            distance_threshold=args.vehicle_person_dist,
            min_frames_before_exit=args.min_frames_before_exit,
            strict_boarding_distance=getattr(args, 'strict_boarding_distance', 30),
            alighting_distance=getattr(args, 'alighting_distance', 15)
        )
        self.current_events = []

    def detect_and_track_with_counting(self, frame, frame_count):
        """检测跟踪并统计上下车"""
        # 调用父类方法获取跟踪结果
        tracked_persons, nearby_vehicles = self.detect_and_track(frame, frame_count)

        # 计算哪些行人在ROI内
        roi_persons = set()
        if self.center_roi:
            rx1, ry1, rx2, ry2 = self.center_roi
            for x1, y1, x2, y2, pid, color, crop in tracked_persons:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    roi_persons.add(pid)

        # 更新上下车计数（传入ROI内行人集合）
        self.current_events = self.passenger_counter.update(
            tracked_persons, nearby_vehicles, roi_persons, frame_count
        )

        return tracked_persons, nearby_vehicles

    def draw_annotations_with_counting(self, frame, tracked_persons, nearby_vehicles, frame_count):
        """绘制标注，包括上下车事件提示"""
        # 先调用父类方法绘制基础标注
        annotated = self.draw_annotations(frame, tracked_persons, nearby_vehicles, frame_count)

        # 计算统计数据
        total_boarding = len(self.passenger_counter.boarding_events)
        total_alighting = len(self.passenger_counter.alighting_events)
        y_offset = 100  # 在基础统计信息下方显示

        # 上车统计
        boarding_text = f"Boarding: {total_boarding}"
        cv2.putText(annotated, boarding_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 下车统计
        alighting_text = f"Alighting: {total_alighting}"
        cv2.putText(annotated, alighting_text, (10, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 显示当前事件（如果有）
        for i, event in enumerate(self.current_events[-3:]):  # 最多显示最近3个事件
            if event['type'] == 'boarding':
                text = f"Boarding: Person {event['person_id']} -> Car {event['vehicle_id']}"
                color = (0, 0, 255)
            else:
                text = f"Alighting: Person {event['person_id']} <- Car {event['vehicle_id']}"
                color = (255, 0, 0)
            cv2.putText(annotated, text, (10, y_offset + 50 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return annotated


def parse_args():
    """解析命令行参数，继承基础参数并添加新参数"""
    parser = argparse.ArgumentParser(description='乘客上下车统计系统')

    # 路径参数
    parser.add_argument('--video', '-v', type=str, default='./videos/20260325150945_448767d6_chunk_012_5fps.mp4',
                        help='输入视频路径')
    parser.add_argument('--model', '-m', type=str, default='./checkpoints/yolo26x.pt',
                        help='YOLO模型路径 (默认: ./checkpoints/yolo26x.pt)')
    parser.add_argument('--output', '-o', type=str, default='./output_passenger_count.mp4',
                        help='输出视频路径')
    parser.add_argument('--crops-dir', type=str, default='./person_crops_count',
                        help='行人截图保存目录 (默认: ./person_crops_count)')
    parser.add_argument('--no-save-crops', action='store_true',
                        help='禁用行人截图保存')

    # 日志参数
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='日志保存目录 (默认: ./logs)')

    # 检测参数
    parser.add_argument('--person-conf', type=float, default=0.25,
                        help='行人检测置信度阈值 (默认: 0.25)')
    parser.add_argument('--vehicle-conf', type=float, default=0.25,
                        help='车辆检测置信度阈值 (默认: 0.25)')
    parser.add_argument('--vehicle-person-dist', type=int, default=80,
                        help='车人距离阈值，单位像素 (默认: 100)')

    # ROI区域参数
    parser.add_argument('--center-region-ratio', type=float, default=0.6,
                        help='中心检测区域占画面比例 (0-1, 默认0.6)')
    parser.add_argument('--center-region-ratio-w', type=float, default=0.9,
                        help='中心检测区域宽度占画面比例 (默认0.9)')
    parser.add_argument('--center-region-ratio-h', type=float, default=0.6,
                        help='中心检测区域高度占画面比例 (默认0.6)')

    # 跟踪参数
    parser.add_argument('--person-track-dist', type=int, default=150,
                        help='行人跟踪距离阈值 (默认: 150)')
    parser.add_argument('--person-max-age', type=int, default=25,
                        help='行人最大丢失帧数 (默认: 25)')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='IOU匹配阈值 (默认: 0.25)')

    # 上下车检测参数
    parser.add_argument('--min-frames-before-exit', type=int, default=3,
                        help='最少存在帧数，低于此值的行人不统计上下车 (默认: 3)')
    parser.add_argument('--strict-boarding-distance', type=int, default=30,
                        help='严格上车距离阈值，只有行人与车中心点距离小于此值且消失才算上车，单位像素 (默认: 30)')
    parser.add_argument('--alighting-distance', type=int, default=15,
                        help='严格下车距离阈值，只有行人与车中心点距离小于此值且前一帧车辆无人检测才计算下车，单位像素 (默认: 15)')

    # 重识别参数
    parser.add_argument('--reid-threshold', type=float, default=0.7,
                        help='余弦相似度阈值 (默认: 0.7)')
    parser.add_argument('--reid-history', type=int, default=40,
                        help='重识别历史帧数 (默认: 40)')
    parser.add_argument('--reid-min-absent-frames', type=int, default=None,
                        help='重识别最小消失帧数，默认3秒对应的帧数 (默认: None=自动计算)')
    parser.add_argument('--reid-absent-seconds', type=int, default=8,
                        help='使用重识别所需的消失秒数阈值，只有消失超过此时长或离开ROI才启用相似度匹配 (默认: 8)')

    # 其他
    parser.add_argument('--no-display', action='store_true',
                        help='不显示实时窗口')
    parser.add_argument('--save-video', action='store_true',
                        help='保存输出视频')

    return parser.parse_args()


def process_video(args):
    """处理视频主流程"""
    # 设置日志
    log_file = setup_logging(args.log_dir)
    logging.info("=" * 60)
    logging.info("乘客上下车统计系统启动")
    logging.info(f"视频: {args.video}")
    logging.info(f"模型: {args.model}")
    logging.info("=" * 60)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logging.error(f"错误: 无法打开视频: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(f"视频信息: {width}x{height}@{fps:.1f}fps, 共{total_frames}帧")

    detector = PassengerCountDetector(args)
    detector.set_fps(fps)

    # 准备输出
    out = None
    if args.save_video:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        logging.info(f"输出视频: {args.output}")

    # 准备截图保存
    save_crops = not args.no_save_crops
    if save_crops:
        os.makedirs(args.crops_dir, exist_ok=True)
        logging.info(f"截图保存到: {args.crops_dir}")

    # 第一帧：初始化
    ret, first_frame = cap.read()
    if not ret:
        logging.error("错误: 无法读取第一帧")
        return

    detector.frame_height, detector.frame_width = first_frame.shape[:2]
    detector.get_center_roi()
    logging.info(f"ROI区域: {detector.center_roi}")

    # 处理第一帧
    tracked_persons, nearby_vehicles = detector.detect_and_track_with_counting(first_frame, 1)
    annotated = detector.draw_annotations_with_counting(
        first_frame.copy(), tracked_persons, nearby_vehicles, 1
    )

    frame_count = 1
    total_crops_saved = 0

    if save_crops:
        for x1, y1, x2, y2, pid, color, crop in tracked_persons:
            if save_person_crop(crop, pid, 1, args.crops_dir):
                total_crops_saved += 1

    if out:
        out.write(annotated)
    if not args.no_display:
        cv2.imshow("Passenger Count", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            return

    logging.info("开始处理视频 (按 'q' 退出)...")

    # 处理后续帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        tracked_persons, nearby_vehicles = detector.detect_and_track_with_counting(frame, frame_count)
        annotated = detector.draw_annotations_with_counting(
            frame.copy(), tracked_persons, nearby_vehicles, frame_count
        )

        if save_crops:
            for x1, y1, x2, y2, pid, color, crop in tracked_persons:
                if save_person_crop(crop, pid, frame_count, args.crops_dir):
                    total_crops_saved += 1

        if out:
            out.write(annotated)
        if not args.no_display:
            cv2.imshow("Passenger Count", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("用户中断处理")
                break

        if frame_count % 30 == 0:
            progress = f"进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)"
            logging.info(progress)

    # 视频结束，处理未完成的跟踪
    detector.passenger_counter.finalize(frame_count)

    # 保存统计结果
    stats_filename = f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    stats_path = Path(args.log_dir) / stats_filename
    vehicle_stats = detector.passenger_counter.save_statistics(stats_path)

    # 输出最终报告
    total_boarding = len(detector.passenger_counter.boarded_persons)
    total_alighting = len(detector.passenger_counter.alighted_persons)

    logging.info("\n" + "=" * 60)
    logging.info("乘客上下车统计报告")
    logging.info("=" * 60)
    logging.info(f"总帧数: {frame_count}")
    logging.info(f"检测到唯一行人数: {detector.person_tracker.next_id - 1}")
    logging.info(f"上车人数（去重）: {total_boarding}")
    logging.info(f"下车人数（去重）: {total_alighting}")
    logging.info(f"严格上车距离阈值: {args.strict_boarding_distance}px")
    logging.info(f"严格下车距离阈值: {args.alighting_distance}px（前一帧车辆必须无人）")

    # 按车辆输出统计
    if vehicle_stats:
        logging.info("\n各车辆上下车统计:")
        for vehicle_key, data in vehicle_stats.items():
            boarding_list = data.get("上车行人", [])
            alighting_list = data.get("下车行人", [])
            logging.info(f"  {vehicle_key}:")
            logging.info(f"    上车行人: {boarding_list if boarding_list else '无'}")
            logging.info(f"    下车行人: {alighting_list if alighting_list else '无'}")
    else:
        logging.info("\n未检测到上下车事件")

    logging.info("=" * 60)
    logging.info(f"详细日志: {log_file}")
    logging.info(f"统计数据: {stats_path}")
    logging.info("=" * 60)

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