#!/usr/bin/env python3
import math
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def normalize_angle(angle: float) -> float:
    """Приводит угол в диапазон [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Переводит плоский угол yaw в кватернион (x, y, z, w)."""
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def voxel_downsample(pts: np.ndarray, voxel_size: float = 0.04) -> np.ndarray:
    """Оставляет не более одной точки на квадратик voxel_size x voxel_size."""
    if len(pts) == 0:
        return pts
    coords = np.floor(pts / voxel_size).astype(np.int32)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return pts[idx]


def transform_points(pts: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    """Поворачивает и сдвигает массив 2D-точек (N, 2)."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return pts @ rot.T + np.array([x, y], dtype=np.float64)


class LidarImuOdomNode(Node):
    def __init__(self):
        # Работаем строго по симуляционному времени (/clock)
        super().__init__(
            "lidar_imu_odom",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True)
            ],
        )

        # Смещение лидара (base_scan) относительно корпуса (trunk) из /tf_static
        self.lidar_offset_x = 0.22
        self.lidar_offset_y = 0.0

        # Текущая оценка позы trunk в системе odom (обновляется на 100 Гц и корректируется на 10 Гц)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Последняя поза, подтверждённая лидаром (ICP)
        self.last_icp_x = 0.0
        self.last_icp_y = 0.0
        self.last_icp_yaw = 0.0

        # Оценка скорости корпуса
        self.vx_body = 0.0
        self.vy_body = 0.0
        self.wz = 0.0
        self.is_stationary = True

        # Временные метки (строго монотонные для TF2)
        self.last_scan_time: float | None = None
        self.last_imu_time: float | None = None
        self.latest_imu_wz = 0.0
        self.last_published_stamp_ns: int = -1

        # Локальная карта из ключевых сканов
        self.initialized = False
        self.keyframes: deque[np.ndarray] = deque(maxlen=30)
        self.local_map_pts: np.ndarray | None = None
        self.local_map_tree: cKDTree | None = None
        self.last_kf_x = 0.0
        self.last_kf_y = 0.0
        self.last_kf_yaw = 0.0

        # QoS строго под RELIABLE-издателей симулятора
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.odom_pub = self.create_publisher(Odometry, "/odom", odom_qos)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.imu_sub = self.create_subscription(
            Imu, "/imu/data", self.imu_callback, imu_qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, scan_qos
        )

        self.get_logger().info("Lidar+IMU Odometry node started (use_sim_time=True)")

    def imu_callback(self, msg: Imu):
        t = stamp_to_sec(msg.header.stamp)
        stamp_ns = stamp_to_ns(msg.header.stamp)
        if t <= 0.0:
            return

        wz = float(msg.angular_velocity.z)
        if not math.isfinite(wz):
            return
        self.latest_imu_wz = wz

        # Интегрируем движение между сканами лидара, только если карта готова и робот не в покое
        if self.initialized and self.last_imu_time is not None and not self.is_stationary:
            dt = t - self.last_imu_time
            if 0.0 < dt < 0.2:
                wz_used = wz if abs(wz) > 0.003 else 0.0
                yaw_mid = self.yaw + 0.5 * wz_used * dt
                c = math.cos(yaw_mid)
                s = math.sin(yaw_mid)
                self.x += (self.vx_body * c - self.vy_body * s) * dt
                self.y += (self.vx_body * s + self.vy_body * c) * dt
                self.yaw = normalize_angle(self.yaw + wz_used * dt)

        self.last_imu_time = t

        # Публикуем /odom и /tf не чаще ~50 Гц (каждые >= 19 мс по sim-time),
        # что с большим запасом выше требуемых 20 Гц.
        if (
            self.last_published_stamp_ns < 0
            or (stamp_ns - self.last_published_stamp_ns) >= 9_000_000
        ):
            self.publish_odometry(msg.header.stamp, stamp_ns)

    def extract_scan_points_trunk(self, msg: LaserScan) -> np.ndarray:
        """Преобразует скан из base_scan в 2D-точки в системе координат корпуса trunk."""
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        n = len(ranges)
        if n == 0:
            return np.empty((0, 2), dtype=np.float64)

        angles = msg.angle_min + np.arange(n, dtype=np.float64) * msg.angle_increment
        r_min = max(float(msg.range_min), 0.20)
        r_max = min(float(msg.range_max) - 0.5, 19.0)
        valid = np.isfinite(ranges) & (ranges >= r_min) & (ranges <= r_max)

        if not np.any(valid):
            return np.empty((0, 2), dtype=np.float64)

        r = ranges[valid]
        a = angles[valid]

        x_trunk = r * np.cos(a) + self.lidar_offset_x
        y_trunk = r * np.sin(a) + self.lidar_offset_y

        # На всякий случай убираем точки ближе 0.35 м от центра корпуса, чтобы не цеплять части самого робота
        outside_body = (np.abs(x_trunk) > 0.35) | (np.abs(y_trunk) > 0.22)
        pts = np.column_stack((x_trunk[outside_body], y_trunk[outside_body]))
        return voxel_downsample(pts, voxel_size=0.04)

    def rebuild_local_map(self):
        all_pts = np.vstack(list(self.keyframes))
        self.local_map_pts = voxel_downsample(all_pts, voxel_size=0.04)
        self.local_map_tree = cKDTree(self.local_map_pts)

    def run_icp(
        self, pts_trunk: np.ndarray, init_x: float, init_y: float, init_yaw: float
    ) -> tuple[float, float, float]:
        """Уточняет позу (x, y, yaw) методом weighted SVD ICP относительно локальной карты."""
        x, y, yaw = init_x, init_y, init_yaw
        tree = self.local_map_tree
        map_pts = self.local_map_pts

        if tree is None or map_pts is None or len(map_pts) < 20:
            return x, y, yaw

        for it in range(15):
            max_dist = 0.35 if it < 4 else 0.15
            pts_world = transform_points(pts_trunk, x, y, yaw)
            dists, indices = tree.query(pts_world, k=1)

            mask = dists < max_dist
            if np.count_nonzero(mask) < 15:
                break

            src = pts_world[mask]
            dst = map_pts[indices[mask]]
            d = dists[mask]

            # Веса Хьюбера для устойчивости к выбросам
            huber_delta = 0.05
            w = np.where(d <= huber_delta, 1.0, huber_delta / np.maximum(d, 1e-6))
            w_sum = float(np.sum(w))
            if w_sum <= 1e-6:
                break

            mu_src = np.sum(w[:, None] * src, axis=0) / w_sum
            mu_dst = np.sum(w[:, None] * dst, axis=0) / w_sum

            src_c = src - mu_src
            dst_c = dst - mu_dst

            cov = (w[:, None] * src_c).T @ dst_c
            u, _, vt = np.linalg.svd(cov)
            rot = vt.T @ u.T
            if np.linalg.det(rot) < 0:
                vt[-1, :] *= -1.0
                rot = vt.T @ u.T

            dyaw = math.atan2(rot[1, 0], rot[0, 0])
            dt_vec = mu_dst - rot @ mu_src

            t_new = rot @ np.array([x, y], dtype=np.float64) + dt_vec
            x_new = float(t_new[0])
            y_new = float(t_new[1])
            yaw_new = normalize_angle(yaw + dyaw)

            step_trans = math.hypot(x_new - x, y_new - y)
            x, y, yaw = x_new, y_new, yaw_new

            if step_trans < 1e-4 and abs(dyaw) < 1e-4:
                break

        return x, y, yaw

    def scan_callback(self, msg: LaserScan):
        t_sec = stamp_to_sec(msg.header.stamp)
        if t_sec <= 0.0:
            return

        pts_trunk = self.extract_scan_points_trunk(msg)
        if len(pts_trunk) < 20:
            return

        if not self.initialized:
            self.initialized = True
            self.x = 0.0
            self.y = 0.0
            self.yaw = 0.0
            self.last_icp_x = 0.0
            self.last_icp_y = 0.0
            self.last_icp_yaw = 0.0
            self.last_scan_time = t_sec
            self.last_kf_x = 0.0
            self.last_kf_y = 0.0
            self.last_kf_yaw = 0.0
            self.keyframes.append(transform_points(pts_trunk, 0.0, 0.0, 0.0))
            self.rebuild_local_map()
            return

        dt = t_sec - (self.last_scan_time if self.last_scan_time is not None else t_sec)
        self.last_scan_time = t_sec

        # Стартуем ICP из текущей позы (она уже обновлена по гироскопу IMU)
        x_new, y_new, yaw_new = self.run_icp(pts_trunk, self.x, self.y, self.yaw)

        step_dist = math.hypot(x_new - self.last_icp_x, y_new - self.last_icp_y)
        step_yaw = abs(normalize_angle(yaw_new - self.last_icp_yaw))

        # Защита от дрейфа в покое: если сдвиг за 0.1 с меньше 2 мм и < 0.002 рад
        if (
            step_dist < 0.002
            and step_yaw < 0.002
            and abs(self.latest_imu_wz) < 0.015
        ):
            self.is_stationary = True
            self.vx_body = 0.0
            self.vy_body = 0.0
            self.wz = 0.0
            self.x = self.last_icp_x
            self.y = self.last_icp_y
            self.yaw = self.last_icp_yaw
        else:
            self.is_stationary = False
            if 0.0 < dt < 0.35:
                dx_w = x_new - self.last_icp_x
                dy_w = y_new - self.last_icp_y
                c = math.cos(yaw_new)
                s = math.sin(yaw_new)
                vx_inst = (dx_w * c + dy_w * s) / dt
                vy_inst = (-dx_w * s + dy_w * c) / dt
                wz_inst = normalize_angle(yaw_new - self.last_icp_yaw) / dt
                alpha = 0.4
                self.vx_body = (1.0 - alpha) * self.vx_body + alpha * vx_inst
                self.vy_body = (1.0 - alpha) * self.vy_body + alpha * vy_inst
                self.wz = (1.0 - alpha) * self.wz + alpha * wz_inst

            self.x = x_new
            self.y = y_new
            self.yaw = yaw_new
            self.last_icp_x = x_new
            self.last_icp_y = y_new
            self.last_icp_yaw = yaw_new

            kf_dist = math.hypot(self.x - self.last_kf_x, self.y - self.last_kf_y)
            kf_yaw = abs(normalize_angle(self.yaw - self.last_kf_yaw))
            if kf_dist > 0.25 or kf_yaw > 0.20:
                self.keyframes.append(
                    transform_points(pts_trunk, self.x, self.y, self.yaw)
                )
                self.rebuild_local_map()
                self.last_kf_x = self.x
                self.last_kf_y = self.y
                self.last_kf_yaw = self.yaw

    def publish_odometry(self, stamp, stamp_ns: int):
        if stamp_ns <= self.last_published_stamp_ns:
            return

        vals = (self.x, self.y, self.yaw, self.vx_body, self.vy_body, self.wz)
        if not all(math.isfinite(v) for v in vals):
            self.get_logger().warn("Non-finite odometry detected, skipping publish")
            return

        qx, qy, qz, qw = yaw_to_quaternion(self.yaw)

        # 1. Публикуем TF: odom -> trunk
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = "odom"
        tf_msg.child_frame_id = "trunk"
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf_msg)

        # 2. Публикуем абсолютно ту же позу в /odom
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "trunk"
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.twist.twist.linear.x = self.vx_body
        odom_msg.twist.twist.linear.y = self.vy_body
        odom_msg.twist.twist.angular.z = self.wz
        self.odom_pub.publish(odom_msg)

        self.last_published_stamp_ns = stamp_ns


def main(args=None):
    rclpy.init(args=args)
    node = LidarImuOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
