"""
file: donkey_sim.py
author: Tawn Kramer
date: 2018-08-31
"""
import json
import base64
import logging
import math
import os
import time
import types
from io import BytesIO
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from gym_donkeycar.core.fps import FPSTimer
from gym_donkeycar.core.message import IMesgHandler
from gym_donkeycar.core.sim_client import SimClient

logger = logging.getLogger(__name__)


# Math helpers added by CireNeikual (222464)
def euler_to_quat(e):
    cx = np.cos(e[0] * 0.5)
    sx = np.sin(e[0] * 0.5)
    cy = np.cos(e[1] * 0.5)
    sy = np.sin(e[1] * 0.5)
    cz = np.cos(e[2] * 0.5)
    sz = np.sin(e[2] * 0.5)

    x = sz * cx * cy - cz * sx * sy
    y = cz * sx * cy + sz * cx * sy
    z = cz * cx * sy - sz * sx * cy
    w = cz * cx * cy + sz * sx * sy

    return [x, y, z, w]


def cross(v0, v1):
    return [v0[1] * v1[2] - v0[2] * v1[1], v0[2] * v1[0] - v0[0] * v1[2], v0[0] * v1[1] - v0[1] * v1[0]]


def rotate_vec(q, v):
    uv = cross(q[0:3], v)
    uuv = cross(q[0:3], uv)

    scaleUv = 2.0 * q[3]

    uv[0] *= scaleUv
    uv[1] *= scaleUv
    uv[2] *= scaleUv

    uuv[0] *= 2.0
    uuv[1] *= 2.0
    uuv[2] *= 2.0

    return [v[0] + uv[0] + uuv[0], v[1] + uv[1] + uuv[1], v[2] + uv[2] + uuv[2]]


class DonkeyUnitySimContoller:
    def __init__(self, conf: Dict[str, Any]):
        logger.setLevel(conf["log_level"])

        self.address = (conf["host"], conf["port"])

        self.handler = DonkeyUnitySimHandler(conf=conf)

        self.client = SimClient(self.address, self.handler)

    def set_car_config(
        self,
        body_style: str,
        body_rgb: Tuple[int, int, int],
        car_name: str,
        font_size: int,
    ) -> None:
        self.handler.send_car_config(body_style, body_rgb, car_name, font_size)

    def set_cam_config(self, **kwargs) -> None:
        self.handler.send_cam_config(**kwargs)

    def set_reward_fn(self, reward_fn: Callable) -> None:
        self.handler.set_reward_fn(reward_fn)

    def set_episode_over_fn(self, ep_over_fn: Callable) -> None:
        self.handler.set_episode_over_fn(ep_over_fn)

    def wait_until_loaded(self) -> None:
        time.sleep(0.1)
        while not self.handler.loaded:
            logger.warning("waiting for sim to start..")
            time.sleep(1.0)
        logger.info("sim started!")

    def reset(self) -> None:
        self.handler.reset()

    def get_sensor_size(self) -> Tuple[int, int, int]:
        return self.handler.get_sensor_size()

    def take_action(self, action: np.ndarray):
        self.handler.take_action(action)

    def observe(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        return self.handler.observe()

    def quit(self) -> None:
        self.client.stop()

    def exit_scene(self) -> None:
        self.handler.send_exit_scene()

    def render(self, mode: str) -> None:
        pass

    def is_game_over(self) -> bool:
        return self.handler.is_game_over()

    def calc_reward(self, done: bool) -> float:
        return self.handler.calc_reward(done)


class DonkeyUnitySimHandler(IMesgHandler):
    def __init__(self, conf: Dict[str, Any]):
        self.conf = conf
        self.SceneToLoad = conf["level"]
        self.loaded = False
        self.max_cte = conf["max_cte"]
        self.timer = FPSTimer()

        # sensor size - height, width, depth
        self.camera_img_size = conf["cam_resolution"]
        self.image_array = np.zeros(self.camera_img_size)
        self.image_array_b = None
        self.last_obs = self.image_array
        self.time_received = time.time()
        self.last_received = self.time_received
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.speed = 0.0
        self.forward_vel = 0.0
        self.missed_checkpoint = False
        self.dq = False
        self.over = False
        self.client = None
        self.fns = {
            "telemetry": self.on_telemetry,
            "scene_selection_ready": self.on_scene_selection_ready,
            "scene_names": self.on_recv_scene_names,
            "car_loaded": self.on_car_loaded,
            "cross_start": self.on_cross_start,
            "race_start": self.on_race_start,
            "race_stop": self.on_race_stop,
            "DQ": self.on_DQ,
            "ping": self.on_ping,
            "aborted": self.on_abort,
            "missed_checkpoint": self.on_missed_checkpoint,
            "need_car_config": self.on_need_car_config,
            "collision_with_starting_line": self.on_collision_with_starting_line,
        }
        self.gyro_x = 0.0
        self.gyro_y = 0.0
        self.gyro_z = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.lidar = []

        # car in Unity lefthand coordinate system: roll is Z, pitch is X and yaw is Y
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # variables required for lidar points decoding into array format
        self.lidar_deg_per_sweep_inc = 1
        self.lidar_num_sweep_levels = 1
        self.lidar_deg_ang_delta = 1

        self.last_lap_time = 0.0
        self.current_lap_time = 0.0
        self.starting_line_index = -1
        self.lap_count = 0

    def on_connect(self, client: SimClient) -> None:  # pytype: disable=signature-mismatch
        logger.debug("socket connected")
        self.client = client

    def on_disconnect(self) -> None:
        logger.debug("socket disconnected")
        self.client = None

    def on_abort(self, message: Dict[str, Any]) -> None:
        self.client.stop()

    def on_need_car_config(self, message: Dict[str, Any]) -> None:
        logger.info("on need car config")
        self.loaded = True
        self.send_config(self.conf)

    def on_collision_with_starting_line(self, message: Dict[str, Any]) -> None:
        if self.current_lap_time == 0.0:
            self.current_lap_time = message["timeStamp"]
            self.starting_line_index = message["starting_line_index"]
        elif self.starting_line_index == message["starting_line_index"]:
            time_at_crossing = message["timeStamp"]
            self.last_lap_time = float(time_at_crossing - self.current_lap_time)
            self.current_lap_time = time_at_crossing
            self.lap_count += 1
            lap_msg = f"New lap time: {round(self.last_lap_time, 2)} seconds"
            logger.info(lap_msg)

    @staticmethod
    def extract_keys(dict_: Dict[str, Any], list_: List[str]) -> Dict[str, Any]:
        return_dict = {}
        for key in list_:
            if key in dict_:
                return_dict[key] = dict_[key]
        return return_dict

    def send_config(self, conf: Dict[str, Any]) -> None:
        if "degPerSweepInc" in conf:
            raise ValueError("LIDAR config keys were renamed to use snake_case name instead of CamelCase")

        logger.info("sending car config.")
        # both ways work, car_config shouldn't interfere with other config, so keeping the two alternative
        self.set_car_config(conf)
        if "car_config" in conf.keys():
            self.set_car_config(conf["car_config"])
            logger.info("done sending car config.")

        if "cam_config" in conf.keys():
            cam_config = self.extract_keys(
                conf["cam_config"],
                [
                    "img_w",
                    "img_h",
                    "img_d",
                    "img_enc",
                    "fov",
                    "fish_eye_x",
                    "fish_eye_y",
                    "offset_x",
                    "offset_y",
                    "offset_z",
                    "rot_x",
                    "rot_y",
                    "rot_z",
                ],
            )
            self.send_cam_config(**cam_config)
            logger.info(f"done sending cam config. {cam_config}")

        if "cam_config_b" in conf.keys():
            cam_config_b = self.extract_keys(
                conf["cam_config_b"],
                [
                    "img_w",
                    "img_h",
                    "img_d",
                    "img_enc",
                    "fov",
                    "fish_eye_x",
                    "fish_eye_y",
                    "offset_x",
                    "offset_y",
                    "offset_z",
                    "rot_x",
                    "rot_y",
                    "rot_z",
                ],
            )
            self.send_cam_config(**cam_config_b, msg_type="cam_config_b")
            logger.info(f"done sending cam config B. {cam_config_b}")
            self.image_array_b = np.zeros(self.camera_img_size)

        if "lidar_config" in conf.keys():
            if "degPerSweepInc" in conf:
                raise ValueError("LIDAR config keys were renamed to use snake_case name instead of CamelCase")

            lidar_config = self.extract_keys(
                conf["lidar_config"],
                [
                    "deg_per_sweep_inc",
                    "deg_ang_down",
                    "deg_ang_delta",
                    "num_sweeps_levels",
                    "max_range",
                    "noise",
                    "offset_x",
                    "offset_y",
                    "offset_z",
                    "rot_x",
                ],
            )
            self.send_lidar_config(**lidar_config)
            logger.info(f"done sending lidar config., {lidar_config}")

        # what follows is needed in order not to break older conf

        cam_config = self.extract_keys(
            conf,
            [
                "img_w",
                "img_h",
                "img_d",
                "img_enc",
                "fov",
                "fish_eye_x",
                "fish_eye_y",
                "offset_x",
                "offset_y",
                "offset_z",
                "rot_x",
                "rot_y",
                "rot_z",
            ],
        )
        if cam_config != {}:
            self.send_cam_config(**cam_config)
            logger.info(f"done sending cam config. {cam_config}")
            logger.warning(
                """This way of passing cam_config is deprecated,
                please wrap the parameters in a sub-dictionary with the key 'cam_config'.
                Example: GYM_CONF = {'cam_config':"""
                + str(cam_config)
                + "}"
            )

        lidar_config = self.extract_keys(
            conf,
            [
                "deg_per_sweep_inc",
                "deg_ang_down",
                "deg_ang_delta",
                "num_sweeps_levels",
                "max_range",
                "noise",
                "offset_x",
                "offset_y",
                "offset_z",
                "rot_x",
            ],
        )
        if lidar_config != {}:
            self.send_lidar_config(**lidar_config)
            logger.info(f"done sending lidar config., {lidar_config}")
            logger.warning(
                """This way of passing lidar_config is deprecated,
                please wrap the parameters in a sub-dictionary with the key 'lidar_config'.
                Example: GYM_CONF = {'lidar_config':"""
                + str(lidar_config)
                + "}"
            )

    def set_car_config(self, conf: Dict[str, Any]) -> None:
        if "body_style" in conf:
            self.send_car_config(
                conf["body_style"],
                conf["body_rgb"],
                conf["car_name"],
                conf["font_size"],
            )

    def set_racer_bio(self, conf: Dict[str, Any]) -> None:
        if "bio" in conf:
            self.send_racer_bio(
                conf["racer_name"],
                conf["car_name"],
                conf["bio"],
                conf["country"],
                conf["guid"],
            )

    def on_recv_message(self, message: Dict[str, Any]) -> None:
        if "msg_type" not in message:
            logger.warn("expected msg_type field")
            return
        msg_type = message["msg_type"]
        logger.debug("got message :" + msg_type)
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            logger.warning(f"unknown message type {msg_type}")

    def send_pose(self,tracked_pose_sim):
        msg = { 'msg_type' : 'tracking', 'x': tracked_pose_sim[0].__str__(), 'y':tracked_pose_sim[1].__str__(), 'angle': tracked_pose_sim[2].__str__() }
        self.client.send_now(msg)
        time.sleep(0.04)

    # ------- Env interface ---------- #



    def reset(self) -> None:
        #print("reset RRRRRRRRRRRRRRRRRR")
        #ROAD = '48.00,0.60,50.00@48.01,0.60,50.31@48.03,0.60,50.63@48.06,0.60,50.95@48.10,0.60,51.26@48.15,0.60,51.59@48.20,0.60,51.91@48.25,0.60,52.24@48.30,0.60,52.58@48.35,0.60,52.92@48.39,0.60,53.27@48.43,0.60,53.63@48.45,0.60,54.00@48.46,0.60,54.37@48.46,0.60,54.76@48.44,0.60,55.15@48.40,0.60,55.56@48.34,0.60,55.98@48.25,0.60,56.42@48.14,0.60,56.87@48.00,0.60,57.33@47.83,0.60,57.81@47.63,0.60,58.30@47.39,0.60,58.80@47.14,0.60,59.30@46.85,0.60,59.80@46.55,0.60,60.30@46.22,0.60,60.79@45.87,0.60,61.27@45.50,0.60,61.73@45.12,0.60,62.17@44.72,0.60,62.59@44.31,0.60,62.98@43.88,0.60,63.35@43.44,0.60,63.67@42.99,0.60,63.96@42.54,0.60,64.20@42.08,0.60,64.39@41.61,0.60,64.54@41.14,0.60,64.63@40.67,0.60,64.66@40.20,0.60,64.63@39.73,0.60,64.54@39.26,0.60,64.39@38.80,0.60,64.20@38.35,0.60,63.96@37.90,0.60,63.67@37.46,0.60,63.35@37.03,0.60,62.98@36.62,0.60,62.59@36.22,0.60,62.17@35.84,0.60,61.73@35.47,0.60,61.27@35.12,0.60,60.79@34.79,0.60,60.30@34.49,0.60,59.80@34.20,0.60,59.30@33.95,0.60,58.80@33.71,0.60,58.30@33.51,0.60,57.81@33.34,0.60,57.33@33.20,0.60,56.87@33.09,0.60,56.42@33.00,0.60,55.98@32.94,0.60,55.56@32.90,0.60,55.15@32.88,0.60,54.76@32.88,0.60,54.37@32.89,0.60,54.00@32.91,0.60,53.63@32.95,0.60,53.27@32.99,0.60,52.92@33.04,0.60,52.58@33.09,0.60,52.24@33.14,0.60,51.91@33.19,0.60,51.59@33.24,0.60,51.26@33.28,0.60,50.95@33.31,0.60,50.63@33.33,0.60,50.31@33.34,0.60,50.00@33.33,0.60,49.69@33.31,0.60,49.37@33.28,0.60,49.05@33.24,0.60,48.74@33.19,0.60,48.41@33.14,0.60,48.09@33.09,0.60,47.76@33.04,0.60,47.42@32.99,0.60,47.08@32.95,0.60,46.73@32.91,0.60,46.37@32.89,0.60,46.00@32.88,0.60,45.63@32.88,0.60,45.24@32.90,0.60,44.85@32.94,0.60,44.44@33.00,0.60,44.02@33.09,0.60,43.58@33.20,0.60,43.13@33.34,0.60,42.67@33.51,0.60,42.19@33.71,0.60,41.70@33.95,0.60,41.20@34.20,0.60,40.70@34.49,0.60,40.20@34.79,0.60,39.70@35.12,0.60,39.21@35.47,0.60,38.73@35.84,0.60,38.27@36.22,0.60,37.83@36.62,0.60,37.41@37.03,0.60,37.02@37.46,0.60,36.65@37.90,0.60,36.33@38.35,0.60,36.04@38.80,0.60,35.80@39.26,0.60,35.61@39.73,0.60,35.46@40.20,0.60,35.37@40.67,0.60,35.34@41.14,0.60,35.37@41.61,0.60,35.46@42.08,0.60,35.61@42.54,0.60,35.80@42.99,0.60,36.04@43.44,0.60,36.33@43.88,0.60,36.65@44.31,0.60,37.02@44.72,0.60,37.41@45.12,0.60,37.83@45.50,0.60,38.27@45.87,0.60,38.73@46.22,0.60,39.21@46.55,0.60,39.70@46.85,0.60,40.20@47.14,0.60,40.70@47.39,0.60,41.20@47.63,0.60,41.70@47.83,0.60,42.19@48.00,0.60,42.67@48.14,0.60,43.13@48.25,0.60,43.58@48.34,0.60,44.02@48.40,0.60,44.44@48.44,0.60,44.85@48.46,0.60,45.24@48.46,0.60,45.63@48.45,0.60,46.00@48.43,0.60,46.37@48.39,0.60,46.73@48.35,0.60,47.08@48.30,0.60,47.42@48.25,0.60,47.76@48.20,0.60,48.09@48.15,0.60,48.41@48.10,0.60,48.74@48.06,0.60,49.05@48.03,0.60,49.37@48.01,0.60,49.69'
        #ROAD = '45.80,0.60,38.27@45.60,0.60,38.37@45.40,0.60,38.49@45.21,0.60,38.63@45.03,0.60,38.78@44.85,0.60,38.94@44.67,0.60,39.11@44.50,0.60,39.29@44.32,0.60,39.48@44.15,0.60,39.68@43.98,0.60,39.89@43.81,0.60,40.10@43.64,0.60,40.31@43.47,0.60,40.52@43.29,0.60,40.73@43.12,0.60,40.95@42.93,0.60,41.15@42.74,0.60,41.36@42.55,0.60,41.56@42.35,0.60,41.75@42.14,0.60,41.94@41.92,0.60,42.11@41.69,0.60,42.28@41.46,0.60,42.44@41.23,0.60,42.60@40.99,0.60,42.75@40.74,0.60,42.90@40.50,0.60,43.05@40.25,0.60,43.20@40.01,0.60,43.35@39.77,0.60,43.50@39.53,0.60,43.66@39.30,0.60,43.83@39.07,0.60,44.01@38.85,0.60,44.19@38.64,0.60,44.39@38.44,0.60,44.60@38.24,0.60,44.83@38.06,0.60,45.07@37.89,0.60,45.33@37.74,0.60,45.60@37.60,0.60,45.90@37.47,0.60,46.21@37.36,0.60,46.55@37.26,0.60,46.90@37.18,0.60,47.26@37.11,0.60,47.64@37.04,0.60,48.03@36.99,0.60,48.43@36.95,0.60,48.85@36.92,0.60,49.27@36.90,0.60,49.70@36.89,0.60,50.13@36.88,0.60,50.57@36.88,0.60,51.01@36.89,0.60,51.46@36.90,0.60,51.90@36.92,0.60,52.35@36.95,0.60,52.79@36.97,0.60,53.23@37.00,0.60,53.66@37.04,0.60,54.09@37.08,0.60,54.52@37.12,0.60,54.93@37.16,0.60,55.34@37.21,0.60,55.74@37.27,0.60,56.13@37.33,0.60,56.51@37.39,0.60,56.89@37.47,0.60,57.24@37.55,0.60,57.59@37.64,0.60,57.93@37.73,0.60,58.25@37.84,0.60,58.56@37.95,0.60,58.85@38.07,0.60,59.13@38.20,0.60,59.39@38.34,0.60,59.63@38.50,0.60,59.86@38.66,0.60,60.07@38.84,0.60,60.26@39.03,0.60,60.43@39.22,0.60,60.59@39.44,0.60,60.72@39.66,0.60,60.85@39.89,0.60,60.95@40.13,0.60,61.05@40.38,0.60,61.13@40.64,0.60,61.20@40.91,0.60,61.26@41.19,0.60,61.32@41.48,0.60,61.37@41.77,0.60,61.41@42.07,0.60,61.45@42.38,0.60,61.49@42.69,0.60,61.52@43.01,0.60,61.56@43.33,0.60,61.60@43.66,0.60,61.64@44.00,0.60,61.68@44.34,0.60,61.73@44.68,0.60,61.78@45.02,0.60,61.84@45.37,0.60,61.90@45.72,0.60,61.97@46.07,0.60,62.03@46.43,0.60,62.09@46.79,0.60,62.15@47.15,0.60,62.20@47.52,0.60,62.24@47.89,0.60,62.28@48.26,0.60,62.30@48.63,0.60,62.31@49.00,0.60,62.31@49.38,0.60,62.29@49.76,0.60,62.25@50.14,0.60,62.19@50.52,0.60,62.11@50.90,0.60,62.01@51.28,0.60,61.88@51.66,0.60,61.73@52.05,0.60,61.55@52.43,0.60,61.34@52.82,0.60,61.11@53.19,0.60,60.85@53.57,0.60,60.58@53.93,0.60,60.29@54.29,0.60,59.98@54.64,0.60,59.66@54.98,0.60,59.33@55.30,0.60,58.99@55.61,0.60,58.64@55.91,0.60,58.28@56.19,0.60,57.92@56.45,0.60,57.56@56.69,0.60,57.20@56.90,0.60,56.85@57.10,0.60,56.50@57.27,0.60,56.15@57.41,0.60,55.82@57.53,0.60,55.50@57.62,0.60,55.19@57.68,0.60,54.89@57.72,0.60,54.60@57.73,0.60,54.32@57.73,0.60,54.05@57.71,0.60,53.79@57.68,0.60,53.53@57.63,0.60,53.28@57.58,0.60,53.03@57.52,0.60,52.77@57.46,0.60,52.52@57.40,0.60,52.27@57.34,0.60,52.01@57.28,0.60,51.75@57.23,0.60,51.48@57.19,0.60,51.20@57.16,0.60,50.92@57.14,0.60,50.63@57.14,0.60,50.32@57.16,0.60,50.00@57.20,0.60,49.67@57.27,0.60,49.32@57.34,0.60,48.96@57.43,0.60,48.60@57.53,0.60,48.22@57.64,0.60,47.84@57.75,0.60,47.45@57.87,0.60,47.06@57.98,0.60,46.67@58.08,0.60,46.28@58.18,0.60,45.88@58.27,0.60,45.50@58.35,0.60,45.11@58.41,0.60,44.73@58.45,0.60,44.36@58.47,0.60,44.00@58.46,0.60,43.65@58.43,0.60,43.31@58.36,0.60,42.98@58.26,0.60,42.67@58.13,0.60,42.38@57.96,0.60,42.10@57.75,0.60,41.83@57.52,0.60,41.59@57.26,0.60,41.35@56.98,0.60,41.13@56.67,0.60,40.92@56.34,0.60,40.72@56.00,0.60,40.54@55.64,0.60,40.36@55.26,0.60,40.20@54.87,0.60,40.04@54.48,0.60,39.89@54.08,0.60,39.75@53.67,0.60,39.61@53.26,0.60,39.48@52.86,0.60,39.36@52.45,0.60,39.24@52.06,0.60,39.12@51.66,0.60,39.00@51.28,0.60,38.89@50.91,0.60,38.79@50.55,0.60,38.68@50.20,0.60,38.58@49.86,0.60,38.49@49.52,0.60,38.40@49.20,0.60,38.32@48.89,0.60,38.24@48.58,0.60,38.18@48.29,0.60,38.12@48.00,0.60,38.08@47.72,0.60,38.05@47.45,0.60,38.02@47.19,0.60,38.01@46.94,0.60,38.02@46.69,0.60,38.04@46.46,0.60,38.07@46.23,0.60,38.12@46.01,0.60,38.19'
        #ROAD = '48.00,0.60,50.00@48.04,0.60,51.57@48.14,0.60,53.15@48.30,0.60,54.73@48.50,0.60,56.32@48.74,0.60,57.94@48.99,0.60,59.57@49.25,0.60,61.22@49.51,0.60,62.90@49.75,0.60,64.61@49.96,0.60,66.36@50.14,0.60,68.15@50.26,0.60,69.98@50.32,0.60,71.86@50.31,0.60,73.79@50.21,0.60,75.77@50.01,0.60,77.81@49.70,0.60,79.92@49.27,0.60,82.09@48.71,0.60,84.33@48.00,0.60,86.65@47.14,0.60,89.04@46.13,0.60,91.49@44.97,0.60,93.98@43.69,0.60,96.49@42.27,0.60,99.00@40.74,0.60,101.49@39.10,0.60,103.94@37.36,0.60,106.34@35.52,0.60,108.65@33.60,0.60,110.87@31.60,0.60,112.96@29.53,0.60,114.92@27.39,0.60,116.73@25.20,0.60,118.35@22.97,0.60,119.78@20.69,0.60,121.00@18.38,0.60,121.97@16.05,0.60,122.70@13.70,0.60,123.15@11.35,0.60,123.30@9.00,0.60,123.15@6.65,0.60,122.70@4.32,0.60,121.97@2.01,0.60,121.00@-0.27,0.60,119.78@-2.50,0.60,118.35@-4.69,0.60,116.73@-6.83,0.60,114.92@-8.90,0.60,112.96@-10.90,0.60,110.87@-12.82,0.60,108.65@-14.66,0.60,106.34@-16.40,0.60,103.94@-18.04,0.60,101.49@-19.57,0.60,99.00@-20.99,0.60,96.49@-22.27,0.60,93.98@-23.43,0.60,91.49@-24.44,0.60,89.04@-25.30,0.60,86.65@-26.01,0.60,84.33@-26.57,0.60,82.09@-27.00,0.60,79.92@-27.31,0.60,77.81@-27.51,0.60,75.77@-27.61,0.60,73.79@-27.62,0.60,71.86@-27.56,0.60,69.98@-27.44,0.60,68.15@-27.26,0.60,66.36@-27.05,0.60,64.61@-26.81,0.60,62.90@-26.55,0.60,61.22@-26.29,0.60,59.57@-26.04,0.60,57.94@-25.80,0.60,56.32@-25.60,0.60,54.73@-25.44,0.60,53.15@-25.34,0.60,51.57@-25.30,0.60,50.00@-25.34,0.60,48.43@-25.44,0.60,46.85@-25.60,0.60,45.27@-25.80,0.60,43.68@-26.04,0.60,42.06@-26.29,0.60,40.43@-26.55,0.60,38.78@-26.81,0.60,37.10@-27.05,0.60,35.39@-27.26,0.60,33.64@-27.44,0.60,31.85@-27.56,0.60,30.02@-27.62,0.60,28.14@-27.61,0.60,26.21@-27.51,0.60,24.23@-27.31,0.60,22.19@-27.00,0.60,20.08@-26.57,0.60,17.91@-26.01,0.60,15.67@-25.30,0.60,13.35@-24.44,0.60,10.96@-23.43,0.60,8.51@-22.27,0.60,6.02@-20.99,0.60,3.51@-19.57,0.60,1.00@-18.04,0.60,-1.49@-16.40,0.60,-3.94@-14.66,0.60,-6.34@-12.82,0.60,-8.65@-10.90,0.60,-10.87@-8.90,0.60,-12.96@-6.83,0.60,-14.92@-4.69,0.60,-16.73@-2.50,0.60,-18.35@-0.27,0.60,-19.78@2.01,0.60,-21.00@4.32,0.60,-21.97@6.65,0.60,-22.70@9.00,0.60,-23.15@11.35,0.60,-23.30@13.70,0.60,-23.15@16.05,0.60,-22.70@18.38,0.60,-21.97@20.69,0.60,-21.00@22.97,0.60,-19.78@25.20,0.60,-18.35@27.39,0.60,-16.73@29.53,0.60,-14.92@31.60,0.60,-12.96@33.60,0.60,-10.87@35.52,0.60,-8.65@37.36,0.60,-6.34@39.10,0.60,-3.94@40.74,0.60,-1.49@42.27,0.60,1.00@43.69,0.60,3.51@44.97,0.60,6.02@46.13,0.60,8.51@47.14,0.60,10.96@48.00,0.60,13.35@48.71,0.60,15.67@49.27,0.60,17.91@49.70,0.60,20.08@50.01,0.60,22.19@50.21,0.60,24.23@50.31,0.60,26.21@50.32,0.60,28.14@50.26,0.60,30.02@50.14,0.60,31.85@49.96,0.60,33.64@49.75,0.60,35.39@49.51,0.60,37.10@49.25,0.60,38.78@48.99,0.60,40.43@48.74,0.60,42.06@48.50,0.60,43.68@48.30,0.60,45.27@48.14,0.60,46.85@48.04,0.60,48.43'
        ROAD = '45.80,0.60,38.27@45.60,0.60,38.37@45.40,0.60,38.49@45.21,0.60,38.63@45.03,0.60,38.78@44.85,0.60,38.94@44.67,0.60,39.11@44.50,0.60,39.29@44.32,0.60,39.48@44.15,0.60,39.68@43.98,0.60,39.89@43.81,0.60,40.10@43.64,0.60,40.31@43.47,0.60,40.52@43.29,0.60,40.73@43.12,0.60,40.95@42.93,0.60,41.15@42.74,0.60,41.36@42.55,0.60,41.56@42.35,0.60,41.75@42.14,0.60,41.94@41.92,0.60,42.11@41.69,0.60,42.28@41.46,0.60,42.44@41.23,0.60,42.60@40.99,0.60,42.75@40.74,0.60,42.90@40.50,0.60,43.05@40.25,0.60,43.20@40.01,0.60,43.35@39.77,0.60,43.50@39.53,0.60,43.66@39.30,0.60,43.83@39.07,0.60,44.01@38.85,0.60,44.19@38.64,0.60,44.39@38.44,0.60,44.60@38.24,0.60,44.83@38.06,0.60,45.07@37.89,0.60,45.33@37.74,0.60,45.60@37.60,0.60,45.90@37.47,0.60,46.21@37.36,0.60,46.55@37.26,0.60,46.90@37.18,0.60,47.26@37.11,0.60,47.64@37.04,0.60,48.03@36.99,0.60,48.43@36.95,0.60,48.85@36.92,0.60,49.27@36.90,0.60,49.70@36.89,0.60,50.13@36.88,0.60,50.57@36.88,0.60,51.01@36.89,0.60,51.46@36.90,0.60,51.90@36.92,0.60,52.35@36.95,0.60,52.79@36.97,0.60,53.23@37.00,0.60,53.66@37.04,0.60,54.09@37.08,0.60,54.52@37.12,0.60,54.93@37.16,0.60,55.34@37.21,0.60,55.74@37.27,0.60,56.13@37.33,0.60,56.51@37.39,0.60,56.89@37.47,0.60,57.24@37.55,0.60,57.59@37.64,0.60,57.93@37.73,0.60,58.25@37.84,0.60,58.56@37.95,0.60,58.85@38.07,0.60,59.13@38.20,0.60,59.39@38.34,0.60,59.63@38.50,0.60,59.86@38.66,0.60,60.07@38.84,0.60,60.26@39.03,0.60,60.43@39.22,0.60,60.59@39.44,0.60,60.72@39.66,0.60,60.85@39.89,0.60,60.95@40.13,0.60,61.05@40.38,0.60,61.13@40.64,0.60,61.20@40.91,0.60,61.26@41.19,0.60,61.32@41.48,0.60,61.37@41.77,0.60,61.41@42.07,0.60,61.45@42.38,0.60,61.49@42.69,0.60,61.52@43.01,0.60,61.56@43.33,0.60,61.60@43.66,0.60,61.64@44.00,0.60,61.68@44.34,0.60,61.73@44.68,0.60,61.78@45.02,0.60,61.84@45.37,0.60,61.90@45.72,0.60,61.97@46.07,0.60,62.03@46.43,0.60,62.09@46.79,0.60,62.15@47.15,0.60,62.20@47.52,0.60,62.24@47.89,0.60,62.28@48.26,0.60,62.30@48.63,0.60,62.31@49.00,0.60,62.31@49.38,0.60,62.29@49.76,0.60,62.25@50.14,0.60,62.19@50.52,0.60,62.11@50.90,0.60,62.01@51.28,0.60,61.88@51.66,0.60,61.73@52.05,0.60,61.55@52.43,0.60,61.34@52.82,0.60,61.11@53.19,0.60,60.85@53.57,0.60,60.58@53.93,0.60,60.29@54.29,0.60,59.98@54.64,0.60,59.66@54.98,0.60,59.33@55.30,0.60,58.99@55.61,0.60,58.64@55.91,0.60,58.28@56.19,0.60,57.92@56.45,0.60,57.56@56.69,0.60,57.20@56.90,0.60,56.85@57.10,0.60,56.50@57.27,0.60,56.15@57.41,0.60,55.82@57.53,0.60,55.50@57.62,0.60,55.19@57.68,0.60,54.89@57.72,0.60,54.60@57.73,0.60,54.32@57.73,0.60,54.05@57.71,0.60,53.79@57.68,0.60,53.53@57.63,0.60,53.28@57.58,0.60,53.03@57.52,0.60,52.77@57.46,0.60,52.52@57.40,0.60,52.27@57.34,0.60,52.01@57.28,0.60,51.75@57.23,0.60,51.48@57.19,0.60,51.20@57.16,0.60,50.92@57.14,0.60,50.63@57.14,0.60,50.32@57.16,0.60,50.00@57.20,0.60,49.67@57.27,0.60,49.32@57.34,0.60,48.96@57.43,0.60,48.60@57.53,0.60,48.22@57.64,0.60,47.84@57.75,0.60,47.45@57.87,0.60,47.06@57.98,0.60,46.67@58.08,0.60,46.28@58.18,0.60,45.88@58.27,0.60,45.50@58.35,0.60,45.11@58.41,0.60,44.73@58.45,0.60,44.36@58.47,0.60,44.00@58.46,0.60,43.65@58.43,0.60,43.31@58.36,0.60,42.98@58.26,0.60,42.67@58.13,0.60,42.38@57.96,0.60,42.10@57.75,0.60,41.83@57.52,0.60,41.59@57.26,0.60,41.35@56.98,0.60,41.13@56.67,0.60,40.92@56.34,0.60,40.72@56.00,0.60,40.54@55.64,0.60,40.36@55.26,0.60,40.20@54.87,0.60,40.04@54.48,0.60,39.89@54.08,0.60,39.75@53.67,0.60,39.61@53.26,0.60,39.48@52.86,0.60,39.36@52.45,0.60,39.24@52.06,0.60,39.12@51.66,0.60,39.00@51.28,0.60,38.89@50.91,0.60,38.79@50.55,0.60,38.68@50.20,0.60,38.58@49.86,0.60,38.49@49.52,0.60,38.40@49.20,0.60,38.32@48.89,0.60,38.24@48.58,0.60,38.18@48.29,0.60,38.12@48.00,0.60,38.08@47.72,0.60,38.05@47.45,0.60,38.02@47.19,0.60,38.01@46.94,0.60,38.02@46.69,0.60,38.04@46.46,0.60,38.07@46.23,0.60,38.12@46.01,0.60,38.19'
        initial_pose = [45.80,38.27,-50]
        logger.debug("reseting")
        self.reset_scenario(0, ROAD)
        time.sleep(0.1)
        self.send_pose(initial_pose)
        #time.sleep(0.4)
        #self.send_reset_car()
        self.timer.reset()
        time.sleep(0.1)
        self.image_array = np.zeros(self.camera_img_size)
        self.image_array_b = None
        self.last_obs = self.image_array
        self.time_received = time.time()
        self.last_received = self.time_received
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.speed = 0.0
        self.forward_vel = 0.0
        self.over = False
        self.missed_checkpoint = False
        self.dq = False
        self.gyro_x = 0.0
        self.gyro_y = 0.0
        self.gyro_z = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.lidar = []
        self.current_lap_time = 0.0
        self.last_lap_time = 0.0
        self.lap_count = 0

        # car
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # display waiting screen   

    def get_sensor_size(self) -> Tuple[int, int, int]:
        return self.camera_img_size

    def take_action(self, action: np.ndarray) -> None:
        self.send_control(action[0], action[1])

    def observe(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        while self.last_received == self.time_received:
            time.sleep(0.001)

        self.last_received = self.time_received
        observation = self.image_array
        done = self.is_game_over()
        reward = self.calc_reward(done)

        info = {
            "pos": (self.x, self.y, self.z),
            "cte": self.cte,
            "speed": self.speed,
            "forward_vel": self.forward_vel,
            "hit": self.hit,
            "gyro": (self.gyro_x, self.gyro_y, self.gyro_z),
            "accel": (self.accel_x, self.accel_y, self.accel_z),
            "vel": (self.vel_x, self.vel_y, self.vel_z),
            "lidar": (self.lidar),
            "car": (self.roll, self.pitch, self.yaw),
            "last_lap_time": self.last_lap_time,
            "lap_count": self.lap_count,
        }

        # Add the second image to the dict
        if self.image_array_b is not None:
            info["image_b"] = self.image_array_b

        # self.timer.on_frame()

        return observation, reward, done, info

    def is_game_over(self) -> bool:
        return self.over

    # ------ RL interface ----------- #

    def set_reward_fn(self, reward_fn: Callable[[], float]):
        """
        allow users to set their own reward function
        """
        self.calc_reward = types.MethodType(reward_fn, self)
        logger.debug("custom reward fn set.")

    def calc_reward(self, done: bool) -> float:
        # Normalization factor, real max speed is around 30
        # but only attained on a long straight line
        # max_speed = 10

        if done:
            return -1.0

        if self.cte > self.max_cte:
            return -1.0

        # Collision
        if self.hit != "none":
            return -2.0

        # going fast close to the center of lane yeilds best reward
        if self.forward_vel > 0.0:
            return (1.0 - (math.fabs(self.cte) / self.max_cte)) * self.forward_vel

        # in reverse, reward doesn't have centering term as this can result in some exploits
        return self.forward_vel

    # ------ Socket interface ----------- #

    def on_telemetry(self, message: Dict[str, Any]) -> None:
        img_string = message["image"]
        image = Image.open(BytesIO(base64.b64decode(img_string)))

        # always update the image_array as the observation loop will hang if not changing.
        self.image_array = np.asarray(image)
        self.time_received = time.time()

        if "image_b" in message:
            img_string_b = message["image_b"]
            image_b = Image.open(BytesIO(base64.b64decode(img_string_b)))
            self.image_array_b = np.asarray(image_b)

        if "pos_x" in message:
            self.x = message["pos_x"]
            self.y = message["pos_y"]
            self.z = message["pos_z"]

        if "speed" in message:
            self.speed = message["speed"]

        e = [self.pitch * np.pi / 180.0, self.yaw * np.pi / 180.0, self.roll * np.pi / 180.0]
        q = euler_to_quat(e)

        forward = rotate_vec(q, [0.0, 0.0, 1.0])

        # dot
        self.forward_vel = forward[0] * self.vel_x + forward[1] * self.vel_y + forward[2] * self.vel_z

        if "gyro_x" in message:
            self.gyro_x = message["gyro_x"]
            self.gyro_y = message["gyro_y"]
            self.gyro_z = message["gyro_z"]
        if "accel_x" in message:
            self.accel_x = message["accel_x"]
            self.accel_y = message["accel_y"]
            self.accel_z = message["accel_z"]
        if "vel_x" in message:
            self.vel_x = message["vel_x"]
            self.vel_y = message["vel_y"]
            self.vel_z = message["vel_z"]

        if "roll" in message:
            self.roll = message["roll"]
            self.pitch = message["pitch"]
            self.yaw = message["yaw"]

        # Cross track error not always present.
        # Will be missing if path is not setup in the given scene.
        # It should be setup in the 4 scenes available now.
        if "cte" in message:
            self.cte = message["cte"]

        if "lidar" in message:
            self.lidar = self.process_lidar_packet(message["lidar"])

        # don't update hit once session over
        if self.over:
            return

        if "hit" in message:
            self.hit = message["hit"]

        self.determine_episode_over()

    def on_cross_start(self, message: Dict[str, Any]) -> None:
        logger.info(f"crossed start line: lap_time {message['lap_time']}")

    def on_race_start(self, message: Dict[str, Any]) -> None:
        logger.debug("race started")

    def on_race_stop(self, message: Dict[str, Any]) -> None:
        logger.debug("race stoped")

    def on_missed_checkpoint(self, message: Dict[str, Any]) -> None:
        logger.info("racer missed checkpoint")
        self.missed_checkpoint = True

    def on_DQ(self, message: Dict[str, Any]) -> None:
        logger.info("racer DQ")
        self.dq = True

    def on_ping(self, message: Dict[str, Any]) -> None:
        """
        no reply needed at this point. Server sends these as a keep alive to make sure clients haven't gone away.
        """
        pass

    def set_episode_over_fn(self, ep_over_fn: Callable[[], bool]):
        """
        allow userd to define their own episode over function
        """
        self.determine_episode_over = types.MethodType(ep_over_fn, self)
        logger.debug("custom ep_over fn set.")

    def determine_episode_over(self):
        # we have a few initial frames on start that are sometimes very large CTE when it's behind
        # the path just slightly. We ignore those.
        if math.fabs(self.cte) > 2 * self.max_cte:
            pass
        elif math.fabs(self.cte) > self.max_cte:
            logger.debug(f"game over: cte {self.cte}")
            self.over = True
        elif self.hit != "none":
            logger.debug(f"game over: hit {self.hit}")
            self.over = True
        elif self.missed_checkpoint:
            logger.debug("missed checkpoint")
            self.over = True
        elif self.dq:
            logger.debug("disqualified")
            self.over = True

        # Disable reset
        if os.environ.get("RACE") == "True":
            self.over = False

    def on_scene_selection_ready(self, message: Dict[str, Any]) -> None:
        logger.debug("SceneSelectionReady")
        self.send_get_scene_names()

    def on_car_loaded(self, message: Dict[str, Any]) -> None:
        logger.debug("car loaded")
        self.loaded = True
        # Enable hand brake, so the car doesn't move
        self.send_control(0, 0, 1.0)
        self.on_need_car_config({})

    def on_recv_scene_names(self, message: Dict[str, Any]) -> None:
        if message:
            names = message["scene_names"]
            logger.debug(f"SceneNames: {names}")
            print("loading scene", self.SceneToLoad)
            if self.SceneToLoad in names:
                self.send_load_scene(self.SceneToLoad)
            else:
                raise ValueError(f"Scene name {self.SceneToLoad} not in scene list {names}")

    def send_control(self, steer: float, throttle: float, brake: float = 0.0) -> None:  
        #print(steer)
        """
        Send command to simulator.

        :param steer: desired steering
        :param throttle: desired throttle
        :param brake: whether to activate or not hand brake
            (can be a continuous value)
        """
        if not self.loaded:
            return
        msg = {
            "msg_type": "control",
            "steering": str(steer),
            "throttle": str(throttle),
            "brake": str(brake),
        }
        self.queue_message(msg)

    def reset_scenario(self,road_style, waypoints: Union[str, None]):
        msg = {
            "msg_type": "regen_road",
            'road_style': road_style.__str__(),
            "wayPoints": waypoints.__str__(),

        }
        print("reset_scenario PPPPPPPPPPPPPPP")

        self.client.send_now(msg)

    def send_reset_car(self) -> None:
        msg = {"msg_type": "reset_car"}
        print(msg)
        self.queue_message(msg)

    def send_get_scene_names(self) -> None:
        msg = {"msg_type": "get_scene_names"}
        self.queue_message(msg)

    def send_load_scene(self, scene_name: str) -> None:
        msg = {"msg_type": "load_scene", "scene_name": scene_name}
        self.queue_message(msg)

    def send_exit_scene(self) -> None:
        msg = {"msg_type": "exit_scene"}
        self.queue_message(msg)

    def send_car_config(
        self,
        body_style: str = "donkey",
        body_rgb: Tuple[int, int, int] = (255, 255, 255),
        car_name: str = "car",
        font_size: int = 100,
    ):
        """
        # body_style = "donkey" | "bare" | "car01" | "f1" | "cybertruck"
        # body_rgb  = (128, 128, 128) tuple of ints
        # car_name = "string less than 64 char"
        """
        assert isinstance(body_style, str)
        assert isinstance(body_rgb, list) or isinstance(body_rgb, tuple)
        assert len(body_rgb) == 3
        assert isinstance(car_name, str)
        assert isinstance(font_size, int) or isinstance(font_size, str)

        msg = {
            "msg_type": "car_config",
            "body_style": body_style,
            "body_r": str(body_rgb[0]),
            "body_g": str(body_rgb[1]),
            "body_b": str(body_rgb[2]),
            "car_name": car_name,
            "font_size": str(font_size),
        }
        self.blocking_send(msg)
        time.sleep(0.1)

    def send_racer_bio(self, racer_name: str, car_name: str, bio: str, country: str, guid: str) -> None:
        # body_style = "donkey" | "bare" | "car01" choice of string
        # body_rgb  = (128, 128, 128) tuple of ints
        # car_name = "string less than 64 char"
        # guid = "some random string"
        msg = {
            "msg_type": "racer_info",
            "racer_name": racer_name,
            "car_name": car_name,
            "bio": bio,
            "country": country,
            "guid": guid,
        }
        self.blocking_send(msg)
        time.sleep(0.1)

    def send_cam_config(
        self,
        msg_type: str = "cam_config",
        img_w: int = 0,
        img_h: int = 0,
        img_d: int = 0,
        img_enc: Union[str, int] = 0,  # 0 is default value
        fov: int = 0,
        fish_eye_x: float = 0.0,
        fish_eye_y: float = 0.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0,
        rot_x: float = 0.0,
        rot_y: float = 0.0,
        rot_z: float = 0.0,
    ) -> None:
        """Camera config
        set any field to Zero to get the default camera setting.
        offset_x moves camera left/right
        offset_y moves camera up/down
        offset_z moves camera forward/back
        rot_x will rotate the camera
        with fish_eye_x/y == 0.0 then you get no distortion
        img_enc can be one of JPG|PNG|TGA
        """
        msg = {
            "msg_type": msg_type,
            "fov": str(fov),
            "fish_eye_x": str(fish_eye_x),
            "fish_eye_y": str(fish_eye_y),
            "img_w": str(img_w),
            "img_h": str(img_h),
            "img_d": str(img_d),
            "img_enc": str(img_enc),
            "offset_x": str(offset_x),
            "offset_y": str(offset_y),
            "offset_z": str(offset_z),
            "rot_x": str(rot_x),
            "rot_y": str(rot_y),
            "rot_z": str(rot_z),
        }
        self.blocking_send(msg)
        time.sleep(0.1)

    def send_lidar_config(
        self,
        deg_per_sweep_inc: float = 2.0,
        deg_ang_down: float = 0.0,
        deg_ang_delta: float = -1.0,
        num_sweeps_levels: int = 1,
        max_range: float = 50.0,
        noise: float = 0.5,
        offset_x: float = 0.0,
        offset_y: float = 0.5,
        offset_z: float = 0.5,
        rot_x: float = 0.0,
    ):
        """Lidar config
        offset_x moves lidar left/right
        the offset_y moves lidar up/down
        the offset_z moves lidar forward/back
        deg_per_sweep_inc : as the ray sweeps around, how many degrees does it advance per sample (int)
        deg_ang_down : what is the starting angle for the initial sweep compared to the forward vector
        deg_ang_delta : what angle change between sweeps
        num_sweeps_levels : how many complete 360 sweeps (int)
        max_range : what it max distance we will register a hit
        noise : what is the scalar on the perlin noise applied to point position

        Here's some sample settings that similate a more sophisticated lidar:
        msg = '{ "msg_type" : "lidar_config",
        "degPerSweepInc" : "2.0", "degAngDown" : "25", "degAngDelta" : "-1.0",
        "numSweepsLevels" : "25", "maxRange" : "50.0", "noise" : "0.2",
        "offset_x" : "0.0", "offset_y" : "1.0", "offset_z" : "1.0", "rot_x" : "0.0" }'
        And here's some sample settings that similate a simple RpLidar A2 one level horizontal scan.
        msg = '{ "msg_type" : "lidar_config", "degPerSweepInc" : "2.0",
        "degAngDown" : "0.0", "degAngDelta" : "-1.0", "numSweepsLevels" : "1",
        "maxRange" : "50.0", "noise" : "0.4",
        "offset_x" : "0.0", "offset_y" : "0.5", "offset_z" : "0.5", "rot_x" : "0.0" }'
        """
        msg = {
            "msg_type": "lidar_config",
            "degPerSweepInc": str(deg_per_sweep_inc),
            "degAngDown": str(deg_ang_down),
            "degAngDelta": str(deg_ang_delta),
            "numSweepsLevels": str(num_sweeps_levels),
            "maxRange": str(max_range),
            "noise": str(noise),
            "offset_x": str(offset_x),
            "offset_y": str(offset_y),
            "offset_z": str(offset_z),
            "rot_x": str(rot_x),
        }
        self.blocking_send(msg)
        time.sleep(0.1)

        self.lidar_deg_per_sweep_inc = float(deg_per_sweep_inc)
        self.lidar_num_sweep_levels = int(num_sweeps_levels)
        self.lidar_deg_ang_delta = float(deg_ang_delta)

    def process_lidar_packet(self, lidar_info: List[Dict[str, float]]) -> np.ndarray:
        point_per_sweep = int(360 / self.lidar_deg_per_sweep_inc)
        points_num = round(abs(self.lidar_num_sweep_levels * point_per_sweep))
        reconstructed_lidar_info = [-1 for _ in range(points_num)]  # we chose -1 to be the "None" value

        if lidar_info is not None:
            for point in lidar_info:
                rx = point["rx"]
                ry = point["ry"]
                d = point["d"]

                x_index = round(abs(rx / self.lidar_deg_per_sweep_inc))
                y_index = round(abs(ry / self.lidar_deg_ang_delta))

                reconstructed_lidar_info[point_per_sweep * y_index + x_index] = d

        return np.array(reconstructed_lidar_info)

    def blocking_send(self, msg: Dict[str, Any]) -> None:
        if self.client is None:
            logger.debug(f"skipping: \n {msg}")
            return

        logger.debug(f"blocking send \n {msg}")
        self.client.send_now(msg)

    def queue_message(self, msg: Dict[str, Any]) -> None:
        if self.client is None:
            logger.debug(f"skipping: \n {msg}")
            return

        logger.debug(f"sending \n {msg}")
        self.client.queue_message(msg)
