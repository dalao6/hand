from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import pybullet as p
    import pybullet_data
except Exception as e:
    raise RuntimeError("Missing dependency: pybullet. Please run `pip install pybullet`.") from e


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    os.environ["PYTHONHASHSEED"] = str(int(seed))


def _norm(s: Any) -> str:
    x = str(s).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in x:
        x = x.replace("__", "_")
    return x


def _category_to_size(category: str) -> Tuple[float, float, float]:
    c = _norm(category)
    if "cup" in c or "bottle" in c:
        return 0.025, 0.025, 0.06
    if "mouse" in c or "remote" in c:
        return 0.035, 0.02, 0.015
    if "book" in c:
        return 0.05, 0.035, 0.02
    if "bowl" in c:
        return 0.04, 0.04, 0.02
    if "laptop" in c:
        return 0.08, 0.055, 0.015
    if "backpack" in c:
        return 0.06, 0.045, 0.08
    if "chair" in c:
        return 0.08, 0.08, 0.12
    if "dining_table" in c or "table" in c:
        return 0.1, 0.06, 0.02
    return 0.04, 0.04, 0.03


def _category_to_rgba(category: str) -> Tuple[float, float, float, float]:
    c = _norm(category)
    if "cup" in c:
        return 0.2, 0.8, 0.2, 1.0
    if "bottle" in c:
        return 0.2, 0.5, 0.9, 1.0
    if "book" in c:
        return 0.8, 0.2, 0.2, 1.0
    if "chair" in c:
        return 0.8, 0.6, 0.2, 1.0
    if "table" in c:
        return 0.6, 0.4, 0.2, 1.0
    return 0.7, 0.7, 0.7, 1.0


@dataclass
class SimObject:
    body_id: int
    obj_id: str
    category: str


@dataclass
class CameraSpec:
    eye: Tuple[float, float, float]
    target: Tuple[float, float, float]
    up: Tuple[float, float, float]
    fov: float = 60.0
    near: float = 0.01
    far: float = 3.0


class PyBulletMiniSim:
    def __init__(
        self,
        gui: bool = False,
        seed: int = 42,
        enable_domain_randomization: bool = False,
        domain_randomization_strength: float = 0.2,
        camera_pose_jitter: float = 0.0,
    ) -> None:
        self.gui = gui
        self.seed = seed
        self.rng = random.Random(seed)
        self.enable_domain_randomization = bool(enable_domain_randomization)
        self.domain_randomization_strength = max(0.0, min(1.0, float(domain_randomization_strength)))
        self.camera_pose_jitter = max(0.0, float(camera_pose_jitter))
        self.cid = -1
        self.robot_id = -1
        self.ee_link = 11
        self.arm_joints = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_joints = [9, 10]
        self.objects: Dict[str, SimObject] = {}
        self.attach_cid: Optional[int] = None
        self.attached_obj: Optional[str] = None
        self.traj_xyz: List[Tuple[float, float, float]] = []
        self.plane_id = -1
        self.table_id = -1
        self.collision_steps = 0
        self.step_count = 0
        self.grasp_stability_scores: List[float] = []
        self._camera_specs: Dict[str, CameraSpec] = {
            "front": CameraSpec(eye=(1.10, 0.0, 0.55), target=(0.55, 0.0, 0.0), up=(0.0, 0.0, 1.0)),
            "top": CameraSpec(eye=(0.55, 0.0, 1.10), target=(0.55, 0.0, 0.02), up=(1.0, 0.0, 0.0)),
            "side": CameraSpec(eye=(0.30, -0.75, 0.55), target=(0.55, 0.0, 0.0), up=(0.0, 0.0, 1.0)),
        }

    def connect(self) -> None:
        mode = p.GUI if self.gui else p.DIRECT
        self.cid = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, numSolverIterations=80)
        p.setTimeStep(1.0 / 240.0)
        p.setGravity(0, 0, -9.81)
        p.resetSimulation()
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", basePosition=[0.55, 0.0, -0.65], useFixedBase=True)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
        self.collision_steps = 0
        self.step_count = 0
        self.grasp_stability_scores = []
        self._reset_robot_pose()

    def close(self) -> None:
        if self.cid >= 0:
            p.disconnect(self.cid)
            self.cid = -1

    def _dr_scale(self, base: float, ratio: float) -> float:
        if not self.enable_domain_randomization or ratio <= 0.0:
            return float(base)
        lo = max(1e-6, 1.0 - ratio * self.domain_randomization_strength)
        hi = 1.0 + ratio * self.domain_randomization_strength
        return float(base * self.rng.uniform(lo, hi))

    def _dr_rgb(self, rgba: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        if not self.enable_domain_randomization:
            return rgba
        delta = 0.28 * self.domain_randomization_strength
        rgb = []
        for c in rgba[:3]:
            rgb.append(float(min(1.0, max(0.0, c + self.rng.uniform(-delta, delta)))))
        return rgb[0], rgb[1], rgb[2], rgba[3]

    def _build_camera_spec(self, view: str, apply_jitter: bool = True) -> CameraSpec:
        base = self._camera_specs.get(view, self._camera_specs["front"])
        eye = [float(v) for v in base.eye]
        target = [float(v) for v in base.target]
        up = [float(v) for v in base.up]
        fov = float(base.fov)

        if apply_jitter and self.enable_domain_randomization and self.camera_pose_jitter > 0.0:
            j = self.camera_pose_jitter * self.domain_randomization_strength
            for i in range(3):
                eye[i] += self.rng.uniform(-j, j)
                target[i] += self.rng.uniform(-j, j)
            fov = max(35.0, min(90.0, fov + self.rng.uniform(-6.0, 6.0) * self.domain_randomization_strength))

        return CameraSpec(
            eye=(eye[0], eye[1], eye[2]),
            target=(target[0], target[1], target[2]),
            up=(up[0], up[1], up[2]),
            fov=fov,
            near=float(base.near),
            far=float(base.far),
        )

    @staticmethod
    def _intrinsics_from_fov(width: int, height: int, fov_deg: float) -> Dict[str, float]:
        fov = max(1e-6, float(fov_deg))
        fy = 0.5 * float(height) / math.tan(math.radians(fov) / 2.0)
        fx = fy
        cx = 0.5 * (float(width) - 1.0)
        cy = 0.5 * (float(height) - 1.0)
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": int(width), "height": int(height)}

    def export_camera_calibration(self, out_path: str, width: int = 640, height: int = 480) -> None:
        payload: Dict[str, Any] = {
            "image_width": int(width),
            "image_height": int(height),
            "views": {},
        }
        for view in ["front", "top", "side"]:
            spec = self._build_camera_spec(view, apply_jitter=False)
            view_mat = p.computeViewMatrix(cameraEyePosition=spec.eye, cameraTargetPosition=spec.target, cameraUpVector=spec.up)
            proj_mat = p.computeProjectionMatrixFOV(
                fov=spec.fov,
                aspect=float(width) / float(height),
                nearVal=spec.near,
                farVal=spec.far,
            )
            w2c = np.array(view_mat, dtype=float).reshape((4, 4), order="F")
            c2w = np.linalg.inv(w2c)
            payload["views"][view] = {
                "eye": [float(x) for x in spec.eye],
                "target": [float(x) for x in spec.target],
                "up": [float(x) for x in spec.up],
                "fov_deg": float(spec.fov),
                "near": float(spec.near),
                "far": float(spec.far),
                "intrinsics": self._intrinsics_from_fov(width, height, spec.fov),
                "view_matrix_col_major": [float(x) for x in view_mat],
                "projection_matrix_col_major": [float(x) for x in proj_mat],
                "world_to_camera_row_major": [[float(v) for v in row] for row in w2c.tolist()],
                "camera_to_world_row_major": [[float(v) for v in row] for row in c2w.tolist()],
            }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _update_collision_stats(self) -> None:
        if self.robot_id < 0:
            return
        contacts = p.getContactPoints(bodyA=self.robot_id)
        if contacts:
            self.collision_steps += 1

    def _step(self, n: int = 120) -> None:
        for _ in range(n):
            p.stepSimulation()
            self.step_count += 1
            self._update_collision_stats()
            ee_pos = p.getLinkState(self.robot_id, self.ee_link)[0]
            self.traj_xyz.append((float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])))

    def _reset_robot_pose(self) -> None:
        home = [0.0, -0.4, 0.0, -2.1, 0.0, 2.0, 0.8]
        for j, v in zip(self.arm_joints, home):
            p.resetJointState(self.robot_id, j, v)
        self._set_gripper(opening=0.04)
        self._step(60)

    def _set_gripper(self, opening: float) -> None:
        opening = max(0.0, min(0.04, opening))
        for j in self.gripper_joints:
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=opening, force=60)

    def _ik_move(self, target_pos: Tuple[float, float, float], yaw_deg: float = 0.0, steps: int = 140) -> None:
        q = p.getQuaternionFromEuler([math.pi, 0.0, math.radians(yaw_deg)])
        joints = p.calculateInverseKinematics(self.robot_id, self.ee_link, targetPosition=target_pos, targetOrientation=q)
        for j, tgt in zip(self.arm_joints, joints[: len(self.arm_joints)]):
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL, targetPosition=float(tgt), force=120)
        self._step(steps)

    def _spawn_box(self, pos: Tuple[float, float, float], size: Tuple[float, float, float], rgba: Tuple[float, float, float, float]) -> int:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size[0] / 2.0, size[1] / 2.0, size[2] / 2.0])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size[0] / 2.0, size[1] / 2.0, size[2] / 2.0], rgbaColor=rgba)
        mass = self._dr_scale(0.08, ratio=0.35)
        body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)
        return int(body)

    def apply_domain_randomization(self) -> None:
        if not self.enable_domain_randomization:
            return
        table_lf = self._dr_scale(0.8, ratio=0.45)
        table_sf = self._dr_scale(0.7, ratio=0.45)
        p.changeDynamics(self.table_id, -1, lateralFriction=table_lf, spinningFriction=table_sf, restitution=self._dr_scale(0.02, ratio=1.0))
        p.changeDynamics(self.plane_id, -1, lateralFriction=self._dr_scale(0.9, ratio=0.3), restitution=self._dr_scale(0.01, ratio=1.0))
        for j in self.arm_joints + self.gripper_joints:
            p.changeDynamics(self.robot_id, j, lateralFriction=self._dr_scale(1.0, ratio=0.15), jointDamping=self._dr_scale(0.04, ratio=0.45))

    def load_scene_from_sample(self, sample: Dict[str, Any]) -> None:
        self.objects = {}
        gt = sample.get("ground_truth", {}) if isinstance(sample.get("ground_truth"), dict) else {}
        objs = gt.get("objects", []) if isinstance(gt.get("objects"), list) else []
        if not objs:
            return

        bboxes = [o.get("bbox", []) for o in objs if isinstance(o, dict) and isinstance(o.get("bbox", []), list) and len(o.get("bbox", [])) == 4]
        max_x = max((float(b[0]) + float(b[2]) for b in bboxes), default=640.0)
        max_y = max((float(b[1]) + float(b[3]) for b in bboxes), default=480.0)
        w_ref = max(max_x, 1.0)
        h_ref = max(max_y, 1.0)

        table_x_min, table_x_max = 0.35, 0.75
        table_y_min, table_y_max = -0.28, 0.28
        z_top = 0.02

        for i, o in enumerate(objs):
            if not isinstance(o, dict):
                continue
            obj_id = str(o.get("id", f"obj_{i+1}"))
            cat = str(o.get("category", "object"))
            b = o.get("bbox", [0.0, 0.0, 0.0, 0.0])
            if not isinstance(b, list) or len(b) != 4:
                b = [0.0, 0.0, 0.0, 0.0]
            cx = (float(b[0]) + float(b[2]) / 2.0) / w_ref
            cy = (float(b[1]) + float(b[3]) / 2.0) / h_ref
            x = table_x_min + cx * (table_x_max - table_x_min)
            y = table_y_min + (1.0 - cy) * (table_y_max - table_y_min)
            size = _category_to_size(cat)
            sx = self._dr_scale(size[0], ratio=0.2)
            sy = self._dr_scale(size[1], ratio=0.2)
            sz = self._dr_scale(size[2], ratio=0.2)
            z = z_top + sz / 2.0
            pos_jitter = 0.01 + 0.02 * self.domain_randomization_strength if self.enable_domain_randomization else 0.01
            x += self.rng.uniform(-pos_jitter, pos_jitter)
            y += self.rng.uniform(-pos_jitter, pos_jitter)
            rgba = self._dr_rgb(_category_to_rgba(cat))
            body = self._spawn_box((x, y, z), (sx, sy, sz), rgba)
            p.changeDynamics(
                body,
                -1,
                lateralFriction=self._dr_scale(0.9, ratio=0.35),
                spinningFriction=self._dr_scale(0.5, ratio=0.5),
                restitution=self._dr_scale(0.03, ratio=1.0),
            )
            self.objects[_norm(obj_id)] = SimObject(body_id=body, obj_id=obj_id, category=cat)

        self._step(120)

    def _resolve_target(self, target: str) -> Optional[SimObject]:
        t = _norm(target)
        if t in self.objects:
            return self.objects[t]
        t_base = t.split("_")[0]
        for k, v in self.objects.items():
            if _norm(v.category) == t or _norm(v.category) == t_base:
                return v
            if t in k or k in t:
                return v
        return None

    def _obj_pos(self, obj: SimObject) -> Tuple[float, float, float]:
        pos, _ = p.getBasePositionAndOrientation(obj.body_id)
        return float(pos[0]), float(pos[1]), float(pos[2])

    def action_move_to(self, target: str) -> None:
        obj = self._resolve_target(target)
        if obj is None:
            return
        x, y, z = self._obj_pos(obj)
        self._ik_move((x, y, max(z + 0.16, 0.20)), steps=120)
        self._ik_move((x, y, max(z + 0.07, 0.12)), steps=110)

    def _measure_grasp_stability(self, obj: SimObject) -> float:
        if self.attach_cid is None:
            return 0.0
        stable = 0
        total = 30
        for _ in range(total):
            self._step(1)
            ee = p.getLinkState(self.robot_id, self.ee_link)[0]
            op = self._obj_pos(obj)
            d = math.sqrt((ee[0] - op[0]) ** 2 + (ee[1] - op[1]) ** 2 + (ee[2] - op[2]) ** 2)
            if d < 0.14 and op[2] > 0.04:
                stable += 1
        return float(stable / total)

    def action_grasp(self, target: str) -> None:
        obj = self._resolve_target(target)
        if obj is None:
            return
        self._set_gripper(0.0)
        self._step(90)
        if self.attach_cid is not None:
            return
        ee_pos = p.getLinkState(self.robot_id, self.ee_link)[0]
        op = self._obj_pos(obj)
        d = math.sqrt((ee_pos[0] - op[0]) ** 2 + (ee_pos[1] - op[1]) ** 2 + (ee_pos[2] - op[2]) ** 2)
        if d < 0.12:
            self.attach_cid = p.createConstraint(
                parentBodyUniqueId=self.robot_id,
                parentLinkIndex=self.ee_link,
                childBodyUniqueId=obj.body_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0.06],
                childFramePosition=[0, 0, 0],
            )
            self.attached_obj = _norm(obj.obj_id)
            self.grasp_stability_scores.append(self._measure_grasp_stability(obj))

    def action_release(self, _target: str) -> None:
        if self.attach_cid is not None:
            p.removeConstraint(self.attach_cid)
            self.attach_cid = None
            self.attached_obj = None
        self._set_gripper(0.04)
        self._step(80)

    def action_push(self, target: str) -> None:
        obj = self._resolve_target(target)
        if obj is None:
            return
        x, y, z = self._obj_pos(obj)
        self._ik_move((x - 0.08, y, max(z + 0.04, 0.10)), steps=100)
        self._ik_move((x + 0.03, y, max(z + 0.04, 0.10)), steps=100)
        ori = p.getBasePositionAndOrientation(obj.body_id)[1]
        p.resetBasePositionAndOrientation(obj.body_id, [x + 0.06, y, z], ori)
        self._step(90)

    def action_rotate(self, target: str) -> None:
        obj = self._resolve_target(target)
        if obj is None:
            return
        x, y, z = self._obj_pos(obj)
        _, quat = p.getBasePositionAndOrientation(obj.body_id)
        eul = p.getEulerFromQuaternion(quat)
        nq = p.getQuaternionFromEuler([eul[0], eul[1], eul[2] + math.radians(25.0)])
        p.resetBasePositionAndOrientation(obj.body_id, [x, y, z], nq)
        self._ik_move((x, y, max(z + 0.10, 0.14)), yaw_deg=25.0, steps=90)

    def execute_sequence(self, action_seq: List[Dict[str, Any]]) -> None:
        for step in action_seq:
            if not isinstance(step, dict):
                continue
            action = _norm(step.get("action", ""))
            target = str(step.get("target", ""))
            if action in {"move_to", "moveto", "navigate", "approach"}:
                self.action_move_to(target)
            elif action in {"grasp", "pick", "pickup"}:
                self.action_grasp(target)
            elif action in {"release", "drop", "place"}:
                self.action_release(target)
            elif action in {"push", "move"}:
                self.action_push(target)
            elif action in {"rotate", "turn"}:
                self.action_rotate(target)
            else:
                self._step(60)

    def camera_image(self, view: str = "front", width: int = 640, height: int = 480) -> Image.Image:
        spec = self._build_camera_spec(view)
        view_mat = p.computeViewMatrix(cameraEyePosition=spec.eye, cameraTargetPosition=spec.target, cameraUpVector=spec.up)
        proj_mat = p.computeProjectionMatrixFOV(
            fov=spec.fov,
            aspect=float(width) / float(height),
            nearVal=spec.near,
            farVal=spec.far,
        )

        light_color = [1.0, 1.0, 1.0]
        light_dir = [0.5, 0.0, -1.0]
        if self.enable_domain_randomization:
            dl = 0.25 * self.domain_randomization_strength
            light_color = [
                float(min(1.0, max(0.4, 0.85 + self.rng.uniform(-dl, dl)))),
                float(min(1.0, max(0.4, 0.85 + self.rng.uniform(-dl, dl)))),
                float(min(1.0, max(0.4, 0.85 + self.rng.uniform(-dl, dl)))),
            ]
            light_dir = [
                float(0.5 + self.rng.uniform(-0.7, 0.7) * self.domain_randomization_strength),
                float(self.rng.uniform(-0.7, 0.7) * self.domain_randomization_strength),
                float(-1.0 + self.rng.uniform(-0.35, 0.25) * self.domain_randomization_strength),
            ]

        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            shadow=1,
            lightDirection=light_dir,
            lightColor=light_color,
        )
        arr = np.reshape(rgba, (height, width, 4)).astype(np.uint8)
        rgb = arr[:, :, :3]
        return Image.fromarray(rgb)


def _rank_debug_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    failed = [r for r in records if _safe_float(r.get("task_success", 0.0)) < 0.5]
    succeeded = [r for r in records if _safe_float(r.get("task_success", 0.0)) >= 0.5]

    def k(r: Dict[str, Any]) -> Tuple[float, float, float, float, str, str]:
        return (
            -float(_safe_int(r.get("difficulty_level", 0))),
            float(_safe_float(r.get("task_success", 0.0))),
            -float(_safe_float(r.get("target_mismatch_rate", 0.0))),
            -float(_safe_float(r.get("illegal_action_rate", 0.0))),
            str(r.get("model", "")),
            str(r.get("sample_id", "")),
        )

    failed.sort(key=k)
    succeeded.sort(key=k)
    return failed + succeeded


def _load_debug(debug_jsonl: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not debug_jsonl or not os.path.exists(debug_jsonl):
        return out
    with open(debug_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def _load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("dataset must be list")
    return data


def _save_traj_xy(traj_xyz: List[Tuple[float, float, float]], out_path: str, title: str) -> None:
    if not traj_xyz:
        return
    xs = [p[0] for p in traj_xyz]
    ys = [p[1] for p in traj_xyz]
    plt.figure(figsize=(5, 4))
    plt.plot(xs, ys, linewidth=1.4)
    plt.scatter([xs[0]], [ys[0]], c="green", s=25)
    plt.scatter([xs[-1]], [ys[-1]], c="red", s=25)
    plt.title(title)
    plt.xlabel("x 轴")
    plt.ylabel("y 轴")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _save_grid_2x3(images: List[str], out_path: str) -> None:
    n = min(6, len(images))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < n and os.path.exists(images[i]):
            try:
                ax.imshow(Image.open(images[i]).convert("RGB"))
                ttl = os.path.basename(images[i]).replace(".png", "")
                ttl = (
                    ttl.replace("front", "正面")
                    .replace("top", "顶视")
                    .replace("side", "侧视")
                    .replace("before", "执行前")
                    .replace("after", "执行后")
                    .replace("ee_xy", "末端轨迹")
                )
                ax.set_title(ttl, fontsize=8)
            except Exception:
                ax.text(0.5, 0.5, "加载失败", ha="center", va="center")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _trajectory_length(traj_xyz: List[Tuple[float, float, float]]) -> float:
    if len(traj_xyz) < 2:
        return 0.0
    dist = 0.0
    for i in range(1, len(traj_xyz)):
        x0, y0, z0 = traj_xyz[i - 1]
        x1, y1, z1 = traj_xyz[i]
        dist += math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    return float(dist)


def _save_success_failure_compare(success_img: str, failure_img: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, pth, ttl in [(axes[0], success_img, "成功案例"), (axes[1], failure_img, "失败案例")]:
        if pth and os.path.exists(pth):
            try:
                ax.imshow(Image.open(pth).convert("RGB"))
            except Exception:
                ax.text(0.5, 0.5, "加载失败", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "未找到", ha="center", va="center")
        ax.set_title(ttl)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _write_sim_metrics(rows: List[Dict[str, Any]], out_dir: str) -> None:
    if not rows:
        return
    sample_csv = os.path.join(out_dir, "sample_metrics_sim.csv")
    with open(sample_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    task_success = [_safe_float(r.get("task_success", 0.0)) for r in rows]
    sim_success = [_safe_float(r.get("sim_success", 0.0)) for r in rows]
    closed_loop_success = [_safe_float(r.get("closed_loop_success", 0.0)) for r in rows]
    collision = [_safe_float(r.get("collision_rate", 0.0)) for r in rows]
    tlen = [_safe_float(r.get("trajectory_length", 0.0)) for r in rows]
    et = [_safe_float(r.get("execution_time_sec", 0.0)) for r in rows]
    eef = [_safe_float(r.get("ee_final_error", 0.0)) for r in rows]
    gs = [_safe_float(r.get("grasp_stability", 0.0)) for r in rows]
    dr_enable = [_safe_float(r.get("enable_domain_randomization", 0.0)) for r in rows]
    dr_strength = [_safe_float(r.get("domain_randomization_strength", 0.0)) for r in rows]
    cam_jitter = [_safe_float(r.get("camera_pose_jitter", 0.0)) for r in rows]

    summary = {
        "n_samples": len(rows),
        "task_success_rate": float(sum(task_success) / len(task_success)) if task_success else 0.0,
        "sim_success_rate": float(sum(sim_success) / len(sim_success)) if sim_success else 0.0,
        "closed_loop_success_rate": float(sum(closed_loop_success) / len(closed_loop_success)) if closed_loop_success else 0.0,
        "collision_rate": float(sum(collision) / len(collision)) if collision else 0.0,
        "trajectory_length_mean": float(sum(tlen) / len(tlen)) if tlen else 0.0,
        "execution_time_mean": float(sum(et) / len(et)) if et else 0.0,
        "ee_final_error_mean": float(sum(eef) / len(eef)) if eef else 0.0,
        "grasp_stability_mean": float(sum(gs) / len(gs)) if gs else 0.0,
        "enable_domain_randomization_rate": float(sum(dr_enable) / len(dr_enable)) if dr_enable else 0.0,
        "domain_randomization_strength_mean": float(sum(dr_strength) / len(dr_strength)) if dr_strength else 0.0,
        "camera_pose_jitter_mean": float(sum(cam_jitter) / len(cam_jitter)) if cam_jitter else 0.0,
    }

    summary_csv = os.path.join(out_dir, "summary_metrics_sim.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(str(r.get("model", "")), []).append(r)
    per_model_rows: List[Dict[str, Any]] = []
    for m, arr in sorted(by_model.items(), key=lambda kv: kv[0]):
        ts = [_safe_float(x.get("task_success", 0.0)) for x in arr]
        ss = [_safe_float(x.get("sim_success", 0.0)) for x in arr]
        cs = [_safe_float(x.get("closed_loop_success", 0.0)) for x in arr]
        cr = [_safe_float(x.get("collision_rate", 0.0)) for x in arr]
        tl = [_safe_float(x.get("trajectory_length", 0.0)) for x in arr]
        ex = [_safe_float(x.get("execution_time_sec", 0.0)) for x in arr]
        ef = [_safe_float(x.get("ee_final_error", 0.0)) for x in arr]
        gb = [_safe_float(x.get("grasp_stability", 0.0)) for x in arr]
        de = [_safe_float(x.get("enable_domain_randomization", 0.0)) for x in arr]
        ds = [_safe_float(x.get("domain_randomization_strength", 0.0)) for x in arr]
        cj = [_safe_float(x.get("camera_pose_jitter", 0.0)) for x in arr]
        per_model_rows.append(
            {
                "model": m,
                "n_samples": len(arr),
                "task_success_rate": float(sum(ts) / len(ts)) if ts else 0.0,
                "sim_success_rate": float(sum(ss) / len(ss)) if ss else 0.0,
                "closed_loop_success_rate": float(sum(cs) / len(cs)) if cs else 0.0,
                "collision_rate": float(sum(cr) / len(cr)) if cr else 0.0,
                "trajectory_length_mean": float(sum(tl) / len(tl)) if tl else 0.0,
                "execution_time_mean": float(sum(ex) / len(ex)) if ex else 0.0,
                "ee_final_error_mean": float(sum(ef) / len(ef)) if ef else 0.0,
                "grasp_stability_mean": float(sum(gb) / len(gb)) if gb else 0.0,
                "enable_domain_randomization_rate": float(sum(de) / len(de)) if de else 0.0,
                "domain_randomization_strength_mean": float(sum(ds) / len(ds)) if ds else 0.0,
                "camera_pose_jitter_mean": float(sum(cj) / len(cj)) if cj else 0.0,
            }
        )

    by_model_csv = os.path.join(out_dir, "summary_metrics_sim_by_model.csv")
    with open(by_model_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_model_rows[0].keys()))
        w.writeheader()
        w.writerows(per_model_rows)


def run_minimal_sim(
    dataset_path: str,
    out_dir: str,
    debug_jsonl: str = "",
    max_samples: int = 6,
    gui: bool = False,
    seed: int = 42,
    enable_domain_randomization: bool = False,
    domain_randomization_strength: float = 0.2,
    camera_pose_jitter: float = 0.0,
    camera_width: int = 640,
    camera_height: int = 480,
    camera_calibration_out: str = "",
) -> str:
    _set_global_seed(seed)
    _ensure_dir(out_dir)
    fig_dir = os.path.join(out_dir, "figures")
    traj_dir = os.path.join(out_dir, "trajectories")
    _ensure_dir(fig_dir)
    _ensure_dir(traj_dir)

    dataset = _load_dataset(dataset_path)
    sample_map = {str(it.get("id", i + 1)): it for i, it in enumerate(dataset) if isinstance(it, dict)}

    debug_records = _rank_debug_records(_load_debug(debug_jsonl))
    queue: List[Dict[str, Any]] = []
    if debug_records:
        for r in debug_records:
            sid = str(r.get("sample_id", ""))
            if sid in sample_map:
                queue.append(r)
    else:
        for i, it in enumerate(dataset):
            queue.append({"sample_id": str(it.get("id", i + 1)), "model": "gt", "rep_actions": it.get("ground_truth", {}).get("expected_action_sequence", []) if isinstance(it.get("ground_truth"), dict) else []})

    sim = PyBulletMiniSim(
        gui=gui,
        seed=seed,
        enable_domain_randomization=bool(enable_domain_randomization),
        domain_randomization_strength=float(domain_randomization_strength),
        camera_pose_jitter=float(camera_pose_jitter),
    )
    sim.connect()

    calib_path = camera_calibration_out.strip() if isinstance(camera_calibration_out, str) else ""
    if not calib_path:
        calib_path = os.path.join(out_dir, "camera_calibration.json")
    sim.export_camera_calibration(calib_path, width=max(16, int(camera_width)), height=max(16, int(camera_height)))

    exported_for_grid: List[str] = []
    sim_metric_rows: List[Dict[str, Any]] = []
    success_after_img = ""
    failure_after_img = ""

    done = 0
    for rec in queue:
        if done >= max_samples:
            break
        sid = str(rec.get("sample_id", ""))
        model = str(rec.get("model", "gt"))
        sample = sample_map.get(sid)
        if not isinstance(sample, dict):
            continue

        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, numSolverIterations=80)
        p.setTimeStep(1.0 / 240.0)
        p.setGravity(0, 0, -9.81)

        table_pos = [0.55, 0.0, -0.65]
        if sim.enable_domain_randomization:
            jitter = 0.03 * sim.domain_randomization_strength
            table_pos[0] += sim.rng.uniform(-jitter, jitter)
            table_pos[1] += sim.rng.uniform(-jitter, jitter)

        sim.plane_id = p.loadURDF("plane.urdf")
        sim.table_id = p.loadURDF("table/table.urdf", basePosition=table_pos, useFixedBase=True)
        sim.robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
        sim.traj_xyz = []
        sim.attach_cid = None
        sim.attached_obj = None
        sim.collision_steps = 0
        sim.step_count = 0
        sim.grasp_stability_scores = []
        sim._reset_robot_pose()
        sim.apply_domain_randomization()
        sim.load_scene_from_sample(sample)

        front_before = sim.camera_image("front", width=max(16, int(camera_width)), height=max(16, int(camera_height)))
        top_before = sim.camera_image("top", width=max(16, int(camera_width)), height=max(16, int(camera_height)))
        action_seq = rec.get("rep_actions", []) if isinstance(rec.get("rep_actions"), list) else []
        if not action_seq:
            gt = sample.get("ground_truth", {}) if isinstance(sample.get("ground_truth"), dict) else {}
            action_seq = gt.get("expected_action_sequence", []) if isinstance(gt.get("expected_action_sequence"), list) else []

        t0 = time.perf_counter()
        sim.execute_sequence(action_seq)
        exec_sec = max(0.0, time.perf_counter() - t0)

        front_after = sim.camera_image("front", width=max(16, int(camera_width)), height=max(16, int(camera_height)))
        top_after = sim.camera_image("top", width=max(16, int(camera_width)), height=max(16, int(camera_height)))
        side_after = sim.camera_image("side", width=max(16, int(camera_width)), height=max(16, int(camera_height)))

        stem = f"{model}_sample_{sid}".replace("/", "_").replace(" ", "_")
        p1 = os.path.join(fig_dir, f"{stem}_front_before.png")
        p2 = os.path.join(fig_dir, f"{stem}_top_before.png")
        p3 = os.path.join(fig_dir, f"{stem}_front_after.png")
        p4 = os.path.join(fig_dir, f"{stem}_top_after.png")
        p5 = os.path.join(fig_dir, f"{stem}_side_after.png")
        front_before.save(p1)
        top_before.save(p2)
        front_after.save(p3)
        top_after.save(p4)
        side_after.save(p5)

        traj_path = os.path.join(traj_dir, f"{stem}_ee_xy.png")
        _save_traj_xy(sim.traj_xyz, traj_path, title=f"{model} 样本={sid} 末端轨迹")

        last_target = ""
        for step in reversed(action_seq):
            if isinstance(step, dict) and str(step.get("target", "")).strip():
                last_target = str(step.get("target", "")).strip()
                break
        tgt = sim._resolve_target(last_target) if last_target else None
        if tgt is not None:
            tx, ty, tz = sim._obj_pos(tgt)
            ee = p.getLinkState(sim.robot_id, sim.ee_link)[0]
            ee_final_error = math.sqrt((ee[0] - tx) ** 2 + (ee[1] - ty) ** 2 + (ee[2] - (tz + 0.08)) ** 2)
        else:
            ee_final_error = 0.0

        task_success = _safe_float(rec.get("task_success", 0.0))
        sim_success = 1.0 if (ee_final_error < 0.18 and (sim.attach_cid is None or _safe_float(np.mean(sim.grasp_stability_scores) if sim.grasp_stability_scores else 1.0) > 0.35)) else 0.0
        closed_loop_success = 1.0 if (task_success >= 0.5 and sim_success >= 0.5) else 0.0
        collision_rate = float(sim.collision_steps / max(sim.step_count, 1))
        traj_len = _trajectory_length(sim.traj_xyz)
        grasp_stability = float(np.mean(sim.grasp_stability_scores)) if sim.grasp_stability_scores else 0.0

        sim_metric_rows.append(
            {
                "model": model,
                "sample_id": sid,
                "difficulty_level": _safe_int(rec.get("difficulty_level", sample.get("difficulty_level", 0))),
                "task_success": task_success,
                "plan_success": task_success,
                "sim_success": sim_success,
                "closed_loop_success": closed_loop_success,
                "collision_rate": collision_rate,
                "trajectory_length": traj_len,
                "execution_time": float(exec_sec),
                "execution_time_sec": float(exec_sec),
                "ee_final_error": float(ee_final_error),
                "grasp_stability": grasp_stability,
                "avg_inference_time": _safe_float(rec.get("avg_inference_time", 0.0)),
                "illegal_action_rate": _safe_float(rec.get("illegal_action_rate", 0.0)),
                "target_mismatch_rate": _safe_float(rec.get("target_mismatch_rate", 0.0)),
                "enable_domain_randomization": int(1 if sim.enable_domain_randomization else 0),
                "domain_randomization_strength": float(sim.domain_randomization_strength),
                "camera_pose_jitter": float(sim.camera_pose_jitter),
                "camera_calibration_path": calib_path,
                "n_actions": len(action_seq),
            }
        )

        if sim_success >= 0.5 and not success_after_img:
            success_after_img = p3
        if sim_success < 0.5 and not failure_after_img:
            failure_after_img = p3

        exported_for_grid.append(p3)
        done += 1

    sim.close()

    _write_sim_metrics(sim_metric_rows, out_dir=out_dir)

    grid_path = os.path.join(fig_dir, "paper_grid_2x3.png")
    _save_grid_2x3(exported_for_grid, grid_path)

    if success_after_img or failure_after_img:
        cmp_path = os.path.join(fig_dir, "success_failure_compare.png")
        _save_success_failure_compare(success_after_img, failure_after_img, cmp_path)

    return grid_path


def main() -> None:
    here = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(here, "ground_truth_dataset_v3.json"))
    parser.add_argument("--debug_jsonl", default="", help="Optional debug.jsonl from evaluator output")
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(here), "results", "pybullet_sim_minimal"))
    parser.add_argument("--max_samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--enable_domain_randomization", action="store_true")
    parser.add_argument("--domain_randomization_strength", type=float, default=0.2)
    parser.add_argument("--camera_pose_jitter", type=float, default=0.0)
    parser.add_argument("--camera_width", type=int, default=640)
    parser.add_argument("--camera_height", type=int, default=480)
    parser.add_argument("--camera_calibration_out", default="")
    args = parser.parse_args()

    grid = run_minimal_sim(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        debug_jsonl=args.debug_jsonl,
        max_samples=max(1, int(args.max_samples)),
        gui=bool(args.gui),
        seed=int(args.seed),
        enable_domain_randomization=bool(args.enable_domain_randomization),
        domain_randomization_strength=float(args.domain_randomization_strength),
        camera_pose_jitter=float(args.camera_pose_jitter),
        camera_width=max(16, int(args.camera_width)),
        camera_height=max(16, int(args.camera_height)),
        camera_calibration_out=str(args.camera_calibration_out),
    )
    print(grid)


if __name__ == "__main__":
    main()
