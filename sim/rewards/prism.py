from dataclasses import dataclass

import mujoco
import numpy as np

# ======Settings=========
FIXED_INNER_PADS = tuple(f"fixed_jaw_pad_{index}" for index in range(1, 5))
MOVING_INNER_PADS = tuple(f"moving_jaw_pad_{index}" for index in range(1, 5))
TRAY_GEOMS = (
    "tray_floor",
    "tray_wall_n",
    "tray_wall_s",
    "tray_wall_e",
    "tray_wall_w",
)
TCP_SITE = "tcp"
# ======Settings=========


@dataclass
class RawRolloutReward:
    success: bool
    success_time_seconds: float | None
    impulse_by_entity: dict[str, float]
    total_impulse: float
    target_contact_seconds: float
    target_proximity_seconds: float
    target_distance_m: float
    proximity_fraction: float
    progress_fraction: float
    grasp_count: int
    contact_reward: float
    proximity_reward: float
    progress_reward: float
    grasp_reward: float
    success_reward: float


def _named_ids(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    names: tuple[str, ...],
) -> set[int]:
    result = set()
    for name in names:
        object_id = mujoco.mj_name2id(model, object_type, name)
        if object_id >= 0:
            result.add(object_id)
    return result


def _body_descendants(model: mujoco.MjModel, root_body_id: int) -> set[int]:
    descendants = set()
    for body_id in range(model.nbody):
        current = body_id
        while current > 0:
            if current == root_body_id:
                descendants.add(body_id)
                break
            current = int(model.body_parentid[current])
    return descendants


def _body_geoms(model: mujoco.MjModel, body_ids: set[int]) -> set[int]:
    return {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
    }


class PrismRewardTracker:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        scene_metadata: dict,
        reward_config: dict,
        max_duration_seconds: float,
    ):
        self.model = model
        self.data = data
        self.config = reward_config
        self.max_duration_seconds = max_duration_seconds
        self.target_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target",
        )
        self.target_geoms = _body_geoms(
            model,
            _body_descendants(model, self.target_body_id),
        )
        self.tcp_site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            TCP_SITE,
        )
        if self.tcp_site_id < 0:
            raise RuntimeError(f"robot model has no {TCP_SITE} site")
        robot_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "Base",
        )
        moving_robot_bodies = _body_descendants(
            model,
            robot_body_id,
        ) - {robot_body_id}
        self.robot_geoms = _body_geoms(model, moving_robot_bodies)
        self.fixed_pad_geoms = _named_ids(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            FIXED_INNER_PADS,
        )
        self.moving_pad_geoms = _named_ids(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            MOVING_INNER_PADS,
        )
        self.table_geoms = _named_ids(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            ("table",),
        )
        self.tray_geoms = _named_ids(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            TRAY_GEOMS,
        )
        self.distractor_geoms = {}
        for index, name in enumerate(scene_metadata["distractors"]):
            body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"distractor_{index}",
            )
            self.distractor_geoms[name] = _body_geoms(
                model,
                _body_descendants(model, body_id),
            )

        tray = scene_metadata["tray"]
        self.tray_center = np.asarray(tray["center_xy"], dtype=np.float64)
        self.tray_half_size = np.asarray(
            [tray["inner_width"] / 2.0, tray["inner_depth"] / 2.0],
            dtype=np.float64,
        )
        self.tray_floor_z = float(tray["floor_thickness"])
        self.tray_top_z = self.tray_floor_z + float(tray["wall_height"])
        self.elapsed_seconds = 0.0
        self.impulse_by_entity = {
            **{name: 0.0 for name in self.distractor_geoms},
            "table": 0.0,
            "tray": 0.0,
        }
        self.target_contact_seconds = 0.0
        self.target_proximity_seconds = 0.0
        self.target_distance_m = self._tcp_to_target_distance()
        self.proximity_fraction = 0.0
        self.grasp_contact_seconds = 0.0
        self.grasp_loss_seconds = 0.0
        self.grasp_armed = True
        self.grasp_count = 0
        self.success_stable_seconds = 0.0
        self.success = False
        self.success_time_seconds = None
        self.initial_distance = self._distance_to_tray()
        self.best_distance = self.initial_distance

    @staticmethod
    def _pair_matches(
        geom_a: int,
        geom_b: int,
        first: set[int],
        second: set[int],
    ) -> bool:
        return (
            geom_a in first
            and geom_b in second
            or geom_b in first
            and geom_a in second
        )

    def _distance_to_tray(self) -> float:
        position = np.asarray(self.data.xpos[self.target_body_id])
        delta_xy = np.maximum(
            np.abs(position[:2] - self.tray_center) - self.tray_half_size,
            0.0,
        )
        delta_z = max(
            self.tray_floor_z - position[2],
            position[2] - self.tray_top_z,
            0.0,
        )
        return float(np.linalg.norm([delta_xy[0], delta_xy[1], delta_z]))

    def _tcp_to_target_distance(self) -> float:
        return float(
            np.linalg.norm(
                self.data.site_xpos[self.tcp_site_id]
                - self.data.xpos[self.target_body_id]
            )
        )

    def _target_inside_tray(self) -> bool:
        position = np.asarray(self.data.xpos[self.target_body_id])
        inside_xy = np.all(
            np.abs(position[:2] - self.tray_center) <= self.tray_half_size
        )
        return bool(
            inside_xy
            and self.tray_floor_z - 0.02
            <= position[2]
            <= self.tray_top_z + 0.03
        )

    def update(self, timestep_seconds: float) -> None:
        self.elapsed_seconds += timestep_seconds
        fixed_contact = False
        moving_contact = False
        force = np.zeros(6, dtype=np.float64)

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom_a = int(contact.geom1)
            geom_b = int(contact.geom2)
            mujoco.mj_contactForce(
                self.model,
                self.data,
                contact_index,
                force,
            )
            impulse = float(np.linalg.norm(force[:3]) * timestep_seconds)

            fixed_contact |= self._pair_matches(
                geom_a,
                geom_b,
                self.target_geoms,
                self.fixed_pad_geoms,
            )
            moving_contact |= self._pair_matches(
                geom_a,
                geom_b,
                self.target_geoms,
                self.moving_pad_geoms,
            )

            if self._pair_matches(
                geom_a,
                geom_b,
                self.robot_geoms,
                self.table_geoms,
            ):
                self.impulse_by_entity["table"] += impulse
            if self._pair_matches(
                geom_a,
                geom_b,
                self.robot_geoms,
                self.tray_geoms,
            ):
                self.impulse_by_entity["tray"] += impulse

            for name, distractor_geoms in self.distractor_geoms.items():
                robot_hit = self._pair_matches(
                    geom_a,
                    geom_b,
                    self.robot_geoms,
                    distractor_geoms,
                )
                target_hit = (
                    self.config["impulse"]["include_target_to_distractor"]
                    and self._pair_matches(
                        geom_a,
                        geom_b,
                        self.target_geoms,
                        distractor_geoms,
                    )
                )
                if robot_hit or target_hit:
                    self.impulse_by_entity[name] += impulse

        any_inner_contact = fixed_contact or moving_contact
        if any_inner_contact:
            self.target_contact_seconds += timestep_seconds
        self.target_distance_m = self._tcp_to_target_distance()
        proximity_decay = float(
            self.config["target_proximity"]["decay_distance_m"]
        )
        self.proximity_fraction = float(
            np.exp(-self.target_distance_m / proximity_decay)
        )
        self.target_proximity_seconds += (
            self.proximity_fraction * timestep_seconds
        )

        valid_grasp_contact = fixed_contact and moving_contact
        grasp_config = self.config["grasp"]
        if valid_grasp_contact:
            self.grasp_loss_seconds = 0.0
            self.grasp_contact_seconds += timestep_seconds
            if (
                self.grasp_armed
                and self.grasp_contact_seconds
                >= float(grasp_config["required_contact_seconds"])
            ):
                self.grasp_count += 1
                self.grasp_armed = False
        else:
            self.grasp_contact_seconds = 0.0
            if not self.grasp_armed:
                self.grasp_loss_seconds += timestep_seconds
                if self.grasp_loss_seconds >= float(
                    grasp_config["loss_reset_seconds"]
                ):
                    self.grasp_armed = True

        self.best_distance = min(
            self.best_distance,
            self._distance_to_tray(),
        )
        linear_speed = float(
            np.linalg.norm(self.data.cvel[self.target_body_id, 3:])
        )
        success_config = self.config["success"]
        stable = (
            self._target_inside_tray()
            and not any_inner_contact
            and linear_speed
            <= float(success_config["max_linear_speed_m_s"])
        )
        self.success_stable_seconds = (
            self.success_stable_seconds + timestep_seconds
            if stable
            else 0.0
        )
        if (
            not self.success
            and self.success_stable_seconds
            >= float(success_config["stable_seconds"])
        ):
            self.success = True
            self.success_time_seconds = self.elapsed_seconds

    def finalize(self) -> RawRolloutReward:
        contact_config = self.config["target_contact"]
        contact_reward = (
            min(
                self.target_contact_seconds,
                float(contact_config["max_seconds"]),
            )
            / float(contact_config["max_seconds"])
            * float(contact_config["max_reward"])
        )
        proximity_config = self.config["target_proximity"]
        proximity_reward = (
            min(
                self.target_proximity_seconds,
                float(proximity_config["max_seconds"]),
            )
            / float(proximity_config["max_seconds"])
            * float(proximity_config["max_reward"])
        )
        progress_fraction = float(
            np.clip(
                (self.initial_distance - self.best_distance)
                / max(self.initial_distance, 1e-8),
                0.0,
                1.0,
            )
        )
        progress_reward = (
            progress_fraction
            * float(self.config["progress"]["max_reward"])
        )
        grasp_reward = min(
            self.grasp_count
            * float(self.config["grasp"]["reward_per_grasp"]),
            float(self.config["grasp"]["max_reward"]),
        )
        success_reward = 0.0
        if self.success:
            success_config = self.config["success"]
            speed_fraction = 1.0 - min(
                float(self.success_time_seconds)
                / self.max_duration_seconds,
                1.0,
            )
            success_reward = float(
                success_config["base_reward"]
            ) + speed_fraction * float(success_config["max_speed_bonus"])

        total_impulse = float(sum(self.impulse_by_entity.values()))
        return RawRolloutReward(
            success=self.success,
            success_time_seconds=self.success_time_seconds,
            impulse_by_entity=dict(self.impulse_by_entity),
            total_impulse=total_impulse,
            target_contact_seconds=self.target_contact_seconds,
            target_proximity_seconds=self.target_proximity_seconds,
            target_distance_m=self.target_distance_m,
            proximity_fraction=self.proximity_fraction,
            progress_fraction=progress_fraction,
            grasp_count=self.grasp_count,
            contact_reward=contact_reward,
            proximity_reward=proximity_reward,
            progress_reward=progress_reward,
            grasp_reward=grasp_reward,
            success_reward=success_reward,
        )


def score_group(
    raw_rewards: list[RawRolloutReward],
    reward_config: dict,
) -> list[dict]:
    impulses = np.asarray(
        [reward.total_impulse for reward in raw_rewards],
        dtype=np.float64,
    )
    positive_impulses = impulses[impulses > 0.0]
    denominator = (
        float(
            np.percentile(
                positive_impulses,
                float(reward_config["impulse"]["normalization_percentile"]),
            )
        )
        if len(positive_impulses)
        else 1.0
    )
    max_penalty = float(reward_config["impulse"]["max_penalty"])
    results = []
    for raw_reward in raw_rewards:
        impulse_penalty = (
            min(raw_reward.total_impulse / max(denominator, 1e-8), 1.0)
            * max_penalty
        )
        total = (
            raw_reward.contact_reward
            + raw_reward.proximity_reward
            + raw_reward.progress_reward
            + raw_reward.grasp_reward
            + raw_reward.success_reward
            - impulse_penalty
        )
        results.append(
            {
                **raw_reward.__dict__,
                "impulse_normalizer": denominator,
                "impulse_penalty": impulse_penalty,
                "total_reward": total,
            }
        )
    return results
