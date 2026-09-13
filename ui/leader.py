from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig

# ======Settings=========
LEADER_ID = "robosim_leader"
LEADER_TYPE = "so100_leader"
PORT_PREFIXES = ("/dev/ttyACM", "/dev/ttyUSB")
ACTION_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
FULL_TURN_MOTOR = "wrist_roll"
RANGE_POLL_S = 0.02
PORT_SETTLE_S = 0.5
SERIAL_CACHE_S = 1.0
_SERIAL_CACHE: tuple[float, list[dict]] = (0.0, [])
_SERIAL_LOCK = threading.Lock()
STEP_HELP = {
    "idle": [
        "Plug in only the lead arm USB and give it power.",
        "Do not unplug it yet.",
        "Click Start detect.",
    ],
    "detect_unplug": [
        "Unplug the USB cable of the lead arm.",
        "Leave every other USB device connected.",
        "Click USB unplugged.",
    ],
    "detect_replug": [
        "Plug the same USB cable back into the lead arm.",
        "Wait until the port comes back.",
        "Click USB plugged in.",
    ],
    "detected": [
        "This port disappeared when you unplugged the lead.",
        "Click Connect to open it.",
    ],
    "connected": [
        "If this is the same arm as last time, use the saved calibration.",
        "Otherwise run a new calibration.",
    ],
    "calibrate_home": [
        "Torque is off. Move the lead to the middle of every joint range.",
        "Hold that pose.",
        "Click Home is set.",
    ],
    "calibrate_range": [
        f"Move every joint except {FULL_TURN_MOTOR} through its full range.",
        "Keep moving until you click Finish calibration.",
    ],
}
# ======Settings=========


def _hwid_key(hwid: str) -> str:
    parts = []
    for token in str(hwid or "").split():
        if token.startswith(("VID:PID=", "SER=", "LOCATION=")):
            if token.startswith("LOCATION="):
                continue
            parts.append(token)
    return " ".join(parts) or str(hwid or "")


def _tty_serial(name: str) -> str:
    for rel in (
        f"/sys/class/tty/{name}/device/serial",
        f"/sys/class/tty/{name}/device/../serial",
    ):
        path = Path(rel)
        try:
            if path.is_file():
                text = path.read_text().strip()
                if text:
                    return text
        except OSError:
            continue
    return ""


def list_serial_candidates(*, fresh: bool = False) -> list[dict]:
    global _SERIAL_CACHE
    now = time.monotonic()
    if not fresh:
        with _SERIAL_LOCK:
            stamp, cached = _SERIAL_CACHE
            if cached and now - stamp < SERIAL_CACHE_S:
                return list(cached)
    found = []
    root = Path("/dev")
    for prefix in PORT_PREFIXES:
        for path in sorted(root.glob(Path(prefix).name + "*")):
            port = str(path)
            found.append(
                {
                    "port": port,
                    "serial": _tty_serial(path.name),
                    "hwid": "",
                    "label": path.name,
                }
            )
    with _SERIAL_LOCK:
        _SERIAL_CACHE = (time.monotonic(), found)
    return list(found)


def load_saved_identity(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def identity_matches(saved: dict | None, candidate: dict | None) -> bool:
    if not saved or not candidate:
        return False
    saved_serial = str(saved.get("serial") or "")
    cand_serial = str(candidate.get("serial") or "")
    if saved_serial and cand_serial:
        return saved_serial == cand_serial
    saved_hwid = _hwid_key(str(saved.get("hwid") or ""))
    cand_hwid = _hwid_key(str(candidate.get("hwid") or ""))
    return bool(saved_hwid and cand_hwid and saved_hwid == cand_hwid)


class LeaderSession:
    def __init__(self, collect_dir: Path) -> None:
        self.collect_dir = collect_dir
        self.calibration_dir = collect_dir / "calibration"
        self.identity_path = collect_dir / "leader.json"
        self.lock = threading.Lock()
        self.leader: SO100Leader | None = None
        self.candidate: dict | None = None
        self.step = "idle"
        self.message = ""
        self._range_stop = threading.Event()
        self._range_thread: threading.Thread | None = None
        self._range_mins: dict[str, int] = {}
        self._range_maxes: dict[str, int] = {}
        self._range_pos: dict[str, int] = {}
        self._homing: dict[str, int] = {}
        self._ports_before: list[str] = []

    def _saved(self) -> dict | None:
        return load_saved_identity(self.identity_path)

    def _calibration_file(self) -> Path:
        return self.calibration_dir / f"{LEADER_ID}.json"

    def _make(self, port: str) -> SO100Leader:
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        return SO100Leader(
            SO100LeaderConfig(
                port=port,
                id=LEADER_ID,
                calibration_dir=self.calibration_dir,
            )
        )

    def _close_leader(self) -> None:
        self._stop_range_locked()
        if self.leader is None:
            return
        try:
            if self.leader.is_connected:
                self.leader.disconnect()
        except Exception:
            pass
        self.leader = None

    def _motor_ports(self) -> list[str]:
        return [item["port"] for item in list_serial_candidates(fresh=True)]

    def _status_locked(self, ports: list[dict] | None = None) -> dict:
        saved = self._saved()
        match = identity_matches(saved, self.candidate)
        return {
            "step": self.step,
            "message": self.message,
            "help": list(STEP_HELP.get(self.step) or STEP_HELP["idle"]),
            "ports": list_serial_candidates() if ports is None else ports,
            "connected": bool(self.leader is not None and self.leader.is_connected),
            "candidate": self.candidate,
            "savedCalibration": None
                if saved is None
                else {
                    "exists": self._calibration_file().is_file(),
                    "match": match,
                    "serial": saved.get("serial") or "",
                    "port": saved.get("port") or "",
                },
            "motors": [],
        }

    def _calibration_motors(self) -> list[dict]:
        with self.lock:
            step = self.step
            leader = self.leader
            mins = dict(self._range_mins)
            maxes = dict(self._range_maxes)
            cached = dict(self._range_pos)
            connected = leader is not None and leader.is_connected
        if step == "calibrate_range":
            names = list(cached) or list(mins)
            return [
                {
                    "name": name,
                    "pos": cached.get(name),
                    "min": mins.get(name),
                    "max": maxes.get(name),
                }
                for name in names
            ]
        if step != "calibrate_home" or not connected:
            return []
        try:
            positions = leader.bus.sync_read(
                "Present_Position",
                normalize=False,
                num_retry=2,
            )
        except Exception:
            return []
        return [
            {
                "name": name,
                "pos": int(value),
                "min": None,
                "max": None,
            }
            for name, value in positions.items()
        ]

    def status(self, *, scan: bool = True) -> dict:
        ports = list_serial_candidates() if scan else []
        motors = self._calibration_motors() if scan else []
        with self.lock:
            payload = self._status_locked(ports)
        payload["motors"] = motors
        return payload

    def _candidate_for_port(self, port: str) -> dict:
        for item in list_serial_candidates(fresh=True):
            if item["port"] == port:
                return item
        return {"port": port, "serial": "", "hwid": "", "label": port}

    def detect_start(self) -> dict:
        with self.lock:
            self._close_leader()
            self.candidate = None
        ports = self._motor_ports()
        with self.lock:
            self._ports_before = ports
            if not self._ports_before:
                self.step = "idle"
                self.message = "No serial ports found. Plug the lead USB in first."
                return self._status_locked()
            self.step = "detect_unplug"
            self.message = (
                "Saw "
                + ", ".join(self._ports_before)
                + ". Unplug the lead USB now."
            )
            return self._status_locked()

    def detect_unplug(self) -> dict:
        time.sleep(PORT_SETTLE_S)
        after = self._motor_ports()
        with self.lock:
            gone = sorted(set(self._ports_before) - set(after))
            if len(gone) == 0:
                self.step = "detect_unplug"
                self.message = (
                    "No port disappeared. Unplug the lead USB, then click again."
                )
                return self._status_locked()
            if len(gone) > 1:
                self.step = "idle"
                self.message = (
                    "More than one port disappeared. Unplug only the lead, then start again."
                )
                return self._status_locked()
            self.candidate = self._candidate_for_port(gone[0])
            self.step = "detect_replug"
            self.message = f"Lead port is {gone[0]}. Plug that USB back in."
            return self._status_locked()

    def detect_replug(self) -> dict:
        time.sleep(PORT_SETTLE_S)
        now = self._motor_ports()
        with self.lock:
            if self.candidate is None:
                raise RuntimeError("start detect first")
            port = self.candidate["port"]
            if port not in now:
                self.step = "detect_replug"
                self.message = f"{port} is still missing. Plug the USB back in."
                return self._status_locked()
            self.candidate = self._candidate_for_port(port)
            saved = self._saved()
            match = identity_matches(saved, self.candidate)
            self.step = "detected"
            if saved and match:
                self.message = "Lead arm is back. Saved calibration matches this arm."
            elif saved:
                self.message = "Lead arm is back. Saved calibration is from another arm."
            else:
                self.message = "Lead arm is back."
            return self._status_locked()

    def connect(self) -> dict:
        with self.lock:
            if self.candidate is None:
                raise RuntimeError("detect the lead arm first")
            self._close_leader()
            self.leader = self._make(self.candidate["port"])
            self.leader.connect(calibrate=False)
            self.step = "connected"
            self.message = "Connected. Calibrate or use the saved file."
            return self._status_locked()

    def use_saved(self) -> dict:
        with self.lock:
            if self.leader is None or not self.leader.is_connected:
                raise RuntimeError("connect the lead arm first")
            if self.candidate is None:
                raise RuntimeError("detect the lead arm first")
            saved = self._saved()
            if not identity_matches(saved, self.candidate):
                raise RuntimeError("saved calibration is not from this arm")
            if not self.leader.calibration:
                raise RuntimeError("no saved calibration file")
            self.leader.bus.write_calibration(self.leader.calibration)
            self.leader.configure()
            self.step = "ready"
            self.message = "Saved calibration loaded."
            return self._status_locked()

    def start_calibrate(self) -> dict:
        with self.lock:
            if self.leader is None or not self.leader.is_connected:
                raise RuntimeError("connect the lead arm first")
            self._stop_range_locked()
            self.leader.bus.disable_torque()
            for motor in self.leader.bus.motors:
                self.leader.bus.write(
                    "Operating_Mode",
                    motor,
                    OperatingMode.POSITION.value,
                )
            self.step = "calibrate_home"
            self.message = "Move the arm to the middle of its range, then set home."
            return self._status_locked()

    def set_home(self) -> dict:
        with self.lock:
            if self.leader is None or not self.leader.is_connected:
                raise RuntimeError("connect the lead arm first")
            if self.step != "calibrate_home":
                raise RuntimeError("start calibration first")
            self._homing = self.leader.bus.set_half_turn_homings()
            self.step = "calibrate_range"
            self.message = (
                f"Move every joint except {FULL_TURN_MOTOR} through its range, then finish."
            )
            return self._status_locked()

    def _range_loop(self, motors: list[str]) -> None:
        tracked = set(motors)
        while not self._range_stop.is_set():
            try:
                with self.lock:
                    leader = self.leader
                    if leader is None or not leader.is_connected:
                        break
                positions = leader.bus.sync_read(
                    "Present_Position",
                    normalize=False,
                    num_retry=5,
                )
                with self.lock:
                    for motor, value in positions.items():
                        raw = int(value)
                        self._range_pos[motor] = raw
                        if motor not in tracked:
                            continue
                        self._range_mins[motor] = min(
                            self._range_mins.get(motor, raw), raw
                        )
                        self._range_maxes[motor] = max(
                            self._range_maxes.get(motor, raw), raw
                        )
            except Exception:
                time.sleep(RANGE_POLL_S)
                continue
            time.sleep(RANGE_POLL_S)

    def _stop_range_locked(self) -> None:
        self._range_stop.set()
        self._range_thread = None

    def _join_range(self, thread: threading.Thread | None) -> None:
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)

    def start_range(self) -> dict:
        with self.lock:
            if self.leader is None or not self.leader.is_connected:
                raise RuntimeError("connect the lead arm first")
            if self.step != "calibrate_range":
                raise RuntimeError("set home first")
            old = self._range_thread
            self._range_stop.set()
            self._range_thread = None
        self._join_range(old)
        with self.lock:
            if self.leader is None or not self.leader.is_connected:
                raise RuntimeError("connect the lead arm first")
            motors = [
                motor
                for motor in self.leader.bus.motors
                if motor != FULL_TURN_MOTOR
            ]
            start = self.leader.bus.sync_read(
                "Present_Position",
                normalize=False,
                num_retry=5,
            )
            self._range_pos = {motor: int(value) for motor, value in start.items()}
            self._range_mins = {motor: int(start[motor]) for motor in motors}
            self._range_maxes = dict(self._range_mins)
            self._range_stop.clear()
            self._range_thread = threading.Thread(
                target=self._range_loop,
                args=(motors,),
                daemon=True,
            )
            self._range_thread.start()
            self.message = "Recording joint ranges…"
            return self._status_locked()

    def finish_calibrate(self) -> dict:
        with self.lock:
            if self.leader is None or not self.leader.is_connected:
                raise RuntimeError("connect the lead arm first")
            if self.step != "calibrate_range":
                raise RuntimeError("set home first")
            old = self._range_thread
            self._range_stop.set()
            self._range_thread = None
        self._join_range(old)
        with self.lock:
            mins = dict(self._range_mins)
            maxes = dict(self._range_maxes)
            for motor, raw in self._range_pos.items():
                if motor not in mins:
                    continue
                mins[motor] = min(mins[motor], raw)
                maxes[motor] = max(maxes[motor], raw)
            same = [motor for motor in mins if mins[motor] == maxes[motor]]
            if same:
                raise RuntimeError(
                    "some joints did not move: " + ", ".join(same)
                )
            mins[FULL_TURN_MOTOR] = 0
            maxes[FULL_TURN_MOTOR] = 4095
            calibration = {}
            for motor, spec in self.leader.bus.motors.items():
                calibration[motor] = MotorCalibration(
                    id=spec.id,
                    drive_mode=0,
                    homing_offset=int(self._homing[motor]),
                    range_min=int(mins[motor]),
                    range_max=int(maxes[motor]),
                )
            self.leader.calibration = calibration
            self.leader.bus.write_calibration(calibration)
            self.leader._save_calibration()
            self.leader.configure()
            if self.candidate is None:
                raise RuntimeError("detect the lead arm first")
            self.collect_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "id": LEADER_ID,
                "type": LEADER_TYPE,
                "port": self.candidate["port"],
                "serial": self.candidate.get("serial") or "",
                "hwid": self.candidate.get("hwid") or "",
            }
            self.identity_path.write_text(json.dumps(payload, indent=2))
            self.step = "ready"
            self.message = "Calibration saved."
            return self._status_locked()

    def action_vector(self) -> list[float]:
        with self.lock:
            if self.leader is None or not self.leader.is_connected:
                raise RuntimeError("lead arm is not connected")
            if self.step != "ready":
                raise RuntimeError("calibrate the lead arm first")
            action = self.leader.get_action()
            return [float(action[key]) for key in ACTION_KEYS]
