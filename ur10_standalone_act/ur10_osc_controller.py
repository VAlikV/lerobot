"""
Process-based OSC (task-space PD) torque controller for the UR10e.

Rebuilt after diffusion_policy's `RTDEInterpolationController`, which ran perfectly
on another UR arm. The control LAW is the same task-space PD via Jacobian transpose
we already had; what makes it work is the ARCHITECTURE:

  - Runs in its OWN `multiprocessing.Process` with real-time (SCHED_RR) priority,
    so the 500 Hz directTorque loop gets predictable, GIL-free timing. (Our earlier
    thread-in-the-gamepad-process version jittered -> torque vibration.)
  - PURE PD: tau = Jᵀ (Kp·e − Kd·ẋ). NO integral, NO disturbance observer, NO
    friction comp. Those were the source of our limit-cycle and velocity-noise
    vibration; the reference simply accepts the small steady-state PD error.
  - Kd = 2·√(Kp)·ζ (unit-mass heuristic). Deliberately LOW — high Kd amplifies
    RTDE's noisy/delayed q̇ and re-introduces vibration.
  - Gravity comp via the robot: by default we DON'T override the pendant payload
    (set_payload=False); set it on the teach pendant (Payload→Measure).

IPC: a tiny shared-memory contract (no diffusion_policy deps):
  - `_tgt`     mp.Array('d', 8): [0:6]=EE target pose [x,y,z,rx,ry,rz] (base, rotvec),
               [6]=gripper cmd (0=open, 1=close), [7]=valid flag (0 until first target).
               (NOT named `_target` — that attribute is reserved/deleted by mp.Process.)
  - `_state`   mp.Array('d', 19): [0:6]=q, [6:12]=q̇, [12:18]=actual TCP pose, [18]=stamp.

Main-process API mirrors the reference: start()/stop()/set_target()/get_state(),
plus context-manager support.

Requirements: UR10e e-Series, PolyScope >= 5.23 (directTorque), ur_rtde, numpy, scipy.
Robot in Remote Control mode, safety NORMAL. Run with the `lerobot` conda env.
"""

import os
import time
import multiprocessing as mp

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from ur10_kinematics import fk_jacobian, pose_to_T, pose_error


class UR10OSCController(mp.Process):
    def __init__(self, robot_ip,
                 frequency=500,
                 kp_pos=1000.0, kp_rot=50.0,
                 damping_ratio_pos=1.0, damping_ratio_rot=1.0,
                 error_delta_pos=0.05, error_delta_rot=0.3,
                 torque_max=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
                 tcp_offset=None,
                 set_payload=False, payload_mass=1.3, payload_cog=(0.0, 0.0, 0.0),
                 gripper_port="/dev/ttyACM0", gripper_baudrate=115200, use_gripper=False,
                 use_dob=False, dob_g=4.0, dob_inertia=0.08, dob_damping=0.3, dob_clip=40.0,
                 joints_init=None, joints_init_speed=1.05,
                 soft_real_time=True, launch_timeout=5.0, verbose=False):
        super().__init__(name="UR10OSCController")
        self.robot_ip = robot_ip
        self.frequency = int(frequency)
        self.dt = 1.0 / frequency

        stiffness = np.array([kp_pos] * 3 + [kp_rot] * 3)
        self.Kp = np.diag(stiffness)
        # Unit-mass critical-damping heuristic (LOW Kd on purpose — see module docstring).
        self.Kd = np.diag(2.0 * np.sqrt(stiffness) * np.array(
            [damping_ratio_pos] * 3 + [damping_ratio_rot] * 3))
        self.error_delta = np.array([error_delta_pos] * 3 + [error_delta_rot] * 3)
        self.torque_max = np.array(torque_max, dtype=float)

        self.tcp_offset = np.array(tcp_offset) if tcp_offset is not None else None
        self.T_tcp = pose_to_T(self.tcp_offset) if tcp_offset is not None else None
        self.set_payload = bool(set_payload)
        self.payload_mass = float(payload_mass)
        self.payload_cog = list(payload_cog)

        self.gripper_port = gripper_port
        self.gripper_baudrate = gripper_baudrate
        self.use_gripper = bool(use_gripper)

        # Optional joint-space disturbance observer (default OFF — finalized controller is
        # pure PD per the reference). Estimates per-joint disturbance torque (mostly dry
        # friction) via a derivative-free Q-filter and feeds it forward to shrink the PD
        # steady-state error. Velocity-based (uses q̇), so watch for waviness; the clean
        # real-time process removes the timing jitter that hurt it in the old thread build.
        self.use_dob = bool(use_dob)
        self.dob_g = float(dob_g)
        self.dob_J = float(dob_inertia)
        self.dob_b = float(dob_damping)
        self.dob_clip = float(dob_clip)

        self.joints_init = None if joints_init is None else np.array(joints_init, dtype=float)
        self.joints_init_speed = float(joints_init_speed)
        self.soft_real_time = bool(soft_real_time)
        self.launch_timeout = float(launch_timeout)
        self.verbose = bool(verbose)

        # Shared memory (created in parent, inherited by child).
        # NOTE: do NOT name this `_target` — mp.Process reserves/deletes that attribute.
        self._tgt = mp.Array('d', 8)   # [pose(6), gripper, valid]
        # [q(6), qd(6), tcp(6), stamp, dob_dhat_norm, safety]; safety: 0=ok 1=protective 2=estop
        self._state = mp.Array('d', 21)
        self._ready = mp.Event()
        self._stop = mp.Event()

    # ---- main-process API --------------------------------------------------
    def start(self, wait=True):
        super().start()
        if wait:
            self._ready.wait(self.launch_timeout)
            assert self.is_alive(), "controller process died during launch"

    def stop(self, wait=True):
        self._stop.set()
        if wait and self.is_alive():
            self.join(timeout=3.0)

    @property
    def is_ready(self):
        return self._ready.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def set_target(self, pose, close_gripper=False):
        """Set the absolute EE target [x,y,z,rx,ry,rz] (base frame, rotvec). Non-blocking."""
        pose = np.asarray(pose, dtype=float).reshape(6)
        with self._tgt.get_lock():
            self._tgt[:6] = pose
            self._tgt[6] = 1.0 if close_gripper else 0.0
            self._tgt[7] = 1.0  # valid
        return pose

    def get_state(self):
        """Latest robot state as a dict (q, qd, tcp pose, timestamp)."""
        with self._state.get_lock():
            s = np.array(self._state[:])
        return {"q": s[0:6], "qd": s[6:12], "tcp": s[12:18], "stamp": s[18],
                "dhat": s[19], "safety": int(s[20])}

    def get_current_tcp(self):
        return self.get_state()["tcp"]

    # ---- child process -----------------------------------------------------
    def run(self):
        if self.soft_real_time:
            try:
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            except (PermissionError, OSError) as e:
                print(f"[osc] soft_real_time scheduling unavailable ({e}); "
                      "run with sudo/rtprio for best timing. Continuing normal priority.")

        from rtde_control import RTDEControlInterface as RTDEControl
        from rtde_receive import RTDEReceiveInterface as RTDEReceive

        rtde_c = RTDEControl(self.robot_ip, float(self.frequency),
                             RTDEControl.FLAG_UPLOAD_SCRIPT)
        rtde_r = RTDEReceive(self.robot_ip, float(self.frequency))

        if self.set_payload:
            rtde_c.setPayload(self.payload_mass, self.payload_cog)
            if self.tcp_offset is not None:
                rtde_c.setTcp(list(self.tcp_offset))
        else:
            print("[osc] using pendant payload/TCP (set_payload=False).")

        gripper = None
        if self.use_gripper:
            from rc10_api.gripper import Gripper
            gripper = Gripper(device=self.gripper_port, baudrate=self.gripper_baudrate)

        try:
            if self.joints_init is not None:
                print(f"[osc] moveJ to init joints {self.joints_init.tolist()} ...")
                rtde_c.moveJ(self.joints_init.tolist(), self.joints_init_speed, 1.4)

            # Initialise target to the live TCP so the first command is a no-op.
            tcp0 = np.array(rtde_r.getActualTCPPose(), dtype=float)
            with self._tgt.get_lock():
                self._tgt[:6] = tcp0
                self._tgt[6] = 0.0
                self._tgt[7] = 0.0  # not yet valid -> hold live pose

            gripper_state_closed = False
            iter_idx = 0
            dob_z = np.zeros(6)          # DOB filter state (local to the control loop)
            dob_uprev = np.zeros(6)      # last clamped joint torque (for the DOB)
            dhat_norm = 0.0
            safety = 0                   # 0=ok 1=protective 2=estop (polled, throttled)

            while not self._stop.is_set():
                t0 = rtde_c.initPeriod()

                q = np.array(rtde_r.getActualQ(), dtype=float)
                qd = np.array(rtde_r.getActualQd(), dtype=float)

                with self._tgt.get_lock():
                    tgt = np.array(self._tgt[:6])
                    grip_close = self._tgt[6] > 0.5
                    valid = self._tgt[7] > 0.5

                T_ee, J = fk_jacobian(q, self.T_tcp)
                p = T_ee[:3, 3]
                R = Rot.from_matrix(T_ee[:3, :3])

                if valid:
                    p_d = tgt[:3]
                    R_d = Rot.from_rotvec(tgt[3:6])
                else:
                    p_d, R_d = p, R  # hold

                e = pose_error(p_d, R_d, p, R)
                e = np.clip(e, -self.error_delta, self.error_delta)
                ee_vel = J @ qd
                F = self.Kp @ e - self.Kd @ ee_vel       # task-space PD
                tau = J.T @ F                             # gravity comp by directTorque

                # Optional joint-space DOB: estimate + cancel the per-joint disturbance
                # torque (mostly dry friction) to shrink the pure-PD steady-state error.
                # Derivative-free Q-filter; control is torque so Kt=1.
                if self.use_dob:
                    g, Jd, bd = self.dob_g, self.dob_J, self.dob_b
                    v = dob_uprev + (g * Jd - bd) * qd
                    dob_z = dob_z + self.dt * g * (v - dob_z)
                    d_hat = np.clip(dob_z - g * Jd * qd, -self.dob_clip, self.dob_clip)
                    tau = tau + d_hat
                    dhat_norm = float(np.linalg.norm(d_hat))

                tau = np.clip(tau, -self.torque_max, self.torque_max)
                if self.use_dob:
                    dob_uprev = tau          # DOB must see the CLAMPED applied torque

                ok = rtde_c.directTorque(tau.tolist(), False)  # friction_comp=False
                if not ok and self.verbose:
                    print("[osc] directTorque returned False")

                # Safety poll (throttled to ~25 Hz; rtde_r is a separate socket so this
                # doesn't contend with directTorque). Surfaced via get_state()["safety"]
                # so the record/eval loop can abort instead of recording a frozen arm.
                if iter_idx % 20 == 0:
                    try:
                        if rtde_r.isEmergencyStopped():
                            safety = 2
                        elif rtde_r.isProtectiveStopped():
                            safety = 1
                        else:
                            safety = 0
                    except Exception:
                        pass

                # publish state
                tcp = np.array(rtde_r.getActualTCPPose(), dtype=float)
                with self._state.get_lock():
                    self._state[0:6] = q
                    self._state[6:12] = qd
                    self._state[12:18] = tcp
                    self._state[18] = time.time()
                    self._state[19] = dhat_norm
                    self._state[20] = float(safety)

                # gripper on transition only (keep serial I/O off the critical path)
                if gripper is not None:
                    if grip_close and not gripper_state_closed:
                        gripper.send(-1); gripper_state_closed = True
                    elif not grip_close and gripper_state_closed:
                        gripper.send(1); gripper_state_closed = False

                rtde_c.waitPeriod(t0)
                if iter_idx == 0:
                    self._ready.set()
                iter_idx += 1

        finally:
            try:
                rtde_c.directTorque([0.0] * 6, False)
                time.sleep(0.05)
                rtde_c.servoJ(rtde_r.getActualQ(), 0.5, 0.5, 0.1, 0.1, 300)
                rtde_c.servoStop()
            except Exception as e:
                print(f"[osc] cleanup warning: {e}")
            finally:
                try:
                    rtde_c.stopScript(); rtde_c.disconnect(); rtde_r.disconnect()
                except Exception:
                    pass
                if gripper is not None:
                    try:
                        gripper.close()
                    except Exception:
                        pass
                self._ready.set()
                print("[osc] controller process stopped.")
