"""
Standalone gripper stress/diagnostic — raw pyserial, no Gripper wrapper, no UR stack.

Isolates the gripper USB/power path from the teleop backend. If the device drops here
(EIO / port vanishes), it is a hardware / power / cable / EMI problem, not the software.

It opens the port ONCE and cycles Open/Close, timestamping every command and reporting:
  - latency of each write+flush,
  - elapsed time since start and since the last good command when a drop happens,
  - the real device node behind the symlink (catches ttyACM0<->ttyACM1 renumbering),
  - a running count of drops, and a summary on Ctrl+C.

Run (lerobot conda env):
    python ur10_standalone_act/gripper_selftest.py --port /dev/rc10_gripper
    # faster cycling to provoke actuation-current brownouts:
    python ur10_standalone_act/gripper_selftest.py --port /dev/rc10_gripper --interval 0.3
    # hold the port open but stop actuating, to see if it drops while idle:
    python ur10_standalone_act/gripper_selftest.py --port /dev/rc10_gripper --idle

Watch the kernel in a second terminal at the same time:
    journalctl -kf | grep -i -E 'usb|tty|acm'      # or:  sudo dmesg -w
"""

import argparse
import os
import time

import serial


def realnode(port):
    try:
        return os.path.realpath(port)
    except OSError:
        return "<gone>"


def open_port(port, baud, timeout, settle):
    ser = serial.Serial(port, baud, timeout=timeout)
    if settle:
        time.sleep(settle)  # ride out the DTR-reset-on-open the MCU may do
    return ser


def main():
    p = argparse.ArgumentParser(description="Raw-serial gripper drop/stress test")
    p.add_argument("--port", default="/dev/rc10_gripper")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--timeout", type=float, default=1.0)
    p.add_argument("--interval", type=float, default=0.75, help="seconds between commands")
    p.add_argument("--settle", type=float, default=2.0, help="post-open settle (reset-on-open)")
    p.add_argument("--cycles", type=int, default=0, help="open+close cycles; 0 = run forever")
    p.add_argument("--idle", action="store_true",
                   help="hold the port open but DO NOT actuate (tests idle-drop vs load-drop)")
    args = p.parse_args()

    print(f"[selftest] port={args.port} -> {realnode(args.port)}  baud={args.baudrate}")
    ser = open_port(args.port, args.baudrate, args.timeout, args.settle)
    print(f"[selftest] opened. settle={args.settle}s. Ctrl+C to stop.\n")

    t_start = time.monotonic()
    last_good = t_start
    n_cmd = 0
    n_drop = 0
    reopens = 0

    try:
        while True:
            if args.idle:
                # Just keep the port open; poll it so a bus drop still surfaces as an error.
                try:
                    ser.in_waiting  # touches the fd; raises if the device vanished
                except (serial.SerialException, OSError) as err:
                    n_drop += 1
                    print(f"\n[DROP #{n_drop}] idle, t+{time.monotonic()-t_start:6.1f}s  "
                          f"since last good {time.monotonic()-last_good:5.1f}s  err={err}")
                    print(f"           node now -> {realnode(args.port)}")
                    ser = _reopen(ser, args); reopens += 1; last_good = time.monotonic()
                else:
                    last_good = time.monotonic()
                time.sleep(args.interval)
                continue

            payload = b"Close\n" if (n_cmd % 2 == 0) else b"Open\n"
            t0 = time.monotonic()
            try:
                ser.write(payload)
                ser.flush()  # force EIO to surface here rather than buffering silently
            except (serial.SerialException, OSError) as err:
                n_drop += 1
                print(f"\n[DROP #{n_drop}] cmd#{n_cmd} {payload!r}  t+{time.monotonic()-t_start:6.1f}s  "
                      f"since last good {time.monotonic()-last_good:5.1f}s  err={err}")
                print(f"           node now -> {realnode(args.port)}")
                ser = _reopen(ser, args); reopens += 1
                last_good = time.monotonic()
            else:
                dt_ms = (time.monotonic() - t0) * 1e3
                last_good = time.monotonic()
                n_cmd += 1
                print(f"  cmd#{n_cmd:4d}  {payload.decode().strip():5s}  "
                      f"{dt_ms:6.1f} ms   t+{time.monotonic()-t_start:6.1f}s   "
                      f"drops={n_drop}", end="\r")
                if args.cycles and n_cmd >= args.cycles:
                    break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            ser.close()
        except Exception:
            pass
        dur = time.monotonic() - t_start
        print(f"\n\n[summary] ran {dur:.1f}s  good_cmds={n_cmd}  drops={n_drop}  reopens={reopens}")
        if n_drop == 0:
            print("[summary] no drops — gripper USB/power path looks clean.")
        else:
            rate = n_drop / max(dur, 1e-9) * 60.0
            print(f"[summary] {rate:.1f} drops/min. A drop under load (not idle) points at "
                  f"actuation-current brownout or EMI; a drop while --idle points at cable/"
                  f"connector/host-port or a bus-powered supply sag.")


def _reopen(old, args):
    try:
        old.close()
    except Exception:
        pass
    for attempt in range(1, 6):
        try:
            return open_port(args.port, args.baudrate, args.timeout, args.settle)
        except (serial.SerialException, OSError) as err:
            print(f"           reopen attempt {attempt}/5 failed: {err} (node {realnode(args.port)})")
            time.sleep(args.settle)
    raise SystemExit("[selftest] device did not come back after 5 reopen attempts — check power/cable")


if __name__ == "__main__":
    main()
