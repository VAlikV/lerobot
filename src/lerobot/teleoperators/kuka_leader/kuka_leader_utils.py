"""
Low-level serial reading helper for the STM32 encoder board.

The STM32 continuously streams ASCII frames like:

    "[1023, 2048, 512, 4095, 10, 3000, 999, 0]\n"
"""

import logging
import threading
import time
import ast

import serial

logger = logging.getLogger(__name__)

FRAME_TERMINATOR = b"\n"
MOVE_DONE = b"MOVE_DONE"


class SerialFrameReader(threading.Thread):
    """Background thread that keeps only the latest decoded CSV frame from a serial port."""

    def __init__(self, ser: serial.Serial, num_channels: int):
        super().__init__(daemon=True)
        self._ser = ser
        self._num_channels = num_channels
        self._lock = threading.Lock()
        self._frame_received = threading.Condition(self._lock)
        self._write_lock = threading.Lock()
        self._latest: list[int] | None = None
        self._latest_ts: float = 0.0
        self._latest_monotonic_ts: float = 0.0
        self._move_completed_ts: float = 0.0
        self._stop_event = threading.Event()
        self._buffer = b""

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._ser.read(self._ser.in_waiting or 1)
            except serial.SerialException as e:
                logger.error(f"Serial read error, stopping reader thread: {e}")
                return
            if not chunk:
                continue

            self._buffer += chunk
            if FRAME_TERMINATOR not in self._buffer:
                continue

            *complete, self._buffer = self._buffer.split(FRAME_TERMINATOR)
            # Process every line: MOVE_DONE and telemetry can share one read.
            for line in complete:
                line = line.strip()
                if line == MOVE_DONE:
                    with self._frame_received:
                        self._move_completed_ts = time.monotonic()
                        self._frame_received.notify_all()
                    continue

                values = self._parse_frame(line)
                if values is None:
                    continue
                with self._frame_received:
                    self._latest = values
                    self._latest_ts = time.time()
                    self._latest_monotonic_ts = time.monotonic()
                    self._frame_received.notify_all()

    def _parse_frame(self, line: bytes) -> list[int] | None:
        if not line:
            return None

        try:
            values = ast.literal_eval(line.decode("ascii"))
        except (UnicodeDecodeError, ValueError, SyntaxError):
            logger.debug(f"Dropping malformed frame from STM32: {line!r}")
            return None

        if not isinstance(values, list):
            logger.debug(f"Dropping malformed frame from STM32: {line!r}")
            return None

        if not all(isinstance(v, int) for v in values):
            logger.debug(f"Dropping frame with non-integer values: {line!r}")
            return None

        if len(values) != self._num_channels:
            logger.debug(f"Dropping frame with wrong channel count: {line!r}")
            return None

        return values

    def latest(self, max_age_s: float | None = None) -> list[int]:
        with self._lock:
            values, ts = self._latest, self._latest_ts
        if values is None:
            raise ConnectionError("No valid frame received from the STM32 board yet.")
        age = time.time() - ts
        if max_age_s is not None and age > max_age_s:
            raise ConnectionError(
                f"Latest STM32 frame is {age:.2f}s old (> {max_age_s}s). "
            )
        return values

    def stop(self) -> None:
        self._stop_event.set()
        with self._frame_received:
            self._frame_received.notify_all()

    def wait_for_frame(self, timeout_s: float = 2.0, *, resume_after_ts: float | None = None) -> list[int]:
        """
        Wait for a valid frame, optionally following explicit move completion.
        """
        deadline = time.monotonic() + timeout_s
        with self._frame_received:
            while self._latest is None or (
                resume_after_ts is not None and (
                    self._move_completed_ts <= resume_after_ts
                    or self._latest_monotonic_ts <= self._move_completed_ts
                )
            ):
                if self._stop_event.is_set():
                    raise ConnectionError("STM32 reader stopped while waiting for a frame.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if resume_after_ts is not None:
                        raise ConnectionError(
                            f"No MOVE_DONE followed by a position frame within {timeout_s}s; "
                            "leader synchronization is not confirmed."
                        )
                    raise ConnectionError(f"No valid STM32 frame received within {timeout_s}s.")
                self._frame_received.wait(timeout=remaining)
            return self._latest.copy()

    def send(self, values: list[int]) -> None:
        """
        Send command frame to STM32 board, e.g. b"[1023, 2048, ..., 0]\n".
        """
        if len(values) != self._num_channels:
            raise ValueError(
                f"Expected {self._num_channels} values, got {len(values)}: {values}"
            )
        frame = "[" + " ".join(str(int(v)) for v in values) + "]\n"
        with self._write_lock:
            self._ser.write(frame.encode("ascii"))
            self._ser.flush()
