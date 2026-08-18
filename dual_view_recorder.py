"""Dual-view video recorder for MuJoCo grasp execution."""

from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


class DualViewRecorder:
    def __init__(
        self,
        env,
        output_dir="grasp_videos",
        camera_left="frontview",
        camera_right="sideview",
        width=640,
        height=480,
        capture_interval=2,
        fps=10,
    ):
        self.env = env
        self.output_dir = Path(output_dir)

        self.camera_left = camera_left
        self.camera_right = camera_right

        self.width = int(width)
        self.height = int(height)
        self.capture_interval = int(capture_interval)
        self.fps = int(fps)

        self.frames = []
        self.step_counter = 0

        self.original_step = None
        self.started = False

    def render_view(self, camera_name):
        color, _, _ = self.env.render_camera({
            "camera_name": camera_name,
            "width": self.width,
            "height": self.height,
        })

        color = np.flipud(color)

        return np.ascontiguousarray(
            color,
            dtype=np.uint8,
        )

    def capture_frame(self):
        left = self.render_view(self.camera_left)
        right = self.render_view(self.camera_right)

        combined = np.concatenate(
            [left, right],
            axis=1,
        )

        self.frames.append(combined)
        return combined

    def add_hold(self, seconds):
        if not self.frames:
            self.capture_frame()

        frame_count = max(
            1,
            int(round(float(seconds) * self.fps)),
        )

        last_frame = self.frames[-1]

        self.frames.extend(
            last_frame.copy()
            for _ in range(frame_count)
        )

    def start(self):
        if self.started:
            return

        self.original_step = self.env.step

        def recorded_step(action):
            result = self.original_step(action)

            self.step_counter += 1

            if (
                self.step_counter
                % self.capture_interval
                == 0
            ):
                self.capture_frame()

            return result

        self.env.step = recorded_step
        self.started = True

        self.capture_frame()

    def stop(self):
        if (
            self.started
            and self.original_step is not None
        ):
            self.env.step = self.original_step

        self.started = False

    def save(self, output_path=None):
        if not self.frames:
            raise RuntimeError(
                "No video frames were recorded."
            )

        if output_path is None:
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            output_path = (
                self.output_dir
                / f"closed_loop_{timestamp}.mp4"
            )
        else:
            output_path = Path(output_path)

        output_path = output_path.expanduser().resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        imageio.mimsave(
            output_path,
            self.frames,
            fps=self.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )

        return output_path
