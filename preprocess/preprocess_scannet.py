"""ScanNet preprocessing utility.

This script extracts RGB (and optionally depth) frames, camera poses, and
intrinsics from a ScanNet ``.sens`` file and writes them into a layout that is
directly consumable by ACE/ACE0 training and evaluation scripts.

Layout per scene (example: ``data/scannet/scene0000_00``)::

    images/000000.jpg
    depth/000000.png              # optional
    poses/000000.txt              # 4x4 cam-to-world matrix
    intrins/000000.txt            # 3x3 intrinsics (fx, fy, cx, cy)
    intrinsics_shared.txt         # same intrinsics, stored once

Notes
-----
- ScanNet stores poses as camera-to-world; ACE expects the same for input
  pose files (they are inverted internally). Poses are preserved as C2W.
- Depth is written as 16-bit PNG in millimetres, matching ScanNet's storage
  convention. ACE's RGB-D pipeline divides by 1000 to obtain metres.
- Only the compression modes used in official ScanNet releases are
  implemented (color: JPEG, depth: zlib_ushort).
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import zlib

_logger = logging.getLogger(__name__)


COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


@dataclass
class _FramePayload:
    camera_to_world: np.ndarray
    timestamp_color: int
    timestamp_depth: int
    color_data: bytes
    depth_data: bytes

    def decode_color(self, compression: str) -> np.ndarray:
        if compression != "jpeg":
            raise ValueError(f"Unsupported color compression: {compression}")
        return imageio.imread(io.BytesIO(self.color_data))

    def decode_depth(self, compression: str, height: int, width: int) -> np.ndarray:
        if compression != "zlib_ushort":
            raise ValueError(f"Unsupported depth compression: {compression}")
        decompressed = zlib.decompress(self.depth_data)
        depth = np.frombuffer(decompressed, dtype=np.uint16).reshape(height, width)
        return depth


@dataclass
class _SensHeader:
    version: int
    sensor_name: str
    intrinsic_color: np.ndarray
    extrinsic_color: np.ndarray
    intrinsic_depth: np.ndarray
    extrinsic_depth: np.ndarray
    color_compression: str
    depth_compression: str
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    depth_shift: float
    num_frames: int


def _read_struct(fh, fmt: str):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, fh.read(size))


def _load_header(fh) -> _SensHeader:
    version, = _read_struct(fh, "I")
    name_len, = _read_struct(fh, "Q")
    sensor_name = fh.read(name_len).decode("utf-8")

    intrinsic_color = np.asarray(_read_struct(fh, "f" * 16), dtype=np.float32).reshape(4, 4)
    extrinsic_color = np.asarray(_read_struct(fh, "f" * 16), dtype=np.float32).reshape(4, 4)
    intrinsic_depth = np.asarray(_read_struct(fh, "f" * 16), dtype=np.float32).reshape(4, 4)
    extrinsic_depth = np.asarray(_read_struct(fh, "f" * 16), dtype=np.float32).reshape(4, 4)

    color_compression = COMPRESSION_TYPE_COLOR[_read_struct(fh, "i")[0]]
    depth_compression = COMPRESSION_TYPE_DEPTH[_read_struct(fh, "i")[0]]

    color_width, = _read_struct(fh, "I")
    color_height, = _read_struct(fh, "I")
    depth_width, = _read_struct(fh, "I")
    depth_height, = _read_struct(fh, "I")

    depth_shift, = _read_struct(fh, "f")
    num_frames, = _read_struct(fh, "Q")

    return _SensHeader(
        version=version,
        sensor_name=sensor_name,
        intrinsic_color=intrinsic_color,
        extrinsic_color=extrinsic_color,
        intrinsic_depth=intrinsic_depth,
        extrinsic_depth=extrinsic_depth,
        color_compression=color_compression,
        depth_compression=depth_compression,
        color_width=color_width,
        color_height=color_height,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_shift=depth_shift,
        num_frames=num_frames,
    )


def _iter_frames(fh, header: _SensHeader) -> Generator[_FramePayload, None, None]:
    for _ in range(header.num_frames):
        camera_to_world = np.asarray(_read_struct(fh, "f" * 16), dtype=np.float32).reshape(4, 4)
        timestamp_color, = _read_struct(fh, "Q")
        timestamp_depth, = _read_struct(fh, "Q")
        color_size, = _read_struct(fh, "Q")
        depth_size, = _read_struct(fh, "Q")
        color_data = fh.read(color_size)
        depth_data = fh.read(depth_size)

        yield _FramePayload(
            camera_to_world=camera_to_world,
            timestamp_color=timestamp_color,
            timestamp_depth=timestamp_depth,
            color_data=color_data,
            depth_data=depth_data,
        )


def _save_matrix(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%.9f")


def preprocess_scene(
    scene_id: str,
    scans_root: Path,
    output_root: Path,
    frame_skip: int = 1,
    max_frames: Optional[int] = None,
    start_index: int = 0,
    export_depth: bool = False,
    resize_long_edge: Optional[int] = None,
    log_every: int = 200,
) -> None:
    """Convert a ScanNet scene into ACE-friendly folders.

    Parameters
    ----------
    scene_id: str
        Name of the ScanNet scene (e.g. ``scene0000_00``).
    scans_root: Path
        Directory that contains ScanNet ``scans`` (defaults to
        ``/hdd1/datasets/scannet/scans``).
    output_root: Path
        Root output directory, a scene subfolder will be created inside.
    frame_skip: int
        Keep every N-th frame. ``1`` keeps all frames.
    max_frames: Optional[int]
        Process at most this many frames after skipping. ``None`` keeps all.
    start_index: int
        First frame index in the .sens file to consider.
    export_depth: bool
        If True, also write 16-bit depth PNGs (in millimetres).
    resize_long_edge: Optional[int]
        If set, resize RGB and depth so the long edge equals this value.
    """

    sens_path = scans_root / scene_id / f"{scene_id}.sens"
    if not sens_path.exists():
        raise FileNotFoundError(f"Could not find {sens_path}")

    scene_out = output_root / scene_id
    images_out = scene_out / "images"
    depth_out = scene_out / "depth"
    poses_out = scene_out / "poses"
    intrins_out = scene_out / "intrins"
    scene_out.mkdir(parents=True, exist_ok=True)

    _logger.info("Reading header from %s", sens_path)
    with sens_path.open("rb") as fh:
        header = _load_header(fh)

        _logger.info(
            "Sensor %s | %dx%d RGB | %dx%d depth | %d frames",
            header.sensor_name,
            header.color_width,
            header.color_height,
            header.depth_width,
            header.depth_height,
            header.num_frames,
        )

        if header.color_compression != "jpeg" or header.depth_compression != "zlib_ushort":
            _logger.warning(
                "Unexpected compression (color=%s, depth=%s); the decoder only supports the default JPEG/zlib_ushort."
                " You might need to extend the script.",
                header.color_compression,
                header.depth_compression,
            )

        color_scale = _compute_scale(header.color_height, header.color_width, resize_long_edge)
        depth_scale = _compute_scale(header.depth_height, header.depth_width, resize_long_edge)

        # Save a shared intrinsics file once. Per-frame files are added below so they can
        # be consumed via `--calibration_files` if desired.
        shared_intrinsics_path = scene_out / "intrinsics_shared.txt"
        _save_matrix(_scale_intrinsics(header.intrinsic_color[:3, :3], color_scale), shared_intrinsics_path)

        candidate_frames = max(header.num_frames - start_index, 0)
        est_total = int(math.ceil(candidate_frames / frame_skip)) if frame_skip > 0 else 0
        target_total = est_total if max_frames is None else min(est_total, max_frames)
        _logger.info("Planning to write up to %d frames (estimated)", target_total)

        keep_count = 0
        for frame_idx, frame in enumerate(_iter_frames(fh, header)):
            if frame_idx < start_index:
                continue
            if (frame_idx - start_index) % frame_skip != 0:
                continue
            if max_frames is not None and keep_count >= max_frames:
                break

            rgb = frame.decode_color(header.color_compression)
            depth = None
            if export_depth:
                depth = frame.decode_depth(
                    header.depth_compression, header.depth_height, header.depth_width
                )

            if color_scale != 1.0:
                rgb = _resize_rgb(rgb, color_scale)
            if depth is not None and depth_scale != 1.0:
                depth = _resize_depth(depth, depth_scale)

            stem = f"{frame_idx:06d}"

            images_out.mkdir(parents=True, exist_ok=True)
            image_path = images_out / f"{stem}.jpg"
            imageio.imwrite(image_path, rgb)

            poses_out.mkdir(parents=True, exist_ok=True)
            _save_matrix(frame.camera_to_world, poses_out / f"{stem}.txt")

            intrins_out.mkdir(parents=True, exist_ok=True)
            _save_matrix(
                _scale_intrinsics(header.intrinsic_color[:3, :3], color_scale),
                intrins_out / f"{stem}.txt",
            )

            if export_depth:
                depth_out.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(depth_out / f"{stem}.png", depth.astype(np.uint16))

            keep_count += 1

            if log_every > 0 and (keep_count % log_every == 0 or keep_count == target_total):
                pct = (keep_count / target_total * 100) if target_total > 0 else 0
                _logger.info("Processed %d / %d frames (%.1f%%)", keep_count, target_total, pct)

        _logger.info("Wrote %d frames to %s", keep_count, scene_out)


def _compute_scale(height: int, width: int, long_edge: Optional[int]) -> float:
    if long_edge is None:
        return 1.0
    return float(long_edge) / float(max(height, width))


def _resize_rgb(rgb: np.ndarray, scale: float) -> np.ndarray:
    """Resize an RGB image while preserving aspect ratio."""
    h, w = rgb.shape[:2]
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _resize_depth(depth: np.ndarray, scale: float) -> np.ndarray:
    """Resize a depth map using nearest-neighbour sampling."""
    h, w = depth.shape[:2]
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def _scale_intrinsics(K: np.ndarray, scale: float) -> np.ndarray:
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale
    K_scaled[1, 1] *= scale
    K_scaled[0, 2] *= scale
    K_scaled[1, 2] *= scale
    return K_scaled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a ScanNet scene into ACE-friendly folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene_id", required=True, help="Scene id, e.g. scene0000_00")
    parser.add_argument(
        "--scans_root",
        type=Path,
        default=Path("/hdd1/datasets/scannet/scans"),
        help="Root path that contains ScanNet scans",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("data/scannet"),
        help="Root directory to place preprocessed scenes",
    )
    parser.add_argument("--frame_skip", type=int, default=1, help="Keep every N-th frame")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of frames after skipping")
    parser.add_argument("--start_index", type=int, default=0, help="First frame index to consider")
    parser.add_argument("--export_depth", action="store_true", help="Also export 16-bit depth PNGs")
    parser.add_argument(
        "--resize_long_edge",
        type=int,
        default=None,
        help="Resize RGB/depth so the long edge equals this value. None keeps native size",
    )
    parser.add_argument("--log_every", type=int, default=200, help="Log progress every N written frames")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    preprocess_scene(
        scene_id=args.scene_id,
        scans_root=args.scans_root,
        output_root=args.output_root,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        start_index=args.start_index,
        export_depth=args.export_depth,
        resize_long_edge=args.resize_long_edge,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
