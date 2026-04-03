"""MCAP ADAS image reader with timestamp-based random access.

Reads H.265-encoded camera images from MCAP files and decodes them
into numpy arrays using H265Decoder.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .h265_decoder import H265Decoder

logger = logging.getLogger(__name__)


class MCAPImageReader:
    """Read and decode ADAS camera images from MCAP files.

    Builds a timestamp-indexed cache on first scan, then supports
    random access by timestamp with tolerance.
    """

    def __init__(
        self,
        mcap_paths: List[str],
        camera_channel: str,
        target_size: Tuple[int, int] = (960, 720),
    ):
        """
        Args:
            mcap_paths: List of ADAS MCAP file paths for this session.
            camera_channel: Camera channel id, e.g. "7" (front) or "6" (rear).
            target_size: (width, height) to resize decoded images.
        """
        self.mcap_paths = mcap_paths
        self.camera_channel = camera_channel
        self.topic = f"image-/dc/adas_{camera_channel}"
        self.target_size = target_size
        self.decoder = H265Decoder()

        # timestamp (ms) -> (mcap_path_index, raw_h265_data)
        self._ts_index: Dict[int, Tuple[int, object]] = {}
        # Sorted list of timestamps that contain IDR frames
        self._idr_timestamps: List[int] = []
        self._indexed = False
        self._readers = {}

    def _ensure_indexed(self):
        """Build timestamp index by scanning all MCAP files."""
        if self._indexed:
            return

        try:
            from cmdk.mcap.reader.file_reader import MCAPReader
        except ImportError:
            raise ImportError(
                "cmdk package required. Install with: pip install cmdk"
            )

        for path_idx, path in enumerate(self.mcap_paths):
            logger.info("Indexing MCAP: %s (topic=%s)", path, self.topic)
            try:
                reader = MCAPReader(path)
                for msg_dict in reader.iter_messages(topics=[self.topic]):
                    if self.topic not in msg_dict:
                        continue
                    msg = msg_dict[self.topic]
                    ts = msg.proto.time_stamp
                    # Store raw data for later decode
                    raw_data = getattr(msg, "data", None)
                    if raw_data and len(raw_data) > 0:
                        self._ts_index[int(ts)] = (path_idx, raw_data[0])
            except Exception as e:
                logger.warning("Failed to index %s: %s", path, e)

        self._indexed = True
        self._idr_timestamps = sorted(
            ts for ts, (_, raw) in self._ts_index.items()
            if self.decoder._contains_idr_frame(raw)
        )
        logger.info(
            "Indexed %d frames (%d IDR) from %d MCAP files for adas_%s",
            len(self._ts_index), len(self._idr_timestamps),
            len(self.mcap_paths), self.camera_channel,
        )

    def get_timestamps(self) -> List[int]:
        """Get all available timestamps (ms) in sorted order."""
        self._ensure_indexed()
        return sorted(self._ts_index.keys())

    def get_frame_by_ts(
        self,
        target_ts_ms: int,
        tolerance_ms: int = 500,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Get decoded + resized image closest to target timestamp.

        Args:
            target_ts_ms: Target timestamp in milliseconds.
            tolerance_ms: Maximum allowed difference.

        Returns:
            (image_bgr, actual_ts_ms) or None if no frame within tolerance.
        """
        self._ensure_indexed()

        # Find closest timestamp
        best_ts = None
        best_diff = tolerance_ms + 1
        for ts in self._ts_index:
            diff = abs(ts - target_ts_ms)
            if diff < best_diff:
                best_diff = diff
                best_ts = ts

        if best_ts is None or best_diff > tolerance_ms:
            return None

        _, raw_data = self._ts_index[best_ts]

        # If target frame is not IDR, prime decoder with nearest prior IDR
        if not self.decoder._contains_idr_frame(raw_data):
            idr_ts = self._find_prior_idr(best_ts)
            if idr_ts is not None and idr_ts != getattr(self, '_last_idr_ts', None):
                _, idr_data = self._ts_index[idr_ts]
                self.decoder.decode(idr_data)
                self._last_idr_ts = idr_ts
            elif idr_ts is None:
                logger.debug("No IDR frame before ts=%d, skipping", best_ts)
                return None

        frame = self.decoder.decode(raw_data)
        if frame is None:
            return None

        if self.target_size:
            frame = cv2.resize(frame, self.target_size)

        return frame, best_ts

    def get_grayscale_by_ts(
        self,
        target_ts_ms: int,
        tolerance_ms: int = 500,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Get decoded grayscale image for SuperPoint input.

        Returns:
            (grayscale_image, actual_ts_ms) or None.
        """
        result = self.get_frame_by_ts(target_ts_ms, tolerance_ms)
        if result is None:
            return None
        frame_bgr, ts = result
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return gray, ts

    def _find_prior_idr(self, target_ts: int) -> Optional[int]:
        """Find the IDR frame timestamp closest to but <= target_ts."""
        if not self._idr_timestamps:
            return None
        # Binary search for the rightmost IDR <= target_ts
        import bisect
        idx = bisect.bisect_right(self._idr_timestamps, target_ts) - 1
        if idx < 0:
            return None
        return self._idr_timestamps[idx]

    def close(self):
        """Release resources."""
        self._ts_index.clear()
        self._idr_timestamps.clear()
        self._indexed = False
        self._last_idr_ts = None
        self.decoder.reset()
