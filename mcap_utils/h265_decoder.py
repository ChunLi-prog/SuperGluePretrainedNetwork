"""H.265 video frame decoder with IDR frame caching.

Decodes individual H.265 NAL units from MCAP ADAS camera messages.
P-frames require a cached IDR frame for reference.
"""

import logging
import os
import tempfile

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class H265Decoder:
    """Stateful H.265 decoder that caches IDR frames for P-frame decoding."""

    def __init__(self):
        self.idr_frame_data = None
        self._temp_counter = 0

    def decode(self, h265_data: bytes) -> np.ndarray:
        """Decode H.265 bytes to a BGR numpy array.

        Args:
            h265_data: Raw H.265 NAL unit bytes from MCAP message.

        Returns:
            BGR image as numpy array, or None on failure.
        """
        if not h265_data:
            return None

        has_idr = self._contains_idr_frame(h265_data)

        if has_idr:
            self.idr_frame_data = h265_data
            return self._decode_data(h265_data)

        if self.idr_frame_data is None:
            logger.debug("No IDR frame cached yet, skipping P-frame")
            return None

        combined = self.idr_frame_data + h265_data
        return self._decode_data(combined)

    def reset(self):
        """Clear cached IDR frame (call when switching MCAP files)."""
        self.idr_frame_data = None

    @staticmethod
    def _contains_idr_frame(data: bytes) -> bool:
        """Check if data contains an H.265 IDR frame (NAL type 19 or 20)."""
        i = 0
        while i < len(data) - 5:
            if data[i:i + 4] == b'\x00\x00\x00\x01':
                nal_type = (data[i + 4] >> 1) & 0x3F
                if nal_type in (19, 20):
                    return True
                i += 4
            elif data[i:i + 3] == b'\x00\x00\x01':
                nal_type = (data[i + 3] >> 1) & 0x3F
                if nal_type in (19, 20):
                    return True
                i += 3
            else:
                i += 1
        return False

    def _decode_data(self, data: bytes) -> np.ndarray:
        """Write data to temp file and decode with OpenCV VideoCapture."""
        self._temp_counter += 1
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=f"_{self._temp_counter}.h265", delete=False
            ) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            last_frame = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                last_frame = frame
            cap.release()
            return last_frame

        except Exception as e:
            logger.warning("H.265 decode failed: %s", e)
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
