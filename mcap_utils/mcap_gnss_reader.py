"""MCAP GNSSFUSION reader for indoor/outdoor classification.

Reads GnssFusionMsg from GNSSFUSION mcap files and provides
per-timestamp indoor/outdoor labels based on fusion_state.

Outdoor: fusion_state in {2 (NORMAL), 5 (CONVERGING), 6 (GNSS_POS)}
Indoor: all other states
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# fusion_state values that indicate outdoor environment
OUTDOOR_FUSION_STATES = {2, 5, 6}

GNSS_TOPIC = "/ndm/gnss_fusion_msg"


class MCAPGnssReader:
    """Read GNSSFUSION mcap and classify timestamps as indoor/outdoor."""

    def __init__(self, mcap_paths: List[str]):
        """
        Args:
            mcap_paths: List of GNSSFUSION#_*.mcap file paths.
        """
        self.mcap_paths = mcap_paths
        # ts_ms -> fusion_state (int)
        self._gnss_data: Dict[int, int] = {}
        self._indexed = False

    def _ensure_indexed(self):
        if self._indexed:
            return

        try:
            from cmdk.mcap.reader.file_reader import MCAPReader
        except ImportError:
            raise ImportError("cmdk package required")

        for path in self.mcap_paths:
            logger.info("Indexing GNSS MCAP: %s", path)
            try:
                reader = MCAPReader(path)
                for msg_dict in reader.iter_messages(topics=[GNSS_TOPIC]):
                    if GNSS_TOPIC not in msg_dict:
                        continue
                    msg = msg_dict[GNSS_TOPIC]
                    ts = int(msg.proto.timestamp)
                    fusion_state = int(msg.proto.fusion_state)
                    self._gnss_data[ts] = fusion_state
            except Exception as e:
                logger.warning("Failed to index GNSS %s: %s", path, e)

        self._indexed = True
        logger.info("Indexed %d GNSS frames", len(self._gnss_data))

    def get_sorted_timestamps(self) -> List[int]:
        self._ensure_indexed()
        return sorted(self._gnss_data.keys())

    def is_outdoor(self, ts_ms: int, tolerance_ms: int = 500) -> Optional[bool]:
        """Check if the given timestamp is in outdoor environment.

        Args:
            ts_ms: Timestamp in milliseconds.
            tolerance_ms: Max allowed time difference.

        Returns:
            True if outdoor, False if indoor, None if no GNSS data available.
        """
        self._ensure_indexed()

        if not self._gnss_data:
            return None

        # Find closest GNSS timestamp
        best_ts = None
        best_diff = tolerance_ms + 1
        for ts in self._gnss_data:
            diff = abs(ts - ts_ms)
            if diff < best_diff:
                best_diff = diff
                best_ts = ts

        if best_ts is None or best_diff > tolerance_ms:
            return None

        fusion_state = self._gnss_data[best_ts]
        return fusion_state in OUTDOOR_FUSION_STATES

    def get_label(self, ts_ms: int, tolerance_ms: int = 500) -> str:
        """Get indoor/outdoor label string.

        Returns:
            "outdoor", "indoor", or "unknown"
        """
        result = self.is_outdoor(ts_ms, tolerance_ms)
        if result is None:
            return "unknown"
        return "outdoor" if result else "indoor"

    def close(self):
        self._gnss_data.clear()
        self._indexed = False
