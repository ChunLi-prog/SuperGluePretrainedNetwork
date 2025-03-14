from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class ImagePair:
    """
    Class representing an image pair with associated data.
    
    This class holds information about a pair of images, including their names,
    rotation information, camera parameters, and the paths for output files.
    """

    def __init__(
        self,
        name0: str,
        name1: str,
        stem0: str,
        stem1: str,
        rot0: int = 0,
        rot1: int = 0,
        intrinsics: Optional[Dict] = None,
        extrinsics: Optional[np.ndarray] = None,
    ):
        """
        Initialize an ImagePair instance.
        
        Args:
            name0: Filename of the first image
            name1: Filename of the second image
            stem0: Stem (filename without extension) of the first image
            stem1: Stem of the second image
            rot0: Rotation angle in degrees for the first image (default: 0)
            rot1: Rotation angle in degrees for the second image (default: 0)
            intrinsics: Dictionary containing K0, K1 camera intrinsic matrices
            extrinsics: T_0to1 transformation matrix
        """
        self.name0 = name0
        self.name1 = name1
        self.stem0 = stem0
        self.stem1 = stem1
        self.rot0 = rot0
        self.rot1 = rot1
        self.image0 = None
        self.image1 = None
        self.inp0 = None
        self.inp1 = None
        self.processed_image0 = None
        self.processed_image1 = None
        self.processed_inp0 = None
        self.processed_inp1 = None
        self.scales0 = None
        self.scales1 = None
        self.intrinsics = intrinsics  # Dictionary containing K0, K1 matrices
        self.extrinsics = extrinsics  # T_0to1 transformation matrix
        self.shape0 = np.zeros(2)
        self.shape1 = np.zeros(2)

        # Paths for outputs
        self.matches_path = None
        self.eval_path = None
        self.viz_path = None
        self.viz_eval_path = None

    def set_paths(self, output_dir: Path, viz_extension: str = "png"):
        """
        Set output file paths for this image pair.
        
        Args:
            output_dir: Directory where output files will be saved
            viz_extension: File extension for visualization images
        """
        self.matches_path = output_dir / f"{self.stem0}_{self.stem1}_matches.npz"
        self.eval_path = output_dir / f"{self.stem0}_{self.stem1}_evaluation.npz"
        self.viz_path = (
            output_dir / f"{self.stem0}_{self.stem1}_matches.{viz_extension}"
        )
        self.viz_eval_path = (
            output_dir / f"{self.stem0}_{self.stem1}_evaluation.{viz_extension}"
        )
