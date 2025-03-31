import os
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
        
        self.kpts0_path = None
        self.kpts1_path = None
        self.desc0_path = None
        self.desc1_path = None
        
        self.kpts0 = None
        self.kpts1 = None
        self.desc0 = None
        self.desc1 = None
        
        # 保存superpoint的特征点、以及sp+sg+ransac匹配完的特征+描述子+分析
        # self.mkpts0_ransac_path = None
        # self.mkpts1_ransac_path = None
        # self.mconf_path = None
        # self.mdesc0_ransac_path = None
        # self.mdesc1_ransac_path = None  
        
        self.sp_kpts0_file = None
        self.sp_kpts1_file = None

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

    def set_generated_gt_paths(self, input_dir: str):
        """
        Set paths for generated ground truth files.
        
        Args:
            output_dir: Directory where output files will be saved
        """
        # self.mkpts0_ransac_path = day_night_mkpts_mdesc_path / f"{self.stem0}_{self.stem1}_mkpts0-ransac.npy"
        # self.mkpts1_ransac_path = day_night_mkpts_mdesc_path / f"{self.stem0}_{self.stem1}_mkpts1-ransac.npy"
        # self.mconf_path = day_night_mkpts_mdesc_path / f"{self.stem0}_{self.stem1}_mconf.npy"
        # self.mdesc0_ransac_path = day_night_mkpts_mdesc_path / f"{self.stem0}_{self.stem1}_mdesc0-ransac.npy"
        # self.mdesc1_ransac_path = day_night_mkpts_mdesc_path / f"{self.stem0}_{self.stem1}_mdesc1-ransac.npy"
        
        # self.sp_kpts0_path = day_label_path / f"{self.stem0}.npy"
        # self.sp_kpts1_path = night_label_path / f"{self.stem1}.npy"
        
        day_label_folder = os.path.join(input_dir, "day_label")
        night_label_folder = os.path.join(input_dir, "night_label")
        if not os.path.exists(day_label_folder):
            os.makedirs(day_label_folder, exist_ok=True)
        if not os.path.exists(night_label_folder):
            os.makedirs(night_label_folder, exist_ok=True)
            
        self.sp_kpts0_file = os.path.join(day_label_folder, f"{self.name0}.npy")
        self.sp_kpts1_file = os.path.join(night_label_folder, f"{self.name1}.npy")
        
        
    def set_kpts_desc_paths(self, input_dir: str) -> bool:
        """
        Set paths for keypoints and descriptors for day and night images.
        
        Args:
            kpts_desc_lists: List of tuples containing keypoints and descriptors paths
        """
        self.kpts0_path = os.path.join(input_dir, "day_locfeat", f"{self.name0}.kpts.txt")
        self.kpts1_path = os.path.join(input_dir, "night_locfeat", f"{self.name1}.kpts.txt")
        
        self.desc0_path = os.path.join(input_dir, "day_locfeat", f"{self.name0}.desc.txt")
        self.desc1_path = os.path.join(input_dir, "night_locfeat", f"{self.name1}.desc.txt")
        
        if not os.path.exists(self.kpts0_path) or not os.path.exists(self.kpts1_path) or not os.path.exists(self.desc0_path) or not os.path.exists(self.desc1_path):
            print(f"Keypoints or descriptors not found for {self.name0} or {self.name1}")
            return False
        else:
            return True

    def load_kpts_desc(self):
        """
        Load keypoints and descriptors for day and night images.
        
        Returns:
            Tuple containing:
                kpts0: Keypoints for day image
                kpts1: Keypoints for night image
                desc0: Descriptors for day image
                desc1: Descriptors for night image
        """
        # Load keypoints for day image
        try:
            with open(self.kpts0_path, 'r') as f:
                lines = f.readlines()
                num_kpts = int(lines[0].strip())
                kpts0 = []
                for i in range(1, num_kpts + 1):
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        if len(parts) >= 3:
                            x, y, response = float(parts[0]), float(parts[1]), float(parts[2])
                            kpts0.append((x, y, response))
                self.kpts0 = np.array(kpts0)
        except Exception as e:
            print(f"Error loading keypoints from {self.kpts0_path}: {e}")
            self.kpts0 = None
            
        # Load keypoints for night image
        try:
            with open(self.kpts1_path, 'r') as f:
                lines = f.readlines()
                num_kpts = int(lines[0].strip())
                kpts1 = []
                for i in range(1, num_kpts + 1):
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        if len(parts) >= 3:
                            x, y, response = float(parts[0]), float(parts[1]), float(parts[2])
                            kpts1.append((x, y, response))
                self.kpts1 = np.array(kpts1)
        except Exception as e:
            print(f"Error loading keypoints from {self.kpts1_path}: {e}")
            self.kpts1 = None
            
        # Load descriptors for day image
        try:
            with open(self.desc0_path, 'r') as f:
                lines = f.readlines()
                header = lines[0].strip().split()
                if len(header) >= 2:
                    num_desc, dim = int(header[0]), int(header[1])
                    desc0 = []
                    for i in range(1, num_desc + 1):
                        if i < len(lines):
                            values = [float(v) for v in lines[i].strip().split()]
                            if len(values) == dim:
                                desc0.append(values)
                    self.desc0 = np.array(desc0, dtype=np.float32)
        except Exception as e:
            print(f"Error loading descriptors from {self.desc0_path}: {e}")
            self.desc0 = None
            
        # Load descriptors for night image
        try:
            with open(self.desc1_path, 'r') as f:
                lines = f.readlines()
                header = lines[0].strip().split()
                if len(header) >= 2:
                    num_desc, dim = int(header[0]), int(header[1])
                    desc1 = []
                    for i in range(1, num_desc + 1):
                        if i < len(lines):
                            values = [float(v) for v in lines[i].strip().split()]
                            if len(values) == dim:
                                desc1.append(values)
                    self.desc1 = np.array(desc1, dtype=np.float32)
        except Exception as e:
            print(f"Error loading descriptors from {self.desc1_path}: {e}")
            self.desc1 = None
            
        # Transpose descriptors and keypoints, before transpose: (num_keypoints, dim), after transpose: (dim, num_keypoints)
        # To align with SuperPoint shape: SuperPoint descriptors are shape (dim, num_keypoints), 
        # In OpenCV Knn method, it will transpose desc to (num_keypoints, dim), so we need to transpose here first
        self.desc0 = self.desc0.T
        self.desc1 = self.desc1.T
        # self.kpts0 = self.kpts0.T
        # self.kpts1 = self.kpts1.T
        
        # Resize keypoints 
        if self.kpts0 is not None and self.kpts1 is not None:
            # Define original and target dimensions
            src_width, src_height = 3840, 2160  # Original image dimensions
            target_width, target_height = 910, 512  # Target dimensions after resize
            
            # Calculate resize ratios
            ratio_x = float(target_width) / src_width
            ratio_y = float(target_height) / src_height
            
            # Apply resize to keypoints
            if self.kpts0.shape[1] >= 2:  # Make sure keypoints have at least x, y coordinates
                self.kpts0[:, 0] *= ratio_x  # Scale x coordinates
                self.kpts0[:, 1] *= ratio_y  # Scale y coordinates
                
            if self.kpts1.shape[1] >= 2:
                self.kpts1[:, 0] *= ratio_x
                self.kpts1[:, 1] *= ratio_y
                
            print(f"Resized keypoints using ratio: x={ratio_x:.4f}, y={ratio_y:.4f}")
        
        return self.kpts0, self.kpts1, self.desc0, self.desc1
        
    def get_kpts_desc(self):
        """
        Get keypoints and descriptors for the image pair.
        If not already loaded, this method will load them.
        
        Returns:
            Tuple containing:
                kpts0: Keypoints for day image
                kpts1: Keypoints for night image
                desc0: Descriptors for day image
                desc1: Descriptors for night image
        """
        if self.kpts0 is None or self.kpts1 is None or self.desc0 is None or self.desc1 is None:
            return self.load_kpts_desc()
        return self.kpts0, self.kpts1, self.desc0, self.desc1