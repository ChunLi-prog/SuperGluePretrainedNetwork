import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from data import ImagePair
# Import utility functions
from models.utils import (error_colormap, make_matching_plot, plot_image_pair,
                          plot_keypoints, plot_matches)

logger = logging.getLogger("MatchingPipeline.Visualizer")


class Visualizer:
    """
    Class for visualizing matching results.
    
    This class provides methods for creating visualizations of feature matching
    results, including matches, RANSAC filtering, and evaluation metrics.
    """

    def __init__(
        self,
        output_dir: Path,
        viz_extension: str = "png",
        show_keypoints: bool = False,
        fast_viz: bool = False,
        opencv_display: bool = False,
        line_width: float = 0.3,
    ):
        """
        Initialize a Visualizer.
        
        Args:
            output_dir: Directory where visualization images will be saved
            viz_extension: File extension for visualization images (default: "png")
            show_keypoints: Whether to show keypoints in visualizations (default: False)
            fast_viz: Use faster OpenCV visualization instead of matplotlib (default: False)
            opencv_display: Display visualizations using OpenCV (default: False)
            line_width: Line width for match lines (default: 0.3)
        """
        self.output_dir = output_dir
        self.viz_extension = viz_extension
        self.show_keypoints = show_keypoints
        self.fast_viz = fast_viz
        self.opencv_display = opencv_display
        self.line_width = line_width

        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def visualize_matches(
        self,
        image_pair: ImagePair,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        mconf: np.ndarray,
        text: List[str],
        title: str = "Matches",
        small_text: List[str] = [],
    ):
        """
        Visualize matches between two images.
        
        Args:
            image_pair: ImagePair object containing the images
            kpts0, kpts1: All keypoints from both images
            mkpts0, mkpts1: Matched keypoints from both images
            mconf: Confidence scores for the matches
            text: List of text lines to display on the visualization
            title: Title for the visualization (default: "Matches")
            small_text: Additional small text to display (default: [])
        """
        color = cm.jet(mconf)

        make_matching_plot(
            image_pair.image0,
            image_pair.image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            image_pair.viz_path,
            self.show_keypoints,
            self.fast_viz,
            self.opencv_display,
            title,
            small_text,
        )

    def visualize_ransac_comparison(
        self,
        image_pair: ImagePair,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        mkpts0_ransac: np.ndarray,
        mkpts1_ransac: np.ndarray,
        mconf: np.ndarray,
        mconf_ransac: np.ndarray,
        text: List[str],
    ):
        """
        Visualize comparison between original and RANSAC-filtered matches.
        
        Args:
            image_pair: ImagePair object containing the images
            kpts0, kpts1: All keypoints from both images
            mkpts0, mkpts1: Original matched keypoints
            mkpts0_ransac, mkpts1_ransac: RANSAC-filtered keypoints
            mconf, mconf_ransac: Confidence scores for original and filtered matches
            text: List of text lines to display on the visualization
        """
        comparison_path = (
            self.output_dir
            / f"{image_pair.stem0}_{image_pair.stem1}_ransac_comparison.{self.viz_extension}"
        )
        color = cm.jet(mconf)
        ransac_color = cm.winter(mconf_ransac)

        # Create a side-by-side visualization comparing original and RANSAC-filtered matches
        self._make_comparison_visualization(
            image_pair.image0,
            image_pair.image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            mkpts0_ransac,
            mkpts1_ransac,
            color,
            ransac_color,
            text,
            comparison_path
        )

    def visualize_matcher_comparison(
        self,
        image_pair: ImagePair,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mkpts0_sg: np.ndarray,
        mkpts1_sg: np.ndarray,
        mconf_sg: np.ndarray,
        mkpts0_knn: np.ndarray,
        mkpts1_knn: np.ndarray,
        mconf_knn: np.ndarray,
        text_sg: List[str],
        text_knn: List[str],
    ):
        """Visualize comparison between SuperGlue and KNN matches"""
        comparison_path = (
            self.output_dir
            / f"{image_pair.stem0}_{image_pair.stem1}_sg_vs_knn.{self.viz_extension}"
        )

        self._make_superglue_knn_comparison_plot(
            image_pair.image0,
            image_pair.image1,
            kpts0,
            kpts1,
            mkpts0_sg,
            mkpts1_sg,
            mconf_sg,
            mkpts0_knn,
            mkpts1_knn,
            mconf_knn,
            text_sg,
            text_knn,
            comparison_path,
            self.fast_viz,
            self.line_width,
        )
        
    def visualize_evaluation(
        self,
        image_pair: ImagePair,
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        epi_errs: np.ndarray,
        err_t: float,
        err_R: float,
        matches: np.ndarray,
        num_correct: int,
        text: List[str],
        small_text: List[str] = [],
    ):
        """
        Visualize evaluation results.
        
        Args:
            image_pair: ImagePair object containing the images
            kpts0, kpts1: All keypoints from both images
            mkpts0, mkpts1: Matched keypoints
            epi_errs: Epipolar errors for each match
            err_t, err_R: Translation and rotation errors
            matches: All matches (for statistics)
            num_correct: Number of correct matches
            text: List of text lines to display on the visualization
            small_text: Additional small text to display (default: [])
        """
        # Create color coding based on epipolar error
        color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
        color = error_colormap(1 - color)

        make_matching_plot(
            image_pair.image0,
            image_pair.image1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            image_pair.viz_eval_path,
            self.show_keypoints,
            self.fast_viz,
            self.opencv_display,
            "Relative Pose",
            small_text,
        )

    def create_videos(self, fps: int = 5):
        """
        Create videos from visualization images.
        
        Args:
            fps: Frames per second for the output videos (default: 5)
        """
        if not self.viz_extension:
            logger.info("Skipping video creation: viz_extension not specified")
            return

        # Generate videos from the visualization images
        logger.info("Creating videos from match visualizations...")
        self._create_matches_video("*_matches", fps)
        self._create_matches_video("*_ransac_matches", fps)
        self._create_matches_video("*_ransac_comparison", fps)

    def _create_matches_video(self, name_pattern: str = "*_matches", fps: int = 5):
        """
        Create a video from match visualization images in the output directory.

        Args:
            name_pattern: Pattern to match filenames (default: "*_matches")
            fps: Frames per second for the output video (default: 5)
        """
        match_files = list(self.output_dir.glob(f"{name_pattern}.{self.viz_extension}"))

        if not match_files:
            logger.info(f"No {name_pattern} visualization files found.")
            return

        # Sort files by names to ensure correct order
        match_files.sort()

        # Read the first image to get dimensions
        first_img = cv2.imread(str(match_files[0]))
        if first_img is None:
            logger.error(f"Cannot read image: {match_files[0]}")
            return

        h, w = first_img.shape[:2]

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID'
        video_name = f"{name_pattern.replace('*', 'all')}_video.mp4"
        video_path = self.output_dir / video_name
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

        logger.info(f"Creating video from {len(match_files)} images...")
        for img_path in match_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                out.write(img)

        out.release()
        logger.info(f"Video saved to {video_path}")

    def _make_comparison_visualization(
        self,
        image0,
        image1,
        kpts0,
        kpts1,
        mkpts0,
        mkpts1,
        mkpts0_alt,
        mkpts1_alt,
        color,
        color_alt,
        text,
        out_path
    ):
        """
        Create a comparison visualization between two sets of matches.
        
        This is a simplified version that creates a basic comparison using OpenCV.
        
        Args:
            image0, image1: Input images
            kpts0, kpts1: All keypoints
            mkpts0, mkpts1: First set of matched keypoints
            mkpts0_alt, mkpts1_alt: Second set of matched keypoints
            color, color_alt: Color mappings for the two sets of matches
            text: Text to display
            out_path: Path where the output image will be saved
        """
        # Use OpenCV for faster visualization
        H, W = image0.shape[:2]
        H2, W2 = image1.shape[:2]

        # Create side-by-side images
        out_img1 = np.ones((max(H, H2), W + W2, 3), np.uint8) * 255
        out_img2 = np.ones((max(H, H2), W + W2, 3), np.uint8) * 255

        # Convert grayscale images to RGB if needed
        if image0.ndim == 2:
            image0_rgb = cv2.cvtColor(image0, cv2.COLOR_GRAY2RGB)
        else:
            image0_rgb = image0

        if image1.ndim == 2:
            image1_rgb = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
        else:
            image1_rgb = image1

        # Place the images on both output canvases
        out_img1[:H, :W] = image0_rgb
        out_img1[:H2, W:] = image1_rgb
        out_img2[:H, :W] = image0_rgb
        out_img2[:H2, W:] = image1_rgb

        # Draw first set of matches
        for (x1, y1), (x2, y2), c in zip(mkpts0, mkpts1, color):
            rgb = np.clip(np.array(c[:3]), 0, 1)
            c_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

            cv2.circle(out_img1, (int(x1), int(y1)), 2, c_color, -1)
            cv2.circle(out_img1, (int(x2) + W, int(y2)), 2, c_color, -1)
            cv2.line(
                out_img1,
                (int(x1), int(y1)),
                (int(x2) + W, int(y2)),
                c_color,
                1,
                cv2.LINE_AA,
            )

        # Draw second set of matches
        for (x1, y1), (x2, y2), c in zip(mkpts0_alt, mkpts1_alt, color_alt):
            rgb = np.clip(np.array(c[:3]), 0, 1)
            c_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

            cv2.circle(out_img2, (int(x1), int(y1)), 2, c_color, -1)
            cv2.circle(out_img2, (int(x2) + W, int(y2)), 2, c_color, -1)
            cv2.line(
                out_img2,
                (int(x1), int(y1)),
                (int(x2) + W, int(y2)),
                c_color,
                1,
                cv2.LINE_AA,
            )

        # Add text to images
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out_img1, "Original Matches", (10, 30), font, 1, (0, 0, 0), 2)
        cv2.putText(out_img2, "Filtered Matches", (10, 30), font, 1, (0, 0, 0), 2)

        # Stack images vertically for a before/after comparison
        comparison = np.vstack((out_img1, out_img2))

        # Save the comparison image
        cv2.imwrite(str(out_path), comparison)

    def _make_superglue_knn_comparison_plot(
        self,
        image0,
        image1,
        kpts0,
        kpts1,
        mkpts0_sg,
        mkpts1_sg,
        mconf_sg,
        mkpts0_knn,
        mkpts1_knn,
        mconf_knn,
        text_sg,
        text_knn,
        out_path,
        fast_viz=True,
        line_width=1,  # Will use this for matplotlib line width
    ):
        """Create a visualization comparing SuperGlue and KNN matching results using matplotlib"""

        if fast_viz:
            # Still use OpenCV for fast_viz mode
            # Ensure line_width is an integer for OpenCV
            cv_line_width = max(1, int(round(line_width)))

            # Use OpenCV for faster visualization
            H, W = image0.shape[:2]
            H2, W2 = image1.shape[:2]

            # Create canvases for SuperGlue and KNN
            sg_img = np.ones((max(H, H2), W + W2, 3), np.uint8) * 255
            knn_img = np.ones((max(H, H2), W + W2, 3), np.uint8) * 255

            # Convert grayscale images to RGB if needed
            if image0.ndim == 2:
                image0_rgb = cv2.cvtColor(image0, cv2.COLOR_GRAY2RGB)
            else:
                image0_rgb = image0

            if image1.ndim == 2:
                image1_rgb = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
            else:
                image1_rgb = image1

            # Place images on both canvases
            sg_img[:H, :W] = image0_rgb
            sg_img[:H2, W:] = image1_rgb
            knn_img[:H, :W] = image0_rgb
            knn_img[:H2, W:] = image1_rgb

            # Draw SuperGlue matches
            sg_color = cm.jet(mconf_sg)
            for (x1, y1), (x2, y2), c in zip(mkpts0_sg, mkpts1_sg, sg_color):
                try:
                    rgb = np.clip(np.array(c[:3]), 0, 1)
                    c_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

                    cv2.circle(sg_img, (int(x1), int(y1)), 2, c_color, -1)
                    cv2.circle(sg_img, (int(x2) + W, int(y2)), 2, c_color, -1)
                    cv2.line(
                        sg_img,
                        (int(x1), int(y1)),
                        (int(x2) + W, int(y2)),
                        c_color,
                        cv_line_width,  # Use integer line_width
                        cv2.LINE_AA,
                    )
                except Exception as e:
                    continue

            # Draw KNN matches
            knn_color = cm.cool(mconf_knn)  # Different colormap for KNN
            for (x1, y1), (x2, y2), c in zip(mkpts0_knn, mkpts1_knn, knn_color):
                try:
                    rgb = np.clip(np.array(c[:3]), 0, 1)
                    c_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

                    cv2.circle(knn_img, (int(x1), int(y1)), 2, c_color, -1)
                    cv2.circle(knn_img, (int(x2) + W, int(y2)), 2, c_color, -1)
                    cv2.line(
                        knn_img,
                        (int(x1), int(y1)),
                        (int(x2) + W, int(y2)),
                        c_color,
                        cv_line_width,  # Use integer line_width
                        cv2.LINE_AA,
                    )
                except Exception as e:
                    continue

            # Add titles and text to images
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Add SuperGlue text
            for i, t in enumerate(text_sg):
                cv2.putText(sg_img, t, (10, 30 + i * 30), font, 1, (0, 0, 0), 2)

            # Add KNN text
            for i, t in enumerate(text_knn):
                cv2.putText(knn_img, t, (10, 30 + i * 30), font, 1, (0, 0, 0), 2)

            # Stack images vertically for comparison
            comparison = np.vstack((sg_img, knn_img))

            # Save comparison image
            cv2.imwrite(str(out_path), comparison)
        else:
            # Use matplotlib to create two separate comparison plots

            # Create output directory if it doesn't exist
            out_dir = Path(out_path).parent
            out_dir.mkdir(exist_ok=True, parents=True)

            # Path for SuperGlue matches
            sg_path = out_dir / f"{Path(out_path).stem}_sg.{Path(out_path).suffix}"

            # Path for KNN matches
            knn_path = out_dir / f"{Path(out_path).stem}_knn.{Path(out_path).suffix}"

            # --- SuperGlue matches visualization ---
            plt.figure(figsize=(12, 6))
            # # Use utils.py functions to create the visualization
            # from models.utils import (plot_image_pair, plot_keypoints,
            #                           plot_matches)

            # Plot image pair side by side
            plot_image_pair([image0, image1])

            # Plot matches
            sg_color = cm.jet(mconf_sg)
            plot_matches(mkpts0_sg, mkpts1_sg, sg_color, lw=line_width)

            # Add title and text
            fig = plt.gcf()
            txt_color = "k" if image0[:100, :150].mean() > 200 else "w"
            fig.text(
                0.01,
                0.99,
                "\n".join(text_sg),
                transform=fig.axes[0].transAxes,
                fontsize=15,
                va="top",
                ha="left",
                color=txt_color,
            )

            plt.savefig(str(sg_path), bbox_inches="tight", pad_inches=0)
            plt.close()

            # --- KNN matches visualization ---
            plt.figure(figsize=(12, 6))

            # Plot image pair
            plot_image_pair([image0, image1])

            # Plot KNN matches
            knn_color = cm.cool(mconf_knn)
            plot_matches(mkpts0_knn, mkpts1_knn, knn_color, lw=line_width)

            # Add title and text
            fig = plt.gcf()
            fig.text(
                0.01,
                0.99,
                "\n".join(text_knn),
                transform=fig.axes[0].transAxes,
                fontsize=15,
                va="top",
                ha="left",
                color=txt_color,
            )

            plt.savefig(str(knn_path), bbox_inches="tight", pad_inches=0)
            plt.close()

            # Create a combined image using OpenCV (for consistent output with fast_viz mode)
            img1 = cv2.imread(str(sg_path))
            img2 = cv2.imread(str(knn_path))

            if img1 is not None and img2 is not None:
                # Stack images vertically
                combined = np.vstack((img1, img2))
                cv2.imwrite(str(out_path), combined)

                # Remove individual images if desired
                if True:  # Change to False to keep individual images
                    sg_path.unlink(missing_ok=True)
                    knn_path.unlink(missing_ok=True)