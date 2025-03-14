#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import yaml

# Create logger
logger = logging.getLogger("MatchingPipeline.Main")

from pipeline import MatchingPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Image pair matching and pose evaluation with SuperGlue",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file is now the primary way to configure the application
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (required)"
    )

    # # All other arguments are now optional overrides for the YAML config
    # parser.add_argument(
    #     "--input_pairs",
    #     type=str,
    #     help="Path to the list of image pairs (overrides config file)",
    # )
    # parser.add_argument(
    #     "--input_dir",
    #     type=str,
    #     help="Path to the directory that contains the images (overrides config file)",
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     help="Path to the directory for outputs (overrides config file)",
    # )

    # parser.add_argument(
    #     "--max_length", type=int, help="Maximum number of pairs to evaluate"
    # )
    # parser.add_argument(
    #     "--resize",
    #     type=int,
    #     nargs="+",
    #     help="Resize the input image before running inference. If two numbers, "
    #     "resize to the exact dimensions, if one number, resize the max "
    #     "dimension, if -1, do not resize",
    # )
    # parser.add_argument(
    #     "--resize_float",
    #     action="store_true",
    #     help="Resize the image after casting uint8 to float",
    # )

    # parser.add_argument(
    #     "--superglue",
    #     choices={"indoor", "outdoor"},
    #     help="SuperGlue weights",
    # )
    # parser.add_argument(
    #     "--max_keypoints",
    #     type=int,
    #     help="Maximum number of keypoints detected by Superpoint",
    # )
    # parser.add_argument(
    #     "--keypoint_threshold",
    #     type=float,
    #     help="SuperPoint keypoint detector confidence threshold",
    # )
    # parser.add_argument(
    #     "--nms_radius",
    #     type=int,
    #     help="SuperPoint Non Maximum Suppression (NMS) radius",
    # )
    # parser.add_argument(
    #     "--sinkhorn_iterations",
    #     type=int,
    #     help="Number of Sinkhorn iterations performed by SuperGlue",
    # )
    # parser.add_argument(
    #     "--match_threshold", type=float, help="SuperGlue match threshold"
    # )

    # parser.add_argument(
    #     "--viz", action="store_true", help="Visualize the matches and dump the plots"
    # )
    # parser.add_argument(
    #     "--eval",
    #     action="store_true",
    #     help="Perform the evaluation (requires ground truth pose and intrinsics)",
    # )
    # parser.add_argument(
    #     "--fast_viz",
    #     action="store_true",
    #     help="Use faster image visualization with OpenCV instead of Matplotlib",
    # )
    # parser.add_argument(
    #     "--cache",
    #     action="store_true",
    #     help="Skip the pair if output .npz files are already found",
    # )
    # parser.add_argument(
    #     "--show_keypoints",
    #     action="store_true",
    #     help="Plot the keypoints in addition to the matches",
    # )
    # parser.add_argument(
    #     "--viz_extension",
    #     type=str,
    #     choices=["png", "pdf"],
    #     help="Visualization file extension. Use pdf for highest-quality.",
    # )
    # parser.add_argument(
    #     "--opencv_display",
    #     action="store_true",
    #     help="Visualize via OpenCV before saving output images",
    # )
    # parser.add_argument(
    #     "--shuffle",
    #     action="store_true",
    #     help="Shuffle ordering of pairs before processing",
    # )
    # parser.add_argument(
    #     "--force_cpu", action="store_true", help="Force pytorch to run in CPU mode."
    # )

    # # RANSAC parameters
    # parser.add_argument(
    #     "--ransac_enabled",
    #     action="store_true",
    #     help="Apply RANSAC to filter matches",
    # )
    # # parser.add_argument(
    # #     "--ransac_threshold",
    # #     type=float,
    # #     help="RANSAC threshold for match filtering",
    # # )
    # # parser.add_argument(
    # #     "--ransac_method",
    # #     type=str,
    # #     choices=["fundamental", "homography"],
    # #     help="RANSAC method to use for filtering",
    # # )
    # parser.add_argument(
    #     "--viz_comparison",
    #     action="store_true",
    #     help="Visualize comparison between before and after RANSAC filtering",
    # )

    # # KNN comparison parameters
    # parser.add_argument(
    #     "--compare_sg_knn_ransac",
    #     action="store_true",
    #     help="Run KNN matching in addition to SuperGlue for comparison",
    # )
    # parser.add_argument(
    #     "--knn_ratio",
    #     type=float,
    #     help="Ratio threshold for Lowe's ratio test in KNN matching",
    # )
    # parser.add_argument(
    #     "--knn_distance",
    #     type=str,
    #     choices=["L2", "L1", "Hamming"],
    #     help="Distance metric for KNN matching",
    # )

    # # Visualization parameters
    # parser.add_argument(
    #     "--line_width",
    #     type=float,
    #     help="Line width for visualizing feature matches",
    # )
    
    # # FeatureBooster parameters
    # parser.add_argument(
    #     "--use_fb",
    #     action="store_true",
    #     help="Use FeatureBooster plugin to boost SuperPoint features",
    # )
    # parser.add_argument(
    #     "--fb_config",
    #     type=str,
    #     help="Path to a config file for FeatureBooster",
    # )
    # parser.add_argument(
    #     "--compare_knn_fb_ransac",
    #     action="store_true",
    #     help="Run KNN matching in addition to FeatureBooster for comparison",
    # )
    # parser.add_argument(
    #     "--compare_fb_sg_ransac",
    #     action="store_true",
    #     help="Visualize comparison between FeatureBooster and SuperGlue matches",
    # )

    return parser.parse_args()


def validate_args(config):
    """Validate configuration settings."""
    assert not (
        config.get("opencv_display") and not config.get("viz")
    ), "Must use viz=true with opencv_display=true"
    assert not (
        config.get("opencv_display") and not config.get("fast_viz")
    ), "Cannot use opencv_display=true without fast_viz=true"
    assert not (
        config.get("fast_viz") and not config.get("viz")
    ), "Must use viz=true with fast_viz=true"
    assert not (
        config.get("fast_viz") and config.get("viz_extension") == "pdf"
    ), "Cannot use pdf extension with fast_viz=true"

    resize = config.get("resize", [])
    if len(resize) == 2 and resize[1] == -1:
        config["resize"] = resize[0:1]
    if len(resize) == 2:
        print(f"Will resize to {resize[0]}x{resize[1]} (WxH)")
    elif len(resize) == 1 and resize[0] > 0:
        print(f"Will resize max dimension to {resize[0]}")
    elif len(resize) == 1:
        print("Will not resize images")
    elif len(resize) > 2:
        raise ValueError("Cannot specify more than two integers for resize")


def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    if not config_path:
        logger.error("No configuration file specified")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        sys.exit(1)


def create_config(opt):
    """Create configuration dictionary from YAML and command-line arguments."""
    # Load YAML configuration (required)
    yaml_config = load_yaml_config(opt.config)
    
    # # Override any settings with command-line arguments if provided
    # for key, value in vars(opt).items():
    #     # Skip the config file path itself
    #     if key == 'config':
    #         continue
            
        # # Only override if the value is not None (i.e., it was explicitly set)
        # if value is not None:
        #     # Handle special cases for nested dictionaries
        #     if key == 'nms_radius' or key == 'keypoint_threshold' or key == 'max_keypoints':
        #         yaml_config.setdefault('superpoint', {})
        #         yaml_config['superpoint'][key] = value
        #     elif key == 'superglue' or key == 'sinkhorn_iterations' or key == 'match_threshold':
        #         yaml_config.setdefault('superglue', {})
        #         if key == 'superglue':
        #             yaml_config['superglue']['weights'] = value
        #         else:
        #             yaml_config['superglue'][key] = value
        #     elif key == 'ransac_enabled':
        #         yaml_config.setdefault('ransac', {})
        #         yaml_config['ransac'][key] = value
        #     elif key == 'knn_ratio' or key == 'knn_distance':
        #         yaml_config.setdefault('knn', {})
        #         yaml_config['knn'][key] = value
        #     # Boolean flags need special handling
        #     elif isinstance(value, bool) and value is True:
        #         yaml_config[key] = value
        #     # Handle all other parameters
        #     elif value is not None:
        #         yaml_config[key] = value
    
    # Ensure all required settings have defaults if not in YAML
    yaml_config.setdefault('input_pairs', "/home/user/data/maploc_data/gen_featpts_gt_dataset/gen_local_feature_dataset/P11_ent1_route1_case2_P11_ent1_route1_case4/scannet_pairs.txt")
    yaml_config.setdefault('input_dir', "/home/user/data/maploc_data/gen_featpts_gt_dataset/gen_local_feature_dataset/P11_ent1_route1_case2_P11_ent1_route1_case4/")
    yaml_config.setdefault('output_dir', "/home/user/data/maploc_data/gen_featpts_gt_dataset/gen_local_feature_dataset/P11_ent1_route1_case2_P11_ent1_route1_case4/dump_match_pairs/")
    yaml_config.setdefault('max_length', -1)
    yaml_config.setdefault('resize', [640, 480])
    yaml_config.setdefault('resize_float', False)
    yaml_config.setdefault('viz', False)
    yaml_config.setdefault('eval', False)
    yaml_config.setdefault('fast_viz', False)
    yaml_config.setdefault('cache', False)
    yaml_config.setdefault('show_keypoints', False)
    yaml_config.setdefault('viz_extension', "png")
    yaml_config.setdefault('opencv_display', False)
    yaml_config.setdefault('shuffle', False)
    yaml_config.setdefault('force_cpu', False)
    yaml_config.setdefault('viz_comparison', False)
    yaml_config.setdefault('compare_sg_knn_ransac', False)
    yaml_config.setdefault('compare_knn_fb_ransac', False)
    yaml_config.setdefault('compare_fb_sg_ransac', False)
    yaml_config.setdefault('use_fb', False)
    yaml_config.setdefault('fb_config', "fb_config.yaml")
    yaml_config.setdefault('line_width', 0.3)
    yaml_config.setdefault('pose_threshold', 1.0)
    
    # Set up nested dictionaries with defaults if not present
    if 'superpoint' not in yaml_config:
        yaml_config['superpoint'] = {}
    yaml_config['superpoint'].setdefault('nms_radius', 4)
    yaml_config['superpoint'].setdefault('keypoint_threshold', 0.005)
    yaml_config['superpoint'].setdefault('max_keypoints', 1024)
    
    if 'superglue' not in yaml_config:
        yaml_config['superglue'] = {}
    yaml_config['superglue'].setdefault('weights', 'outdoor')
    yaml_config['superglue'].setdefault('sinkhorn_iterations', 20)
    yaml_config['superglue'].setdefault('match_threshold', 0.2)
    
    if 'ransac' not in yaml_config:
        yaml_config['ransac'] = {}
    logger.info(f"ransac: {yaml_config['ransac']}")
    yaml_config['ransac'].setdefault('enabled', False)
    yaml_config['ransac'].setdefault('ransac_threshold', 3.0)
    yaml_config['ransac'].setdefault('ransac_method', 'fundamental')
    
    if 'knn' not in yaml_config:
        yaml_config['knn'] = {}
    yaml_config['knn'].setdefault('knn_ratio', 0.8)
    yaml_config['knn'].setdefault('knn_distance', 'L2')
    
    # Ensure image preprocessing settings are present
    if 'day_preprocess' not in yaml_config:
        yaml_config['day_preprocess'] = [{"name": "clahe", "params": {"clip_limit": 3.0, "tile_grid_size": [8, 8]}}]
    if 'night_preprocess' not in yaml_config:
        yaml_config['night_preprocess'] = [{"name": "denoise", "params": {"h": 10.0}}]
    
    if 'featurebooster' not in yaml_config:
        yaml_config['featurebooster'] = {
            "keypoint_dim": 3,
            "keypoint_encoder": [32, 64, 128, 256],
            "descriptor_encoder": [256, 256],
            "descriptor_dim": 256,
            "Attentional_layers": 9,
            "l2_normalization": True,
            "output_dim": 256,
        }
    
    # Validate the complete configuration
    validate_args(yaml_config)
    
    # Re-format the configuration for the pipeline
    config = {
        "input_pairs": yaml_config['input_pairs'],
        "input_dir": yaml_config['input_dir'],
        "output_dir": yaml_config['output_dir'],
        "max_length": yaml_config['max_length'],
        "resize": yaml_config['resize'],
        "resize_float": yaml_config['resize_float'],
        "superpoint": yaml_config['superpoint'],
        "superglue": yaml_config['superglue'],
        "featurebooster": yaml_config['featurebooster'],
        "viz": yaml_config['viz'],
        "eval": yaml_config['eval'],
        "fast_viz": yaml_config['fast_viz'],
        "cache": yaml_config['cache'],
        "show_keypoints": yaml_config['show_keypoints'],
        "viz_extension": yaml_config['viz_extension'],
        "opencv_display": yaml_config['opencv_display'],
        "shuffle": yaml_config['shuffle'],
        "force_cpu": yaml_config['force_cpu'],
        "ransac": yaml_config['ransac'].get('enabled', False),
        "ransac_threshold": yaml_config['ransac']['ransac_threshold'],
        "ransac_method": yaml_config['ransac']['ransac_method'],
        "viz_comparison": yaml_config['viz_comparison'],
        "compare_sg_knn_ransac": yaml_config['compare_sg_knn_ransac'],
        "knn_ratio": yaml_config['knn']['knn_ratio'],
        "knn_distance": yaml_config['knn']['knn_distance'],
        "line_width": yaml_config['line_width'],
        "pose_threshold": yaml_config['pose_threshold'],
        "fb_config": yaml_config['fb_config'],
        "use_fb": yaml_config['use_fb'],
        "compare_knn_fb_ransac": yaml_config['compare_knn_fb_ransac'],
        "compare_fb_sg_ransac": yaml_config['compare_fb_sg_ransac'],
        "day_preprocess": yaml_config['day_preprocess'],
        "night_preprocess": yaml_config['night_preprocess'],
    }
    
    return config


def main():
    """Run the matching pipeline."""
    # Parse command line arguments
    opt = parse_args()
    
    # Create configuration by combining YAML config with command-line overrides
    config = create_config(opt)

    # Add preprocess modes to the configuration
    config['preprocess_modes'] = {
        'day': config.get('day_preprocess', []),
        'night': config.get('night_preprocess', [])
    }

    # Create and run the matching pipeline
    pipeline = MatchingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run the main function
    main() 