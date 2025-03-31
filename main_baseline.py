#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import yaml

# Create logger
logger = logging.getLogger("GenBaselinePipeline")

from pipeline_baseline import GenBaselinePipeline


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

    return parser.parse_args()

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
    
    # Ensure all required settings have defaults if not in YAML
    yaml_config.setdefault('input_dir', "/home/user/data/maploc_data/gen_featpts_gt_dataset/gen_local_feature_dataset/")
    yaml_config.setdefault('output_dir', "/home/user/data/maploc_data/gen_featpts_gt_dataset/gen_local_feature_dataset/")
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
        yaml_config['day_preprocess'] = []
    if 'night_preprocess' not in yaml_config:
        yaml_config['night_preprocess'] = []

    # Re-format the configuration for the pipeline
    config = {
        # "input_pairs": yaml_config['input_pairs'],
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
    pipeline = GenBaselinePipeline(config)
    
    # KNN matching  
    pipeline.run()
    
    # Plot violin diagram
    pipeline.plot_violin_diagram()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run the main function
    main() 