import argparse
import datetime
import getpass
import json
import os
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np
import yaml  # 用于加载YAML配置
from loguru import logger
from scipy.spatial.transform import Rotation, Slerp

current_user = getpass.getuser()


@dataclass
class Trajectory:
    timestamp: float
    position: np.ndarray
    quaternion: np.ndarray


def get_cases_list(dataset_dir: str, list_name: str) -> list:
    logger.info(f"Dataset: {dataset_dir}")
    cases_list = read_csv(os.path.join(dataset_dir, list_name))
    cases_list = sorted(cases_list, key=lambda x: x["day1"])
    assert cases_list and len(cases_list) > 0, "Empty Dataset"
    logger.info(f"Size of dataset: {len(cases_list)}")
    return cases_list


def read_csv(filepath: str) -> list:
    if not os.path.exists(filepath):
        logger.info(f"File {filepath} does not exist.")
        return None

    with open(filepath, "r") as file:
        lines = file.readlines()

    if not lines:
        logger.info("File is empty.")
        return None

    # 第一行为标题
    headers = lines[0].strip().split(",")
    headers = [i for i in headers if i != ""]
    cases_list = []

    # 其他数据
    for line in lines[1:]:
        data = line.strip().split(",")
        # data = [i for i in data if i != ""]
        single_case = {"day1": "", "day2": "", "night1": "", "night2": ""}
        single_case["day1"] = data[1]
        single_case["day2"] = data[2]
        single_case["night1"] = data[3]
        single_case["night2"] = data[4]
        cases_list.append(single_case)
    return cases_list


def setup_logging_system(
    loc_level: int,
) -> str:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = "log"
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, f"gen_locfeat_gt_{current_time}.log")
    # Configure logger with best practices
    logger.add(
        LOG_FILE,
        level=loc_level,  # Set log level from configuration
        rotation="20 MB",  # Rotate log file when it reaches 10MB
        retention="10 days",  # Retain log files for 10 days
        # format="{time:YYYYMMDD-HH:mm:ss} | {level} | {message}",
        format="{time:YYYYMMDD-HH:mm:ss}|{level}|{file}:{line}|{message}",
        enqueue=True,  # Use queued logging for thread/process safety
    )

    return LOG_FILE


def load_trajectory_files(case: str, gt_path: str) -> dict:
    """Read ground truth trajectory files for a case."""
    if not os.path.exists(gt_path):
        logger.error(f"Path {gt_path} does not exist.")
        return None

    case_name = case.split("_")[0]  # Extract case prefix (PXX)
    traj_dict = {f"case{i}": "" for i in range(1, 5)}

    # Look for directories matching the case name
    for dir_name in os.listdir(gt_path):
        if not dir_name.startswith(f"outdoor-{case_name}"):
            continue

        gt_traj_dir = os.path.join(gt_path, dir_name)

        # Find all trajectory files for this case
        for file in os.listdir(gt_traj_dir):
            if file.startswith(f"outdoor-{case_name}") and file.endswith(
                "_loc_trajectory.txt"
            ):
                for case_num in range(1, 5):
                    if f"case{case_num}_loc_trajectory.txt" in file:
                        traj_dict[f"case{case_num}"] = os.path.join(gt_traj_dir, file)
                        break

    return traj_dict


def parse_trajectory_file(traj_file: str) -> list:
    """Parse trajectory file into a list of Trajectory objects."""
    trajectories = []
    if not os.path.exists(traj_file):
        logger.error(f"Trajectory file {traj_file} does not exist")
        return trajectories

    with open(traj_file, "r") as f:
        for line in f:
            data = line.strip().split()
            if len(data) >= 8:  # timestamp, px, py, pz, qx, qy, qz，qw
                timestamp = float(data[0])
                position = np.array([float(data[1]), float(data[2]), float(data[3])])
                quaternion = np.array(
                    [float(data[4]), float(data[5]), float(data[6]), float(data[7])]
                )
                trajectories.append(Trajectory(timestamp, position, quaternion))
    logger.info(f"Loaded {len(trajectories)} trajectory points from {traj_file}")
    return trajectories


def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions."""
    return np.linalg.norm(pos1 - pos2)


def quaternion_to_euler(quaternion):
    """Convert quaternion to euler angles (roll, pitch, yaw)."""
    # quaternion format: [qx, qy, qz, qw]
    euler_angles = Rotation.from_quat(quaternion).as_euler("zyx", degrees=False)
    yaws_rad = euler_angles[0]

    return np.array([0, 0, yaws_rad])


def calculate_angle_difference(quat1, quat2):
    """Calculate the angular difference between two quaternions in degrees."""
    # Convert quaternions to euler angles
    euler1 = quaternion_to_euler(quat1)
    euler2 = quaternion_to_euler(quat2)

    # Calculate the difference in yaw (heading) angle
    yaw_diff_rad = abs(euler1[2] - euler2[2])

    # Normalize to the range [0, pi]
    if yaw_diff_rad > np.pi:
        yaw_diff_rad = 2 * np.pi - yaw_diff_rad

    # Convert to degrees
    return np.degrees(yaw_diff_rad)


def find_closest_frame(reference_pos, reference_quaternion, trajectories):
    """Find the trajectory entry closest to the reference position and with similar orientation."""
    min_dist = float("inf")
    min_angle_deg = float("inf")
    closest_traj = None

    for traj in trajectories:
        dist = calculate_distance(reference_pos, traj.position)

        # Only compute angle for relatively close positions to save computation
        if dist < min_dist + 5.0:  # Consider positions within 5m of the current best
            angle_diff_deg = calculate_angle_difference(
                reference_quaternion, traj.quaternion
            )

            # Prioritize position match, but use angle as a tiebreaker for similar positions
            if dist < min_dist or (
                abs(dist - min_dist) < 0.1 and angle_diff_deg < min_angle_deg
            ):
                min_dist = dist
                min_angle_deg = angle_diff_deg
                closest_traj = traj

    return closest_traj, min_dist, min_angle_deg


def iterate_and_find_keyframe_pairs(
    day_case: str, night_case: str, gt_trajs: str, config: dict
):
    """
    Iterate through day trajectory, extract keyframes with distance gap,
    and find closest frames from night trajectory considering both position and orientation.

    Args:
        day_case: Day case name (e.g., "P11_ent1_route1_case1")
        night_case: Night case name (e.g., "P11_ent1_route1_case3")
        gt_trajs: Dictionary of trajectory file paths
        config: Configuration dictionary containing thresholds

    Returns:
        Dictionary containing day-night keyframe pairs
    """
    # Get case number from day_case and night_case
    out_path = config.get("out_path", "")
    if not os.path.exists(out_path):
        # create output directory
        os.makedirs(out_path, exist_ok=True)

    day_case_num = "".join(filter(str.isdigit, day_case.split("_")[-1]))
    night_case_num = "".join(filter(str.isdigit, night_case.split("_")[-1]))

    # Get trajectory files
    day_traj_file = gt_trajs.get(f"case{day_case_num}", "")
    night_traj_file = gt_trajs.get(f"case{night_case_num}", "")

    if not day_traj_file or not night_traj_file:
        logger.error(
            f"Missing trajectory files for day case {day_case} or night case {night_case}"
        )
        return {}

    logger.info(f"Processing day case: {day_case} with file: {day_traj_file}")
    logger.info(f"Processing night case: {night_case} with file: {night_traj_file}")

    # Parse trajectory files
    day_trajectories = parse_trajectory_file(day_traj_file)
    night_trajectories = parse_trajectory_file(night_traj_file)

    if not day_trajectories or not night_trajectories:
        logger.error(
            f"Failed to parse trajectories for day case {day_case} or night case {night_case}"
        )
        return {}

    # Get thresholds from config
    keyframe_dist_meter = config.get("keyframe_dist", 2.0)
    max_match_dist_meter = config.get("max_match_dist_threshold", 1.0)
    max_match_angle_deg = config.get("max_match_angle_threshold", 10.0)
    # Load new neighbor constraints from config
    neighbor_time_threshold = config.get("neighbor_time_threshold", 3.0)
    neighbor_distance_threshold = config.get("neighbor_distance_threshold", 4.0)

    logger.info(f"Using keyframe distance threshold: {keyframe_dist_meter}m")
    logger.info(f"Using max match distance threshold: {max_match_dist_meter}m")
    logger.info(f"Using max match angle threshold: {max_match_angle_deg}°")

    # Extract keyframes from day trajectory with specified distance gap
    keyframe_pairs = {}
    last_keyframe_pos = None
    pair_idx = 0

    for idx, day_traj in enumerate(day_trajectories):
        # If this is the first frame or we've moved more than the threshold since the last keyframe
        if (
            last_keyframe_pos is None
            or calculate_distance(day_traj.position, last_keyframe_pos)
            >= keyframe_dist_meter
        ):

            # Find closest frame in night trajectory
            closest_night_traj, matched_dist_m, matched_angle_deg = find_closest_frame(
                day_traj.position, day_traj.quaternion, night_trajectories
            )

            # Only add the pair if it meets both distance and angle criteria
            if (
                matched_dist_m <= max_match_dist_meter
                and matched_angle_deg <= max_match_angle_deg
            ):
                # Build candidate list: include closest_night_traj and additional neighbors
                candidate_night_frames = [closest_night_traj]
                for night_candidate in night_trajectories:
                    # Skip the closest frame itself
                    if night_candidate == closest_night_traj:
                        continue
                    time_diff = abs(night_candidate.timestamp - closest_night_traj.timestamp)
                    if time_diff <= neighbor_time_threshold:
                        abs_dis = calculate_distance(closest_night_traj.position, night_candidate.position)
                        if abs_dis < neighbor_distance_threshold:
                            candidate_night_frames.append(night_candidate)
                # For each candidate, create an independent keyframe pair entry
                for candidate in candidate_night_frames:
                    # Recompute distance and angle difference using the candidate night frame vs day_traj
                    candidate_dist = calculate_distance(day_traj.position, candidate.position)
                    candidate_angle = calculate_angle_difference(day_traj.quaternion, candidate.quaternion)
                    pair_key = f"pair_{pair_idx}"
                    keyframe_pairs[pair_key] = {
                        "day_case": day_case,
                        "day_timestamp": day_traj.timestamp,
                        "day_position": day_traj.position.tolist(),
                        "day_quaternion": day_traj.quaternion.tolist(),
                        "night_case": night_case,
                        "night_timestamp": candidate.timestamp,
                        "night_position": candidate.position.tolist(),
                        "night_quaternion": candidate.quaternion.tolist(),
                        "distance": float(candidate_dist),
                        "angle_diff": float(candidate_angle),
                    }
                    pair_idx += 1

            last_keyframe_pos = day_traj.position

    logger.info(
        f"Found {len(keyframe_pairs)} day-night keyframe pairs for {day_case} and {night_case}"
    )

    # Save keyframe pairs to a JSON file for later use
    kf_pair_dir = os.path.join(out_path, "keyframe_pairs")
    if not os.path.exists(kf_pair_dir):
        os.makedirs(kf_pair_dir, exist_ok=True)
    day_night_pairs_json = os.path.join(
        kf_pair_dir, f"{day_case}-{night_case}_pairs.json"
    )

    keyframe_pairs["json_file"] = day_night_pairs_json
    keyframe_pairs["day_case"] = day_case
    keyframe_pairs["night_case"] = night_case

    with open(day_night_pairs_json, "w") as f:
        json.dump(keyframe_pairs, f, indent=2)
    logger.info(f"Saved keyframe pairs to {day_night_pairs_json}")

    return keyframe_pairs


def read_all_keyframe_pairs(keyframe_pairs_dir: str) -> list:
    """
    读取目录下所有的关键帧配对 JSON 文件
    
    Args:
        keyframe_pairs_dir: 包含关键帧配对 JSON 文件的目录路径
        
    Returns:
        包含所有关键帧配对信息的列表
    """
    logger.info(f"Reading keyframe pairs from directory: {keyframe_pairs_dir}")
    
    if not os.path.exists(keyframe_pairs_dir):
        logger.error(f"Keyframe pairs directory {keyframe_pairs_dir} does not exist")
        return []
        
    # 列出目录中所有JSON文件
    json_files = [f for f in os.listdir(keyframe_pairs_dir) if f.endswith('_pairs.json')]
    logger.info(f"Found {len(json_files)} keyframe pair JSON files")
    
    all_keyframe_pairs = []
    for json_file in json_files:
        file_path = os.path.join(keyframe_pairs_dir, json_file)
        logger.info(f"Reading keyframe pairs from {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                keyframe_pairs = json.load(f)
                
                # 确保文件包含必要的元数据
                if 'day_case' not in keyframe_pairs or 'night_case' not in keyframe_pairs:
                    # 如果缺少必要的元数据，尝试从文件名中提取
                    if '_pairs.json' in json_file:
                        case_part = json_file.replace('_pairs.json', '')
                        if '-' in case_part:
                            day_case, night_case = case_part.split('-', 1)
                            keyframe_pairs['day_case'] = day_case
                            keyframe_pairs['night_case'] = night_case
                
                # 确保包含json_file路径
                if 'json_file' not in keyframe_pairs:
                    keyframe_pairs['json_file'] = file_path
                
                # 计算有多少对匹配的关键帧
                pair_count = sum(1 for key in keyframe_pairs.keys() if key.startswith('pair_'))
                logger.info(f"  - Found {pair_count} day-night keyframe pairs")
                
                all_keyframe_pairs.append(keyframe_pairs)
        except Exception as e:
            logger.error(f"Error reading keyframe pairs from {file_path}: {e}")
    
    logger.info(f"Total keyframe pair files loaded: {len(all_keyframe_pairs)}")
    return all_keyframe_pairs


def dump_imgs_and_vis_res(
    all_keyframe_pairs, out_path, pack_path, program_path, log_file
):
    """
    Dump keyframe images and visualization results to output path

    Args:
        case: Case dictionary containing day and night data
        out_path: Path to save results
    """
    for i, kf_pair in enumerate(all_keyframe_pairs):
        # if i != 0:
        #     continue  # TODO: Remove this check once we have a proper case structure

        json_file = kf_pair.get("json_file", "")
        if not json_file:
            logger.error("Missing keyframe pairs JSON file")
            continue

        day_case = kf_pair.get("day_case", "")
        night_case = kf_pair.get("night_case", "")

        day_pack_dir = os.path.join(pack_path, "outdoor", day_case)
        night_pack_dir = os.path.join(pack_path, "outdoor", night_case)

        if not os.path.exists(day_pack_dir) or not os.path.exists(night_pack_dir):
            logger.error(
                f"Missing pack directories for day case {day_case} or night case {night_case}"
            )
            continue

        # Create output directory for this case
        case_out_dir = os.path.join(out_path, f"{day_case}-{night_case}")
        if not os.path.exists(case_out_dir):
            os.makedirs(case_out_dir, exist_ok=True)
        command = [program_path, day_pack_dir, night_pack_dir, json_file, case_out_dir]

        logger.info(f"Running command to extract images: {command}")

        start_time = time.time()

        with open(log_file, "a") as log:
            try:
                log.write(
                    f"\n\n--- Start run bin cmd at: {datetime.datetime.now()} ---\n\n"
                )

                subprocess.run(
                    command,
                    stdout=log,  # Redirect stdout to the log file
                    stderr=subprocess.STDOUT,  # Redirect stderr to the log file
                    check=True,  # Raise an exception if the command fails
                    text=True,  # Handle output as text
                )
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Command failed for day pack {day_case} and night pack {night_case}. See log file {log_file} for details."
                )
                raise

        end_time = time.time()
        execution_time = end_time - start_time


def main():
    parser = argparse.ArgumentParser(
        description="Run dataset based localization evaluation"
    )
    parser.add_argument(
        "--config",
        default="config/gen_locfeat.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    # 加载YAML配置并使用命令行参数作为备选项
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    loc_level = config.get("log_level")
    log_file = setup_logging_system(loc_level)

    default_map_path = (
        "/mnt/data/lichenchen/batch_test/run_parking_test_driving_parking/map_dataset"
    )
    default_out_path = (
        "/mnt/data/lichenchen/batch_test/run_parking_test_driving_parking/loc_replay"
    )
    default_gt_path = (
        "/home/user/data/maploc_data/gen_featpts_gt_dataset/lidar_map_dataset"
    )
    raw_pack_path = config.get("raw_pack_path")
    list_name = config.get("csv_file")

    # 加载YAML配置并使用default_xxx_path作为备选项
    map_path = config.get("map_path", default_map_path)
    out_path = config.get("out_path", default_out_path)
    gt_path = config.get("gt_path", default_gt_path)

    maploc_path = config.get("maploc_path")
    logger.info(
        "----------- Start generate local_feature ground-truth dataset --------------"
    )
    logger.info("Parameters check: generate gt dataset with following params:")
    logger.info(f"Path to CSV file : {list_name}")
    logger.info(f"Path to raw pack datasets: {raw_pack_path}")
    logger.info(f"Path to sp_map directory: {map_path}")
    logger.info(f"Path to gt trajectory directory: {gt_path}")
    logger.info(f"Path to generated gt dataset directory: {out_path}")

    logger.info(
        "\n----------- Start run loc_init fillback and evaluation --------------"
    )
    cases_list = get_cases_list(dataset_dir=raw_pack_path, list_name=list_name)

    # 找出pack中的关键帧，读取真值中对应的map_traj和loc_traj，筛选距离相近的帧；
    # Dump关键帧以及候选帧的图片到out_path中, 并且保存对应的匹配信息

    logger.info(f"Start processing {len(cases_list)} cases")
    logger.info(f"Processing cases: {cases_list}")

    # all_keyframe_pairs = []
    # for case in cases_list:
    #     # day1 = case["day1"]
    #     day2 = case["day2"]
    #     # night1 = case["night1"]
    #     night2 = case["night2"]

    #     # Read gt trajectory from gt_path
    #     gt_trajs = load_trajectory_files(day2, gt_path)

    #     # Read day and night trajectory from gt_trajs with Trajectory dataclass format, iterate through day traj, extract keyframes with distance gap of 1m, and find closest frames from night traj. Record the day-night keyframe pairs.
    #     keyframe_pairs = iterate_and_find_keyframe_pairs(day2, night2, gt_trajs, config)
    #     all_keyframe_pairs.append(keyframe_pairs)


    #  read all_keyframe_pairs from extracted file
    all_keyframe_pairs_dir = os.path.join(out_path, "keyframe_pairs")
    all_keyframe_pairs = read_all_keyframe_pairs(all_keyframe_pairs_dir)

    # Dump 白天黑夜图片以及对应特征点描述子到out_path中
    program_path = os.path.join(
        maploc_path, "build_x64", "bin", "test_gen_locfeat_baseline"
    )
    dump_imgs_and_vis_res(
        all_keyframe_pairs,
        out_path,
        pack_path=raw_pack_path,
        program_path=program_path,
        log_file=log_file,
    )  # Dump关键帧以及候选帧的图片到out_path中


if __name__ == "__main__":
    main()
