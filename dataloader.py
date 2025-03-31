#!/usr/bin/env python3
"""
功能说明
这个脚本实现了以下功能：

- 读取case目录：读取指定的案例目录，验证必要的子目录和文件是否存在。
- 加载图像对：从 scannet_pairs.txt 读取白天-黑夜图像对列表。
- 处理每对图像：
    - 读取白天和黑夜图像
    - 从 day_label 和 night_label 目录加载 SuperPoint 特征点
    - 将 SuperPoint 特征点可视化并保存到 vis_superpoint 目录
    - 从 dump_match_pairs 目录读取匹配文件 (.npz)
    - 提取 RANSAC 过滤后的匹配点
    - 将 RANSAC 匹配点可视化并保存到 vis_ransac 目录
    - 创建匹配连线可视化，显示两个图像之间的特征点对应关系
    - 输出目录结构
运行后，脚本将在案例目录下创建两个新的子目录：

- vis_superpoint：包含 SuperPoint 特征点可视化
- vis_ransac：包含 RANSAC 过滤后的匹配点可视化和匹配线可视化

每个可视化图像包含特征点数量和其他相关信息的标注。
"""

import argparse
import logging
import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')  # 非交互式后端，适合服务器环境

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("MapLocDataLoader")


class MapLocDataLoader:
    """MapLoc数据集加载和可视化类"""
    
    def __init__(self, case_dir, use_npz_keypoints=False):
        """
        初始化数据加载器
        
        Args:
            case_dir: 案例目录路径
            use_npz_keypoints: 是否从npz文件中提取特征点，而不是从day_label/night_label目录
        """
        self.case_dir = Path(case_dir)
        self.use_npz_keypoints = use_npz_keypoints
        
        if not self.case_dir.exists() or not self.case_dir.is_dir():
            raise ValueError(f"案例目录不存在或不是有效目录: {case_dir}")
            
        # 验证必要的子目录是否存在
        expected_dirs = ["day", "night", "dump_match_pairs"]
        if not self.use_npz_keypoints:
            expected_dirs.extend(["day_label", "night_label"])
            
        for dir_name in expected_dirs:
            if not (self.case_dir / dir_name).exists():
                logger.warning(f"目录 {dir_name} 在案例目录中不存在")
        
        # 验证scannet_pairs.txt是否存在
        self.pairs_file = self.case_dir / "scannet_pairs.txt"
        if not self.pairs_file.exists():
            raise ValueError(f"匹配对文件不存在: {self.pairs_file}")
            
        # 创建输出目录
        self.vis_sp_dir = self.case_dir / "vis_superpoint"
        self.vis_ransac_dir = self.case_dir / "vis_ransac"
        os.makedirs(self.vis_sp_dir, exist_ok=True)
        os.makedirs(self.vis_ransac_dir, exist_ok=True)
        
        logger.info(f"初始化MapLocDataLoader，案例目录: {case_dir}")
        
    def load_image_pairs(self):
        """
        加载图像对列表
        
        Returns:
            包含图像对信息的列表[(day_img_name, night_img_name), ...]
        """
        pairs = []
        with open(self.pairs_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    day_img, night_img = parts[0], parts[1]
                    pairs.append((day_img, night_img))
        
        logger.info(f"从 {self.pairs_file} 加载了 {len(pairs)} 对图像")
        return pairs
    
    def read_image(self, img_path):
        """
        读取图像
        
        Args:
            img_path: 图像路径
            
        Returns:
            读取的图像(BGR格式)
        """
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"无法读取图像: {img_path}")
            return None
        return img
    
    def load_superpoint_keypoints(self, label_path):
        """
        加载SuperPoint特征点
        
        Args:
            label_path: 特征点文件路径(.npy)
            
        Returns:
            特征点坐标数组
        """
        try:
            keypoints = np.load(str(label_path))
            return keypoints
        except Exception as e:
            logger.error(f"加载特征点文件失败 {label_path}: {e}")
            return None
    
    def visualize_keypoints(self, img, keypoints, title="Keypoints", size=2, color=(0, 0, 255)):
        """
        在图像上可视化特征点
        
        Args:
            img: 输入图像
            keypoints: 特征点坐标
            title: 标题
            size: 特征点大小
            color: 特征点颜色 (BGR)
            
        Returns:
            带有特征点的图像
        """
        vis_img = img.copy()
        
        # 绘制特征点
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(vis_img, (x, y), size, color, -1)
        
        # 添加标题和特征点数量信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        info_text = f"{title}: {len(keypoints)} points"
        cv2.putText(vis_img, info_text, (10, 30), font, 0.7, color, 2)
        
        return vis_img
    
    def load_and_parse_matches(self, match_file):
        """
        加载和解析匹配文件
        
        Args:
            match_file: 匹配文件路径(.npz)
            
        Returns:
            匹配数据字典
        """
        try:
            match_data = np.load(str(match_file))
            # 将数据转换为字典
            data_dict = {key: match_data[key] for key in match_data.files}
            return data_dict
        except Exception as e:
            logger.error(f"加载匹配文件失败 {match_file}: {e}")
            return None
            
    def create_matching_visualization(self, img1, img2, kpts1, kpts2, title="Matches"):
        """
        创建匹配可视化
        
        Args:
            img1: 第一张图像
            img2: 第二张图像
            kpts1: 第一张图像的特征点
            kpts2: 第二张图像的特征点
            title: 可视化标题
            
        Returns:
            匹配可视化图像
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 创建组合图像，左侧是img1，右侧是img2（水平放置）
        vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis_img[:h1, :w1] = img1
        vis_img[:h2, w1:w1+w2] = img2
        
        # 在图像上绘制匹配特征点
        color = (0, 255, 0)  # 绿色
        for i in range(min(len(kpts1), len(kpts2))):
            pt1 = (int(kpts1[i][0]), int(kpts1[i][1]))
            pt2 = (int(kpts2[i][0]) + w1, int(kpts2[i][1]))
            cv2.circle(vis_img, pt1, 2, color, -1)
            cv2.circle(vis_img, pt2, 2, color, -1)
            cv2.line(vis_img, pt1, pt2, color, 1)
        
        # 添加标题和匹配点数量信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        info_text = f"{title}: {len(kpts1)} matches"
        cv2.putText(vis_img, info_text, (10, 30), font, 0.7, (0, 255, 0), 2)
        
        return vis_img
        
    def process_image_pair(self, day_img_name, night_img_name):
        """
        处理单个图像对
        
        Args:
            day_img_name: 白天图像文件名
            night_img_name: 黑夜图像文件名
        """
        logger.info(f"处理图像对: {day_img_name} - {night_img_name}")
        
        # 构建文件路径
        day_img_path = self.case_dir / "day" / day_img_name
        night_img_path = self.case_dir / "night" / night_img_name
        
        # 去掉文件扩展名
        day_stem = Path(day_img_name).stem
        night_stem = Path(night_img_name).stem
        
        # 构建匹配文件路径
        match_file = self.case_dir / "dump_match_pairs" / f"{day_stem}_{night_stem}_matches.npz"
        
        # 读取图像
        day_img = self.read_image(day_img_path)
        night_img = self.read_image(night_img_path)
        
        if day_img is None or night_img is None:
            logger.error("无法读取图像，跳过此对")
            return
        
        # 根据配置选择从不同来源加载SuperPoint特征点
        if self.use_npz_keypoints:
            # 如果启用从npz文件加载特征点
            if not match_file.exists():
                logger.error(f"匹配文件不存在: {match_file}")
                return
                
            match_data = self.load_and_parse_matches(match_file)
            if match_data is None:
                logger.error("无法加载匹配数据，跳过此对")
                return
            
            # 从npz文件中提取原始SuperPoint特征点
            day_kpts = match_data.get("keypoints0")
            night_kpts = match_data.get("keypoints1")
            
            if day_kpts is None or night_kpts is None:
                logger.error("匹配文件中缺少原始SuperPoint特征点数据")
                return
                
            logger.info(f"从npz文件加载了 {len(day_kpts)} 个白天特征点和 {len(night_kpts)} 个黑夜特征点")
        else:
            # 否则从day_label和night_label目录加载特征点
            day_label_path = self.case_dir / "day_label" / f"{day_img_name}.npy"
            night_label_path = self.case_dir / "night_label" / f"{night_img_name}.npy"
            
            day_kpts = self.load_superpoint_keypoints(day_label_path)
            night_kpts = self.load_superpoint_keypoints(night_label_path)
            
            if day_kpts is None or night_kpts is None:
                logger.error("无法从label目录加载特征点，跳过此对")
                return
                
            logger.info(f"从label目录加载了 {len(day_kpts)} 个白天特征点和 {len(night_kpts)} 个黑夜特征点")
        
        # 可视化SuperPoint特征点
        day_vis = self.visualize_keypoints(day_img, day_kpts, "Day SuperPoint", color=(0, 0, 255))
        night_vis = self.visualize_keypoints(night_img, night_kpts, "Night SuperPoint", color=(0, 0, 255))
        
        # 保存SuperPoint特征点可视化
        sp_output_path = self.vis_sp_dir / f"{day_stem}_{night_stem}_superpoint.jpg"
        combined_sp = np.hstack([day_vis, night_vis])
        cv2.imwrite(str(sp_output_path), combined_sp)
        logger.info(f"保存SuperPoint可视化: {sp_output_path}")
        
        # 加载匹配文件以获取RANSAC过滤后的匹配点
        if not match_file.exists():
            logger.error(f"匹配文件不存在: {match_file}")
            return
            
        if not self.use_npz_keypoints or match_data is None:
            match_data = self.load_and_parse_matches(match_file)
            if match_data is None:
                logger.error("无法加载匹配数据，跳过此对")
                return
        
        # 提取RANSAC过滤后的匹配点
        mkpts0_ransac = match_data.get("mkpts0_ransac")
        mkpts1_ransac = match_data.get("mkpts1_ransac")
        
        if mkpts0_ransac is None or mkpts1_ransac is None:
            logger.error("匹配文件中缺少RANSAC匹配点数据")
            return
            
        # 可视化RANSAC匹配点
        day_ransac_vis = self.visualize_keypoints(day_img, mkpts0_ransac, "Day RANSAC", color=(0, 255, 0))
        night_ransac_vis = self.visualize_keypoints(night_img, mkpts1_ransac, "Night RANSAC", color=(0, 255, 0))
        
        # 保存RANSAC匹配点可视化
        ransac_output_path = self.vis_ransac_dir / f"{day_stem}_{night_stem}_ransac.jpg"
        combined_ransac = np.hstack([day_ransac_vis, night_ransac_vis])
        cv2.imwrite(str(ransac_output_path), combined_ransac)
        logger.info(f"保存RANSAC可视化: {ransac_output_path}")
        
        # 创建匹配可视化
        match_vis = self.create_matching_visualization(day_img, night_img, mkpts0_ransac, mkpts1_ransac, "RANSAC Matches")
        match_output_path = self.vis_ransac_dir / f"{day_stem}_{night_stem}_matches.jpg"
        cv2.imwrite(str(match_output_path), match_vis)
        logger.info(f"保存匹配线可视化: {match_output_path}")
        
    def process_all_pairs(self):
        """处理案例目录中的所有图像对"""
        pairs = self.load_image_pairs()
        logger.info(f"开始处理 {len(pairs)} 对图像...")
        
        for i, (day_img, night_img) in enumerate(pairs):
            logger.info(f"处理对 {i+1}/{len(pairs)}: {day_img} - {night_img}")
            self.process_image_pair(day_img, night_img)
            
        logger.info("所有图像对处理完成")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="MapLoc数据集加载和可视化工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--case_dir",
        type=str,
        required=True,
        help="案例目录路径，例如 P11_ent1_route1_case2-P11_ent1_route1_case4/"
    )
    
    parser.add_argument(
        "--use_npz_keypoints",
        action="store_true",
        help="从npz文件中提取特征点，而不是从day_label/night_label目录"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    try:
        # 创建数据加载器并处理所有图像对
        data_loader = MapLocDataLoader(args.case_dir, args.use_npz_keypoints)
        data_loader.process_all_pairs()
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # python dataLoader.py --case_dir /path/to/P11_ent1_route1_case2-P11_ent1_route1_case4
    main()