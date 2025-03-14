from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import torch


class ImagePreprocessor:
    """Base class for image preprocessing pipeline."""
    
    def __init__(self, processors: List[Dict[str, Any]] = None):
        """
        Initialize the image preprocessing pipeline.
        
        Args:
            processors: List of processor configurations, each with:
                - name: Name of the processor
                - params: Parameters for the processor
                - enabled: Whether to use this processor (default: True)
        """
        self.processors = []
        if processors:
            for processor_config in processors:
                name = processor_config["name"]
                params = processor_config.get("params", {})
                enabled = processor_config.get("enabled", True)
                
                if enabled:
                    processor_func = self._get_processor_by_name(name)
                    if processor_func:
                        self.processors.append((processor_func, params))
    
    def _get_processor_by_name(self, name: str) -> Optional[Callable]:
        """Get processor function by name."""
        processors = {
            "clahe": self.apply_clahe,
            "histogram_equalization": self.apply_histogram_equalization,
            "gamma_correction": self.apply_gamma_correction,
            "white_balance": self.apply_white_balance,
            "normalize": self.apply_normalization,
            "denoise": self.apply_denoising,
            "auto_contrast": self.apply_auto_contrast,
        }
        return processors.get(name)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all enabled processors to the image.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Processed image
        """
        if len(self.processors) == 0:
            return image
            
        result = image.copy()
        
        # Handle grayscale vs. color images
        is_grayscale = len(result.shape) == 2
        if is_grayscale:
            # For grayscale, expand to 3D temporarily for consistent processing
            result = np.expand_dims(result, axis=2)
        
        # Apply each processor in sequence
        for processor_func, params in self.processors:
            result = processor_func(result, **params)
        
        # Convert back to grayscale if input was grayscale
        if is_grayscale:
            if len(result.shape) == 3:
                result = result[:, :, 0]
        
        return result
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, 
                   tile_grid_size: tuple = (8, 8)) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        
        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            CLAHE processed image
        """
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Process each channel independently for color images
        result = np.zeros_like(image)
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        if channels == 1:
            result = clahe.apply(image[:, :, 0])
            result = np.expand_dims(result, axis=2)
        else:
            # For color images, apply CLAHE to the L channel in LAB color space
            if channels == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                l_clahe = clahe.apply(l)
                lab = cv2.merge((l_clahe, a, b))
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Apply to each channel independently
                for i in range(channels):
                    result[:, :, i] = clahe.apply(image[:, :, i])
                    
        return result
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply standard histogram equalization.
        
        Args:
            image: Input image
            
        Returns:
            Histogram equalized image
        """
        result = np.zeros_like(image)
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        if channels == 1:
            result = cv2.equalizeHist(image[:, :, 0])
            result = np.expand_dims(result, axis=2)
        else:
            # For color images, apply equalization to each channel
            for i in range(channels):
                result[:, :, i] = cv2.equalizeHist(image[:, :, i])
                
        return result
    
    def apply_gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply gamma correction.
        
        Args:
            image: Input image
            gamma: Gamma value (1.0 is no change, <1.0 brightens, >1.0 darkens)
            
        Returns:
            Gamma corrected image
        """
        # Make sure image is converted to float for gamma operation
        img_float = image.astype(np.float32) / 255.0
        
        # Apply gamma correction
        result = np.power(img_float, gamma)
        
        # Convert back to original data type
        result = (result * 255).clip(0, 255).astype(image.dtype)
        
        return result
    
    def apply_white_balance(self, image: np.ndarray, method: str = "gray_world") -> np.ndarray:
        """
        Apply white balance correction.
        
        Args:
            image: Input image
            method: White balance method ('gray_world' or 'perfect_reflector')
            
        Returns:
            White balanced image
        """
        # Only process color images
        if image.shape[2] < 3:
            return image
            
        if method == "gray_world":
            # Gray world assumption: the average of each channel should be equal
            result = image.astype(np.float32)
            avg_b = np.mean(result[:, :, 0])
            avg_g = np.mean(result[:, :, 1])
            avg_r = np.mean(result[:, :, 2])
            avg = (avg_r + avg_g + avg_b) / 3
            
            result[:, :, 0] = np.clip(result[:, :, 0] * (avg / avg_b), 0, 255)
            result[:, :, 1] = np.clip(result[:, :, 1] * (avg / avg_g), 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] * (avg / avg_r), 0, 255)
            
            return result.astype(image.dtype)
            
        elif method == "perfect_reflector":
            # Perfect reflector assumption: the brightest pixel should be white
            result = image.astype(np.float32)
            max_b = np.max(result[:, :, 0])
            max_g = np.max(result[:, :, 1])
            max_r = np.max(result[:, :, 2])
            
            result[:, :, 0] = np.clip(result[:, :, 0] * (255 / max_b), 0, 255)
            result[:, :, 1] = np.clip(result[:, :, 1] * (255 / max_g), 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] * (255 / max_r), 0, 255)
            
            return result.astype(image.dtype)
            
        return image
    
    def apply_normalization(self, image: np.ndarray, 
                         method: str = "minmax") -> np.ndarray:
        """
        Apply image normalization.
        
        Args:
            image: Input image
            method: Normalization method ('minmax', 'zscore')
            
        Returns:
            Normalized image
        """
        result = image.astype(np.float32)
        
        if method == "minmax":
            # Min-max normalization for each channel
            for c in range(image.shape[2]):
                min_val = np.min(result[:, :, c])
                max_val = np.max(result[:, :, c])
                if max_val > min_val:
                    result[:, :, c] = (result[:, :, c] - min_val) * (255.0 / (max_val - min_val))
                
        elif method == "zscore":
            # Z-score normalization for each channel
            for c in range(image.shape[2]):
                mean_val = np.mean(result[:, :, c])
                std_val = np.std(result[:, :, c])
                if std_val > 0:
                    result[:, :, c] = ((result[:, :, c] - mean_val) / std_val) * 64 + 128
        
        return np.clip(result, 0, 255).astype(image.dtype)
    
    def apply_denoising(self, image: np.ndarray, 
                      h: float = 10.0, template_window_size: int = 7, 
                      search_window_size: int = 21) -> np.ndarray:
        """
        Apply denoising to the image.
        
        Args:
            image: Input image
            h: Filter strength
            template_window_size: Size of template patch
            search_window_size: Size of window for searching similar patches
            
        Returns:
            Denoised image
        """
        # Fast non-local means denoising
        return cv2.fastNlMeansDenoisingColored(
            image, None, h, h, template_window_size, search_window_size
        ) if image.shape[2] >= 3 else cv2.fastNlMeansDenoising(
            image, None, h, template_window_size, search_window_size
        )
    
    def apply_auto_contrast(self, image: np.ndarray, clip_hist_percent: float = 1.0) -> np.ndarray:
        """
        Apply automatic contrast adjustment.
        
        Args:
            image: Input image
            clip_hist_percent: Percentage of histogram to clip at each end
            
        Returns:
            Contrast enhanced image
        """
        result = np.zeros_like(image)
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        for i in range(channels):
            channel = image[:, :, i]
            
            # Calculate histogram
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            hist_size = len(hist)
            
            # Calculate cumulative distribution from the histogram
            accumulator = []
            accumulator.append(float(hist[0]))
            for index in range(1, hist_size):
                accumulator.append(accumulator[index - 1] + float(hist[index]))
            
            # Locate points to clip
            maximum = accumulator[-1]
            clip_hist_percent *= (maximum/100.0)
            clip_hist_percent /= 2.0
            
            # Locate left cut
            minimum_gray = 0
            while accumulator[minimum_gray] < clip_hist_percent:
                minimum_gray += 1
            
            # Locate right cut
            maximum_gray = hist_size - 1
            while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
                maximum_gray -= 1
            
            # Calculate alpha and beta values
            alpha = 255 / (maximum_gray - minimum_gray)
            beta = -minimum_gray * alpha
            
            # Apply contrast stretching
            result[:, :, i] = cv2.convertScaleAbs(channel, alpha=alpha, beta=beta)
            
        return result

def preprocess_torch_image(image: torch.Tensor, processor: ImagePreprocessor) -> torch.Tensor:
    """
    Preprocess a PyTorch tensor image.
    
    Args:
        image: Input image tensor (CxHxW)
        processor: ImagePreprocessor instance
    
    Returns:
        Processed image tensor
    """
    # Convert from tensor to numpy for preprocessing
    if image.dim() == 3:  # CxHxW
        image_np = image.permute(1, 2, 0).cpu().numpy()
    elif image.dim() == 4:  # BxCxHxW (batch of 1)
        image_np = image[0].permute(1, 2, 0).cpu().numpy()
    else:
        raise ValueError(f"Unexpected tensor shape: {image.shape}")
    
    # Apply preprocessing
    processed_np = processor.process(image_np)
    
    # Convert back to tensor
    if len(processed_np.shape) == 2:  # Handle grayscale result
        processed = torch.from_numpy(processed_np).unsqueeze(0)
    else:
        processed = torch.from_numpy(processed_np).permute(2, 0, 1)
    
    # Match original tensor device and dimensions
    processed = processed.to(image.device)
    if image.dim() == 4:  # Add batch dimension if needed
        processed = processed.unsqueeze(0)
    
    return processed

# Example usage:
# Daytime preprocessing configuration
processor_config_day = [
    {"name": "clahe", "params": {"clip_limit": 3.0, "tile_grid_size": (8, 8)}},
    {"name": "gamma_correction", "params": {"gamma": 0.8}},
]
preprocessor_day = ImagePreprocessor(processor_config_day)
# processed_image_day = preprocessor_day.process(image)

# Nighttime preprocessing configuration
processor_config_night = [
    {"name": "denoise", "params": {"h": 10.0, "template_window_size": 7, "search_window_size": 21}},
    {"name": "auto_contrast", "params": {"clip_hist_percent": 1.0}},
]
preprocessor_night = ImagePreprocessor(processor_config_night)
# processed_image_night = preprocessor_night.process(image)
