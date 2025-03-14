from abc import ABC, abstractmethod
from typing import Dict


class FeatureMatcher(ABC):
    """
    Base class for feature matchers like SuperGlue.
    
    This abstract class defines the interface for feature matching algorithms.
    All concrete matching implementations should inherit from this class.
    """

    @abstractmethod
    def match(self, data: Dict) -> Dict:
        """
        Match features between two images.
        
        Args:
            data: Dictionary containing keypoints, descriptors, and other data
                 needed for matching.
                 
        Returns:
            Dictionary containing matching results, including:
            - matches0: Array of indices for matching points (-1 if no match)
            - matching_scores0: Confidence scores for each match
        """
        pass
