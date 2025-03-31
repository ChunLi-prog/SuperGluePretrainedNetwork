# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import sys
from pathlib import Path

import torch

from .superglue import SuperGlue
from .superpoint import SuperPoint

feature_booster_path = Path(__file__).parent.parent / "FeatBooster/FeatureBooster"
sys.path.append(str(feature_booster_path))
from featurebooster import FeatureBooster

torch.set_grad_enabled(False)


class Matching(torch.nn.Module):
    """Image Matching Frontend (SuperPoint + SuperGlue)"""

    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get("superpoint", {}))
        self.superglue = SuperGlue(config.get("superglue", {}))

    def forward(self, data):
        """Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if "keypoints0" not in data:
            pred0 = self.superpoint({"image": data["image0"]})
            pred = {**pred, **{k + "0": v for k, v in pred0.items()}}
        if "keypoints1" not in data:
            pred1 = self.superpoint({"image": data["image1"]})
            pred = {**pred, **{k + "1": v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**pred, **self.superglue(data)}

        return pred


class FeatBoostEnhancedMatching(torch.nn.Module):
    """Image Matching Frontend with Feature Enhancement (SuperPoint + FeatureBooster + SuperGlue)"""

    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get("superpoint", {}))
        self.featurebooster = FeatureBooster(config.get("featurebooster", {}))
        self.superglue = SuperGlue(config.get("superglue", {}))

        # load the model
        import os

        # feat_bst_model = Path(__file__).parent / "FeatBooster/FeatureBooster/models"
        feat_bst_model = "/home/user/project/maploc/SuperGluePretrainedNetwork/FeatBooster/FeatureBooster/models"
        fb_descriptor = "SuperPoint+Boost-F"
        model_path = os.path.join(feat_bst_model, fb_descriptor + ".pth")
        
        model_path = "/home/user/project/maploc/SuperGluePretrainedNetwork/FeatBooster/FeatureBooster/models/SuperPoint+Boost-F.pth"
        print(model_path)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # self.featurebooster.cuda()
        self.featurebooster.eval()
        self.featurebooster.load_state_dict(torch.load(model_path))
        
    def _normalize_keypoints(self, keypoints, h, w):
        import numpy as np

        # h, w = image_shape[0], image_shape[1]
        x0 = w / 2
        y0 = h / 2
        scale = max(w, h) * 0.7
        kps = np.array(keypoints)
        kps[:, 0] = (keypoints[:, 0] - x0) / scale
        kps[:, 1] = (keypoints[:, 1] - y0) / scale
        return kps

    def forward(self, data):
        """Run SuperPoint, FeatureBooster, and SuperGlue with extensive debugging"""
        pred = {}

        # Extract SuperPoint features if not provided
        if "keypoints0" not in data:
            pred0 = self.superpoint({"image": data["image0"]})
            
            # Handle keypoints (list or tensor)
            if isinstance(pred0["keypoints"], (list, tuple)):
                kpts0 = pred0["keypoints"][0]  # Shape (N, 2)
            else:
                kpts0 = pred0["keypoints"][0]  # Get first batch item
            
            # Handle scores (list, tuple or tensor)
            if isinstance(pred0["scores"], (list, tuple)):
                scores0 = pred0["scores"][0]  # Shape (N,)
            else:
                scores0 = pred0["scores"][0]
            
            # Handle descriptors (list or tensor)
            if isinstance(pred0["descriptors"], (list, tuple)):
                desc0 = pred0["descriptors"][0]  # Shape (256, N)
            else:
                desc0 = pred0["descriptors"][0]  # Shape (256, N)

            kpts0_norm = self._normalize_keypoints(kpts0, data["shape0"], data["shape1"])
            kpts0_norm = torch.tensor(kpts0_norm, dtype=torch.float32)
            # Create 3D keypoints (x, y, score)
            kpts_3d = torch.cat([kpts0_norm, scores0.unsqueeze(-1)], dim=1)  # Shape (N, 3)
            
            # Transpose descriptors to (N, 256)
            desc_reshape = desc0.t()  # Shape (N, 256)
            
            # Apply L2 normalization before boosting
            desc_reshape = torch.nn.functional.normalize(desc_reshape, p=2, dim=1)
            
            # Enhance the descriptors
            enhanced_desc = self.featurebooster(desc_reshape, kpts_3d)
            
            # Normalize enhanced descriptors
            enhanced_desc = torch.nn.functional.normalize(enhanced_desc, p=2, dim=1)
            
            # Transpose back to shape expected by SuperGlue (256, N)
            enhanced_desc = enhanced_desc.t()  # Shape (256, N)
            
            # Make sure we have batch dimension
            enhanced_desc = enhanced_desc.unsqueeze(0)  # Shape (1, 256, N)
            
            # enhanced_desc = desc_reshape.t().unsqueeze(0)  # Skip enhancement
            kpts0 = kpts0.unsqueeze(0)  # Shape (1, N, 2)
            scores0 = scores0.unsqueeze(0)  # Shape (1, N)
            
            # Update with enhanced descriptors
            pred0["descriptors"] = enhanced_desc
            pred0["keypoints"] = kpts0
            pred0["scores"] = scores0
            
            pred = {**pred, **{k + "0": v for k, v in pred0.items()}}

        # Similar fix for image 1
        if "keypoints1" not in data:
            pred1 = self.superpoint({"image": data["image1"]})
            
            # Handle keypoints (list or tensor)
            if isinstance(pred1["keypoints"], (list, tuple)):
                kpts1 = pred1["keypoints"][0]  # Shape (N, 2)
            else:
                kpts1 = pred1["keypoints"][0]  # Get first batch item
            
            # Handle scores (list, tuple or tensor) 
            if isinstance(pred1["scores"], (list, tuple)):
                scores1 = pred1["scores"][0]  # Shape (N,)
            else:
                scores1 = pred1["scores"][0]
            
            # Handle descriptors (list or tensor)
            if isinstance(pred1["descriptors"], (list, tuple)):
                desc1 = pred1["descriptors"][0]  # Shape (256, N)
            else:
                desc1 = pred1["descriptors"][0]  # Shape (256, N)

            kpts1_norm = self._normalize_keypoints(kpts1, data["shape0"], data["shape1"])
            kpts1_norm = torch.tensor(kpts1_norm, dtype=torch.float32)
            # Rest of processing as before
            kpts_3d = torch.cat([kpts1_norm, scores1.unsqueeze(-1)], dim=1)

            desc_reshape = desc1.t()
            desc_reshape = torch.nn.functional.normalize(desc_reshape, p=2, dim=1)
            enhanced_desc = self.featurebooster(desc_reshape, kpts_3d)
            enhanced_desc = torch.nn.functional.normalize(enhanced_desc, p=2, dim=1)
            enhanced_desc = enhanced_desc.t().unsqueeze(0)
            # enhanced_desc = desc_reshape.t().unsqueeze(0)  # Skip enhancement
            
            kpts1 = kpts1.unsqueeze(0)
            scores1 = scores1.unsqueeze(0)
            
            pred1["descriptors"] = enhanced_desc
            pred1["keypoints"] = kpts1
            pred1["scores"] = scores1
            
            pred = {**pred, **{k + "1": v for k, v in pred1.items()}}

        # Batch all features for SuperGlue
        data = {**data, **pred}
        print(f"Keys for SuperGlue: {list(data.keys())}")
        
        # Convert any remaining lists/tuples to tensors
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
        
        # Debug key shapes
        if "keypoints0" in data:
            print(f"Final keypoints0 shape: {data['keypoints0'].shape}")
        if "descriptors0" in data:
            print(f"Final descriptors0 shape: {data['descriptors0'].shape}")
        
        # Perform matching
        superglue_output = self.superglue(data)
        
        # Debug matches
        if "matches0" in superglue_output:
            valid_matches = (superglue_output["matches0"] > -1).sum().item()
            print(f"Number of valid matches found: {valid_matches}")
        
        pred = {**pred, **superglue_output}
        
        return pred