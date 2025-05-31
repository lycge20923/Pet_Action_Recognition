import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stgcn import ST_GCN
from .I3D import InceptionI3d
from ..utils.cli_args import ModelArguments, DataArguments

class ActionRecognitionModel(nn.Module):
    def __init__(self, 
                 model_params:ModelArguments,
                 data_params:DataArguments,
                 coords):
        super().__init__()
        self.only_optical_flow = model_params.only_optical_flow
        self.use_optical_flow = model_params.add_optical_flow
        
        # initiate model
        if not self.only_optical_flow: # at least we use skeleton information 
            self.skel_model = ST_GCN(params=model_params, data_params=data_params, coords=coords)
            self._skel_feat_dim = self.skel_model.fc.in_channels
        else: # skip skeleton information
            self.skel_model = None
            self._skel_feat_dim = 0
        # feat
        self.I3D = None
        self._flow_feat_dim = 0
        self._projected_flow_feat_dim = 0
        
        if self.use_optical_flow or self.only_optical_flow:
            self.I3D = InceptionI3d(in_channels=2)
            
            # load weight 
            i3d_weights_path = os.path.join(model_params.pretrained_weight_dir,
                                                model_params.I3D_weights_dir_name,
                                                model_params.I3D_weights_file_name)
            self.I3D.load_state_dict(torch.load(i3d_weights_path))
            self._flow_feat_dim = model_params.I3D_raw_feat_dim
            self._projected_flow_feat_dim = model_params.I3D_project_dim
            self.flow_feature_projector = nn.Linear(self._flow_feat_dim, self._projected_flow_feat_dim)
        
        # concate feature's size, for final classifier
        self.total_feature_dimension = self._skel_feat_dim
        if (self.use_optical_flow or self.only_optical_flow) and self.I3D is not None:
            self.total_feature_dimension += self._projected_flow_feat_dim
        self.final_classifier = nn.Linear(self.total_feature_dimension, model_params.num_classes)
    
        
    def forward(self, skeleton: torch.Tensor, flow: torch.Tensor = None):
        # 1. only optical flow
        if self.only_optical_flow:
            flow_map = self.I3D.extract_features(flow)
            flow_feat = flow_map.mean(dim=2).view(flow_map.size(0), -1)
            projected_flow_feat = self.flow_feature_projector(flow_feat)
            combined_feat = projected_flow_feat
            main_task_logits = self.final_classifier(combined_feat)
            return combined_feat, main_task_logits
        
        # 2. at least use skeleton information
        skel_feat, _ = self.skel_model(skeleton)
        combined_feat = skel_feat
        if self.use_optical_flow and self.I3D is not None and flow is not None:
            flow_map = self.I3D.extract_features(flow)
            flow_feat = flow_map.mean(dim=2).view(flow_map.size(0), -1)
            projected_flow_feat = self.flow_feature_projector(flow_feat)
            combined_feat = torch.cat([skel_feat, projected_flow_feat], dim=1)

        main_task_logits = self.final_classifier(combined_feat)
        return combined_feat, main_task_logits

class ContrastiveActionWrapper(nn.Module):
    def __init__(self,
                 backbone: ActionRecognitionModel,
                 emb_dim: int,                        
                 num_classes_wrapper_head: int
                ):
        super().__init__()
        self.backbone = backbone
        backbone_output_feat_dim = self.backbone.total_feature_dimension
        self.proj_head = nn.Linear(backbone_output_feat_dim, emb_dim)
        
    def forward(self, skeleton: torch.Tensor, flow: torch.Tensor = None):
        feat_from_backbone, main_logits = self.backbone(skeleton, flow=flow)
        emb = F.normalize(self.proj_head(feat_from_backbone), dim=1)
        return emb, main_logits

if __name__ == "__main__":
    import numpy as np 
    model_args, data_args = ModelArguments(), DataArguments()
    coords = np.load(os.path.join(model_args.pretrained_weight_dir, model_args.stgcn_coords_file_name))
    act_model = ActionRecognitionModel(model_args, data_args, coords)
    act_model.eval()

    # --- begin test for I3D.extract_features output shape ---
    # Create a dummy flow tensor of shape (batch, channels=2, frames, H, W)
    # You can substitute data_args.num_samples and your actual spatial size
    B, C, T = 1, 2, data_args.num_samples
    H, W = 224, 224  # or whatever your flow input size is
    dummy_flow = torch.randn(B, C, T, H, W)

    with torch.no_grad():
        flow_map = act_model.I3D.extract_features(dummy_flow)
    print("I3D.extract_features returned a tensor of shape:", flow_map.shape)
    # --- end test ---

    # If you want to also see the feature‐vector after your pooling:
    pooled = flow_map.mean(dim=2).view(B, -1)
    print("After mean‐pooling over time and flattening:", pooled.shape)
    
    # for dim examination
    print(act_model._skel_feat_dim, act_model._flow_feat_dim, act_model.total_feature_dimension)
    