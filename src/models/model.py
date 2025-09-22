import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .stgcn import ST_GCN
from .tdgcn import TD_GCN
from .degcn import DE_GCN
from .ctrgcn import CTR_GCN
from .infogcn import Info_GCN
from .I3D import InceptionI3d
from ..utils.cli_args import ModelArguments, DataArguments, AugmentationArguments
from ..utils.common import saving_self_training_best_gcn_weights_path

from .augment import Augmentation

class ActionRecognitionModel(nn.Module):
    def __init__(self, 
                 model_params:ModelArguments,
                 data_params:DataArguments,
                 aug_params:AugmentationArguments = None,
                 coords = None):
        super(ActionRecognitionModel, self).__init__()
        self.only_I3D_branch = model_params.only_I3D_branch
        self.add_I3D_branch = model_params.add_I3D_branch
        
        # initiate model
        if not self.only_I3D_branch: # at least we use skeleton information 
            if model_params.gcn_model_name == "tdgcn":
                self.skel_model = TD_GCN(model_params=model_params, data_params=data_params)
                self._skel_feat_dim = self.skel_model.fc.in_features
            elif model_params.gcn_model_name == "stgcn":
                self.skel_model = ST_GCN(model_params=model_params, data_params=data_params, coords=coords)
                self._skel_feat_dim = self.skel_model.fc.in_channels
            elif model_params.gcn_model_name == "infogcn":
                self.skel_model = Info_GCN(model_params=model_params, data_params=data_params)
                self._skel_feat_dim = self.skel_model.decoder.in_features
            elif model_params.gcn_model_name == "degcn":
                self.skel_model = DE_GCN(model_params=model_params, data_params=data_params)
                self._skel_feat_dim = self.skel_model.fc[0].in_features
            elif model_params.gcn_model_name == "ctrgcn":
                self.skel_model = CTR_GCN(model_params=model_params, data_params=data_params)
                self._skel_feat_dim = self.skel_model.fc.in_features
                
        else: # skip skeleton information
            self.skel_model = None
            self._skel_feat_dim = 0
        # feat
        self.I3D = None
        self._flow_feat_dim = 0
        self._projected_flow_feat_dim = 0
        
        if self.add_I3D_branch or self.only_I3D_branch:
            self.I3D = InceptionI3d(in_channels=2)
            
            # load weight 
            i3d_weights_path = os.path.join(model_params.pretrained_weights_root_dir_name,
                                                model_params.I3D_weights_dir_name,
                                                model_params.I3D_weights_file_name)
            self.I3D.load_state_dict(torch.load(i3d_weights_path))
            self._flow_feat_dim = model_params.I3D_raw_feat_dim
            self._projected_flow_feat_dim = model_params.I3D_project_dim
            self.flow_feature_projector = nn.Linear(self._flow_feat_dim, self._projected_flow_feat_dim)

        # for using both skeleton and optical flow, we should load the pretrained weights of GCN
        if (model_params.load_gcn_weights or self.add_I3D_branch) and not self.only_I3D_branch:
            gcn_weights_path = saving_self_training_best_gcn_weights_path(data_params, model_params)
            ckpt = torch.load(gcn_weights_path)
            info = self.load_state_dict(ckpt["model_state_dict"], strict=False)            
            print("Missing keys:", info.missing_keys)
            print("Unexpected keys:", info.unexpected_keys)
        
        # concate feature's size, for final classifier
        self.total_feature_dimension = self._skel_feat_dim
        if (self.add_I3D_branch or self.only_I3D_branch) and self.I3D is not None:
            self.total_feature_dimension += self._projected_flow_feat_dim
        self.final_classifier = nn.Linear(self.total_feature_dimension, model_params.num_classes)
        
        self.aug_module = Augmentation(augment_params=aug_params)
    
    def forward(self, skeleton: torch.Tensor, flow: torch.Tensor = None):
        
        skeleton, flow = self.aug_module(skeleton, flow)
        # 1. only optical flow
        if self.only_I3D_branch:
            flow_map = self.I3D.extract_features(flow)
            flow_feat = flow_map.mean(dim=2).view(flow_map.size(0), -1)
            projected_flow_feat = self.flow_feature_projector(flow_feat)
            combined_feat = projected_flow_feat
            main_task_logits = self.final_classifier(combined_feat)
            return combined_feat, main_task_logits
        
        # 2. at least use skeleton information
        skel_feat, _ = self.skel_model(skeleton, flow)
        combined_feat = skel_feat
        if self.add_I3D_branch and self.I3D is not None and flow is not None:
            flow_map = self.I3D.extract_features(flow)
            flow_feat = flow_map.mean(dim=2).view(flow_map.size(0), -1)
            projected_flow_feat = self.flow_feature_projector(flow_feat)
            combined_feat = torch.cat([skel_feat, projected_flow_feat], dim=1)

        main_task_logits = self.final_classifier(combined_feat)
        return combined_feat, main_task_logits

class ContrastiveActionWrapper(nn.Module):
    def __init__(self, backbone: ActionRecognitionModel, emb_dim:int):
        super().__init__()
        self.backbone = backbone
        backbone_output_feat_dim = self.backbone.total_feature_dimension
        self.proj_head = nn.Linear(backbone_output_feat_dim, emb_dim)
    
    '''
    def forward(self, skeleton: torch.Tensor, flow: torch.Tensor = None, labels=None):
        feat_from_backbone, main_logits = self.backbone(skeleton, flow=flow, labels=labels)
    '''
    def forward(self, skeleton: torch.Tensor, flow: torch.Tensor = None):
        feat_from_backbone, main_logits = self.backbone(skeleton, flow=flow)
        emb = F.normalize(self.proj_head(feat_from_backbone), dim=1)
        return emb, main_logits

if __name__ == "__main__":
    import numpy as np 
    model_params, data_params = ModelArguments(), DataArguments()
    coords = np.load(os.path.join(model_params.pretrained_weights_root_dir_name, model_params.gcn_weights_dir_name, model_params.stgcn_coords_file_name))
    act_model = ActionRecognitionModel(model_params, data_params, coords=coords)
    act_model.eval()

    # --- begin test for I3D.extract_features output shape ---
    # Create a dummy flow tensor of shape (batch, channels=2, frames, H, W)
    # You can substitute data_args.num_samples and your actual spatial size
    B, C, T = 1, 2, data_params.num_samples
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
    