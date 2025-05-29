import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .stgcn import ST_GCN
from .I3D import InceptionI3d
from ..utils.cli_args import TrainingArguments, ModelArguments, DataArguments

class ActionRecognitionModel(nn.Module):
    def __init__(self, 
                 train_params:TrainingArguments,
                 model_params:ModelArguments,
                 data_params:DataArguments,
                 coords):
        super().__init__()
        self.use_optical_flow = model_params.add_optical_flow
        
        # initiate model
        self.skel_model = ST_GCN(params=model_params, data_params=data_params, coords=coords)
        self.I3D = None
        
        # feat
        self._skel_feat_dim = self.skel_model.fc.in_channels
        self._flow_feat_dim = 0
        if self.use_optical_flow:
            self.I3D = InceptionI3d(in_channels=2)
            
            # load weight 
            i3d_weights_path = os.path.join(model_params.pretrained_weight_dir,
                                                model_params.I3D_weights_dir_name,
                                                model_params.I3D_weights_file_name)
            self.I3D.load_state_dict(torch.load(i3d_weights_path))
            self._flow_feat_dim = 4096
        
        # concate feature's size, for final classifier
        self.total_feature_dimension = self._skel_feat_dim
        if self.use_optical_flow and self.I3D is not None:
            self.total_feature_dimension += self._flow_feat_dim
        self.final_classifier = nn.Linear(self.total_feature_dimension, model_params.num_classes)
    
        
    def forward(self, skeleton: torch.Tensor, flow: torch.Tensor = None):
        skel_feat, _ = self.skel_model(skeleton)
        combined_feat = skel_feat
        if self.I3D is not None and self.use_optical_flow:
            if flow is not None:
                flow_map = self.I3D.extract_features(flow)
                flow_feat = flow_map.mean(dim=2).view(flow_map.size(0), -1)
                combined_feat = torch.cat([skel_feat, flow_feat], dim=1)

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
        self.class_head = nn.Linear(backbone_output_feat_dim, num_classes_wrapper_head)
        
    def forward(self, skeleton: torch.Tensor, flow: torch.Tensor = None):
        feat_from_backbone, _ = self.backbone(skeleton, flow=flow)
        emb = F.normalize(self.proj_head(feat_from_backbone), dim=1)
        wrapper_logits = self.class_head(feat_from_backbone)
        return emb, wrapper_logits