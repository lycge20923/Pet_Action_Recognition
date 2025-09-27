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
        assert sum([model_params.is_local_flow_stream, model_params.is_frame_diff_stream, model_params.is_I3D_stream]) < 2,\
            "Train one stream at one time"
        self.is_I3D_stream = model_params.is_I3D_stream
        self.final_feature_dim = 0
        
        # augmentation
        self.aug_module = Augmentation(augment_params=aug_params)
        
        # initiate skeleton model
        self.skel_model = None
        if not self.is_I3D_stream: # at least we use skeleton information 
            print("use skele model")
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
            self.final_feature_dim = self._skel_feat_dim
            
        # initiate I3D
        else:
            print("use i3d model")
            self.I3D = InceptionI3d(in_channels=2, num_classes=model_params.num_classes)
            
            # load weight 
            i3d_weights_path = os.path.join(model_params.pretrained_weights_root_dir_name,
                                                model_params.I3D_weights_dir_name,
                                                model_params.I3D_weights_file_name)
            i3d_state = torch.load(i3d_weights_path)
            filtered = {}
            for k, v in i3d_state.items():
                if k.startswith("logits.") or "logits." in k:
                    print(k)
                    continue
                filtered[k] = v
            info = self.I3D.load_state_dict(filtered, strict=False)
            print("Loading I3D pretrained weights...")
            print("Missing keys:", info.missing_keys)
            print("Unexpected keys:", info.unexpected_keys)
            
            self._projected_flow_feat_dim = model_params.I3D_project_dim
            self.flow_feature_projector = nn.Linear(model_params.I3D_raw_feat_dim, self._projected_flow_feat_dim)
            self.final_feature_dim = self._projected_flow_feat_dim
        
        # final classifier
        self.final_classifier = nn.Linear(self.final_feature_dim, model_params.num_classes)
    
    def forward(self, skeleton: torch.Tensor, flow: torch.Tensor = None):
        
        # conduct augmentation
        skeleton, flow = self.aug_module(skeleton, flow)
        
        # I3D stream
        if self.is_I3D_stream:
            I3D_feat, _ = self.I3D(flow)
            feat = self.flow_feature_projector(I3D_feat)
        
        # skeleton stream
        else:
            skel_feat, _ = self.skel_model(skeleton, flow)
            feat = skel_feat
            
        main_task_logits = self.final_classifier(feat)
        return feat, main_task_logits

class ContrastiveActionWrapper(nn.Module):
    def __init__(self, backbone: ActionRecognitionModel, emb_dim:int):
        super().__init__()
        self.backbone = backbone
        backbone_output_feat_dim = self.backbone.final_feature_dim
        self.proj_head = nn.Linear(backbone_output_feat_dim, emb_dim)
    
    '''
    def forward(self, skeleton: torch.Tensor, flow: torch.Tensor = None, labels=None):
        feat_from_backbone, main_logits = self.backbone(skeleton, flow=flow, labels=labels)
    '''
    def forward(self, skeleton: torch.Tensor, flow: torch.Tensor = None):
        feat_from_backbone, main_logits = self.backbone(skeleton, flow=flow)
        emb = F.normalize(self.proj_head(feat_from_backbone), dim=1)
        return emb, main_logits