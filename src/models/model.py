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
from .i3dgcn import I3D_GCN
from .I3D import InceptionI3d
from ..utils.cli_args import ModelArguments, DataArguments, AugmentationArguments

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
        self.final_feature_dim = 0
        self.is_I3D_stream = model_params.is_I3D_stream
        self.is_i3dgcn_stream = model_params.is_i3dgcn_stream
        self.I3D_mode = model_params.I3D_mode 
        
        # augmentation
        self.aug_module = Augmentation(augment_params=aug_params)
        
        # initiate I3D model
        if self.is_I3D_stream: 
            print("use i3d model")
            
            # load the corresponding weight
            if self.I3D_mode == "flow":
                i3d_in_channels = 2
                pretrained_file_name = "flow_imagenet.pt"
            elif self.I3D_mode == "rgb":
                i3d_in_channels = 3
                pretrained_file_name = "rgb_imagenet.pt"
            
            self.I3D = InceptionI3d(in_channels=i3d_in_channels, num_classes=data_params.num_classes)
            
            # load weight 
            if model_params.load_i3d_weights:
                i3d_weights_path = os.path.join(model_params.pretrained_weights_root_dir_name,
                                                    model_params.I3D_weights_dir_name,
                                                    pretrained_file_name)
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
        elif self.is_i3dgcn_stream:
            print("Use I3D_GCN model")
            num_nodes, neighbor_base, num_classes = data_params.num_nodes, data_params.neighbor_base, data_params.num_classes
            self.i3dgcn_model = I3D_GCN(num_nodes, neighbor_base, num_classes, T=data_params.num_samples) 
            self.final_feature_dim = self.i3dgcn_model.head[1].in_features
            
        else: # at least we use skeleton information 
            num_nodes, neighbor_base, num_classes, num_samples = data_params.num_nodes, data_params.neighbor_base, data_params.num_classes, data_params.num_samples
            is_local_flow_stream, is_frame_diff_stream, is_bone_stream = model_params.is_local_flow_stream, model_params.is_frame_diff_stream, model_params.is_bone_stream
            gcn_include_blocks, in_channels, base_channels = model_params.gcn_include_blocks, model_params.in_channels, model_params.base_channels
            
            if model_params.gcn_model_name == "tdgcn":
                self.skel_model = TD_GCN(num_nodes, neighbor_base, num_classes, is_frame_diff_stream, gcn_include_blocks, in_channels = in_channels, base_channels = base_channels)
                self._skel_feat_dim = self.skel_model.fc.in_features
            elif model_params.gcn_model_name == "stgcn":
                self.skel_model = ST_GCN(num_nodes, neighbor_base, num_classes, coords, is_frame_diff_stream, gcn_include_blocks, in_channels = in_channels, base_channels = base_channels)
                self._skel_feat_dim = self.skel_model.fc.in_channels
            elif model_params.gcn_model_name == "infogcn":
                self.skel_model = Info_GCN(num_nodes, neighbor_base, num_classes, is_frame_diff_stream, gcn_include_blocks, in_channels = in_channels, base_channels = base_channels)
                self._skel_feat_dim = self.skel_model.decoder.in_features
            elif model_params.gcn_model_name == "degcn":
                self.skel_model = DE_GCN(num_nodes, neighbor_base, num_classes, num_samples,
                                        is_local_flow_stream, is_frame_diff_stream, is_bone_stream,
                                        gcn_include_blocks, flow_block_size=model_params.flow_block_size, 
                                        in_channels = in_channels, base_channels = base_channels)
                self._skel_feat_dim = self.skel_model.fc[0].in_features
            elif model_params.gcn_model_name == "ctrgcn":
                self.skel_model = CTR_GCN(num_nodes, neighbor_base, num_classes, is_frame_diff_stream, gcn_include_blocks, in_channels = in_channels, base_channels = base_channels)
                self._skel_feat_dim = self.skel_model.fc.in_features
            self.final_feature_dim = self._skel_feat_dim

        # final classifier
        self.final_classifier = nn.Linear(self.final_feature_dim, data_params.num_classes)
    
    def forward(self, skeleton: torch.Tensor, flow: torch.Tensor = None, rgb: torch.Tensor = None):
        
        # conduct augmentation
        skeleton, flow, rgb = self.aug_module(skeleton, flow, rgb)
        
        cos_loss = None        
        if self.is_I3D_stream: # I3D stream
            if self.I3D_mode == "flow":
                I3D_feat, _ = self.I3D(flow)
            else: # rgb
                I3D_feat, _ = self.I3D(rgb)
            feat = self.flow_feature_projector(I3D_feat)
        elif self.is_i3dgcn_stream: # i3dgcn stream
            trial_feat, _, cos_loss = self.i3dgcn_model(skeleton, flow)
            feat = trial_feat
        else: # skeleton stream
            skel_feat, _ = self.skel_model(skeleton, flow)
            feat = skel_feat
        main_task_logits = self.final_classifier(feat)
        
        return feat, main_task_logits, cos_loss

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
    def forward(self, skeleton: torch.Tensor, flow: torch.Tensor = None, rgb: torch.Tensor = None):
        feat_from_backbone, main_logits, cos_loss = self.backbone(skeleton, flow=flow, rgb=rgb)
        emb = F.normalize(self.proj_head(feat_from_backbone), dim=1)
        return emb, main_logits, cos_loss