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
        self.late_fusion_use_transformer = model_params.late_fusion_use_transformer
        self.late_fusion_use_mlp = model_params.late_fusion_use_mlp
        self.add_gate_node = model_params.add_gate_node
        self.use_mid_level_fusion = model_params.use_mid_level_fusion
        
        if not self.use_mid_level_fusion:
            # initiate model
            if not self.only_optical_flow: # at least we use skeleton information 
                self.skel_model = ST_GCN(model_params=model_params, data_params=data_params, coords=coords)
                self._skel_feat_dim = self.skel_model.fc.in_channels
            else: # skip skeleton information
                self.skel_model = None
                self._skel_feat_dim = 0
            # feat
            self.I3D = None
            self._flow_feat_dim = 0
            self._projected_flow_feat_dim = 0
            
            if self.use_optical_flow or self.only_optical_flow:
                in_channels = 2 if data_params.skip_kps_for_i3d else 3
                self.I3D = InceptionI3d(in_channels=in_channels)
                
                # load weight 
                if data_params.skip_kps_for_i3d:
                    i3d_weights_path = os.path.join(model_params.pretrained_weight_dir,
                                                    model_params.I3D_weights_dir_name,
                                                    model_params.I3D_weights_file_name)
                    self.I3D.load_state_dict(torch.load(i3d_weights_path))
                self._flow_feat_dim = model_params.I3D_raw_feat_dim
                self._projected_flow_feat_dim = model_params.I3D_project_dim
                self.flow_feature_projector = nn.Linear(self._flow_feat_dim, self._projected_flow_feat_dim)

            # for using both skeleton and optical flow, we should load the pretrained weights of ST-GCN
            if self.use_optical_flow and not self.only_optical_flow:
                st_gcn_weights_path = os.path.join(model_params.pretrained_weight_dir, model_params.stgcn_weights_dir_name, model_params.stgcn_weights_file_name)
                self.skel_model.load_state_dict(torch.load(st_gcn_weights_path))            
            
            # for late fusion using transformer
            if self.late_fusion_use_transformer and self.use_optical_flow and not self.only_optical_flow:
                self.k_skel_token = model_params.late_fusion_transformer_k_skel_token  
                self.k_flow_token = model_params.late_fusion_transformer_k_flow_token  
                self.d_model = model_params.late_fusion_transformer_dim
                self.register_parameter('type_emb_skel', nn.Parameter(torch.zeros(1, 1, self.d_model)))
                self.register_parameter('type_emb_flow', nn.Parameter(torch.zeros(1, 1, self.d_model)))
                self.flow_feature_projector = nn.Linear(self._flow_feat_dim, self.d_model * self.k_flow_token)
                self.skeleton_feature_proj = nn.Linear(self._skel_feat_dim, self.d_model * self.k_skel_token)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model, 
                    nhead=model_params.late_fusion_transformer_heads, 
                    dim_feedforward=self.d_model * 4, 
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_params.late_fusion_transformer_layers)
                self.final_classifier = nn.Linear(self.d_model, model_params.num_classes)
                self.total_feature_dimension = self.d_model
            else:
                # concate feature's size, for final classifier
                self.total_feature_dimension = self._skel_feat_dim
                if (self.use_optical_flow or self.only_optical_flow) and self.I3D is not None:
                    self.total_feature_dimension += self._projected_flow_feat_dim
                
                if not self.late_fusion_use_mlp:
                    self.final_classifier = nn.Linear(self.total_feature_dimension, model_params.num_classes)
                else: # use mlp to be a classifier
                    fused_dim = self.total_feature_dimension 
                    hidden_dim = model_params.late_fusion_mlp_hidden_dim
                    self.fuse_bn = nn.BatchNorm1d(fused_dim)
                    self.fuse_fc1 = nn.Linear(fused_dim, hidden_dim)
                    self.fuse_relu = nn.ReLU(inplace=True)
                    self.fuse_drop = nn.Dropout(p=0.5)
                    self.fuse_fc2 = nn.Linear(hidden_dim, model_params.num_classes)
                    self.fuse_shortcut = nn.Linear(fused_dim, model_params.num_classes)
        else:
            from .interaction import CrossModalInteractionBlock
            from .mid_fusion_backbones import MidFusionI3D, MidFusionSTGCN
            original_stgcn = ST_GCN(model_params=model_params, data_params=data_params, coords=coords)
            original_i3d = InceptionI3d(in_channels=2 if data_params.skip_kps_for_i3d else 3)
            st_gcn_weights_path = os.path.join(model_params.pretrained_weight_dir, model_params.stgcn_weights_dir_name, model_params.stgcn_weights_file_name)     
            original_stgcn.load_state_dict(torch.load(st_gcn_weights_path))
            i3d_weights_path = os.path.join(model_params.pretrained_weight_dir,
                                                    model_params.I3D_weights_dir_name,
                                                    model_params.I3D_weights_file_name)
            original_i3d.load_state_dict(torch.load(i3d_weights_path))
            
            self.skel_model = MidFusionSTGCN(original_stgcn)
            self.i3d_model = MidFusionI3D(original_i3d)
            
            skel_channels_at_fusion = 128
            flow_channels_at_fusion = 832
            
            self.skel_proj = nn.Conv2d(skel_channels_at_fusion, model_params.mid_level_d_model, 1)
            self.flow_proj = nn.Conv3d(flow_channels_at_fusion, model_params.mid_level_d_model, 1)

            self.interaction_block = CrossModalInteractionBlock(
                skel_channels=model_params.mid_level_d_model, 
                flow_channels=model_params.mid_level_d_model
            )
            self.total_feature_dimension = model_params.mid_level_d_model * 2
            self.final_classifier = nn.Linear(model_params.mid_level_d_model * 2, model_params.num_classes)
        
    def forward(self, skeleton: torch.Tensor, flow: torch.Tensor = None):
        
        if not self.use_mid_level_fusion:
            
            # 1. only optical flow
            if self.only_optical_flow:
                flow_map = self.I3D.extract_features(flow)
                flow_feat = flow_map.mean(dim=2).view(flow_map.size(0), -1)
                projected_flow_feat = self.flow_feature_projector(flow_feat)
                combined_feat = projected_flow_feat
                main_task_logits = self.final_classifier(combined_feat)
                return combined_feat, main_task_logits
            
            # 2. at least use skeleton information
            if self.add_gate_node:
                skel_feat, _ = self.skel_model(skeleton, flow)
            else:
                skel_feat, _ = self.skel_model(skeleton)
            combined_feat = skel_feat
            if self.use_optical_flow and self.I3D is not None and flow is not None:
                flow_map = self.I3D.extract_features(flow)
                flow_feat = flow_map.mean(dim=2).view(flow_map.size(0), -1)
                projected_flow_feat = self.flow_feature_projector(flow_feat)
                if self.late_fusion_use_transformer:
                    N = skel_feat.size(0)
                    skel_tokens = self.skeleton_feature_proj(skel_feat)
                    skel_tokens = skel_tokens.view(N, self.k_skel_token, self.d_model)
                    flow_tokens = projected_flow_feat.view(N, self.k_flow_token, self.d_model)
                    
                    skel_tokens = skel_tokens + self.type_emb_skel
                    flow_tokens = flow_tokens + self.type_emb_flow
                    
                    tokens = torch.cat([skel_tokens, flow_tokens], dim=1)
                    
                    fused_seq = self.transformer_encoder(tokens)
                    fused_feat = fused_seq.mean(dim=1)
                    combined_feat = fused_feat
                else:
                    combined_feat = torch.cat([skel_feat, projected_flow_feat], dim=1)
                    
                    # use mlp as a classifier
                    if self.late_fusion_use_mlp:
                        x = self.fuse_bn(combined_feat)
                        x1 = self.fuse_relu(self.fuse_fc1(x))
                        x1 = self.fuse_drop(x1) 
                        logits_mlp = self.fuse_fc2(x1)  
                        logits_sc  = self.fuse_shortcut(combined_feat)
                        main_task_logits = logits_mlp + logits_sc 
                        return combined_feat, main_task_logits

            main_task_logits = self.final_classifier(combined_feat)
            return combined_feat, main_task_logits
        else:
            skel_inter, skel_final_map = self.skel_model(skeleton)
            flow_inter, flow_final_map = self.i3d_model(flow)
            
            skel_inter_proj = self.skel_proj(skel_inter)
            flow_inter_proj = self.flow_proj(flow_inter)
            
            fused_skel_vec, fused_flow_vec = self.interaction_block(skel_inter_proj, flow_inter_proj)
            
            combined_feat = torch.cat([fused_skel_vec, fused_flow_vec], dim=1)
            main_task_logits = self.final_classifier(combined_feat)
            
            return combined_feat, main_task_logits

class ContrastiveActionWrapper(nn.Module):
    def __init__(self, backbone: ActionRecognitionModel, emb_dim:int):
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
    coords = np.load(os.path.join(model_args.pretrained_weight_dir, model_args.stgcn_weights_dir_name, model_args.stgcn_coords_file_name))
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
    