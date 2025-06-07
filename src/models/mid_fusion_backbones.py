import torch
import torch.nn as nn
from .I3D import InceptionI3d
from .stgcn import ST_GCN # 確保您可以從正確的路徑導入您的 ST_GCN

class MidFusionI3D(nn.Module):
    def __init__(self, original_i3d_model:InceptionI3d):
        super().__init__()
        # 直接持有原始模型的引用，而不是複製每一層
        self.backbone = original_i3d_model
        self.fusion_endpoint = 'Mixed_4f'

    def forward(self, x):
        intermediate_feature = None
        
        for end_point in self.backbone.VALID_ENDPOINTS:
            if end_point in self.backbone._modules:
                x = self.backbone._modules[end_point](x)
                if end_point == self.fusion_endpoint:
                    intermediate_feature = x # 攔截 'Mixed_4f' 的輸出
            # 如果到達 logits 層定義之前的最後一個 inception block，就停止
            if end_point == 'Mixed_5c': 
                break # 我們不需要執行後續的 avg_pool 和 logits

        # 返回中間層特徵，以及最後的特徵圖
        return intermediate_feature, x

class MidFusionSTGCN(nn.Module):
    """
    針對您提供的 ST_GCN 程式碼客製化的包裝器
    """
    def __init__(self, original_stgcn_model: ST_GCN):
        super().__init__()
        self.backbone = original_stgcn_model
        # 我們的目標是在第3個block後攔截 (索引為2)
        self.fusion_block_idx = 2 

    def forward(self, x: torch.Tensor, x_flow: torch.Tensor = None):
        """
        這個 forward 方法完整複製了您原始 ST_GCN 的執行流程，
        並在指定位置返回中間特徵。
        
        Args:
            x (torch.Tensor): 骨架點輸入, shape [N, C, T, V]
            x_flow (torch.Tensor, optional): 光流輸入，用於 node_gate. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - intermediate_feature: 第3個block後的特徵圖 [N, C_final, T_inter, V]
                - final_feature_map: 第4個block後的特徵圖 [N, C_final, T_final, V]
        """
        # --- 複製 ST_GCN.forward 的流程 ---

        N, C, T, V = x.size()
        # 1. 保存原始坐標，用於 node_gate
        orig_coords = x.clone() 

        # 2. 處理可學習節點 (如果啟用)
        if self.backbone.add_learnable_node:
            V = V + 1
            glb = self.backbone.global_token.unsqueeze(0).repeat(N, 1)
            glb = glb.unsqueeze(-1).repeat(1, 1, T)
            glb = glb.unsqueeze(-1)
            x = torch.cat([x, glb], dim=-1)
        
        # 3. 初始的 BatchNorm
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.backbone.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # 4. 依次執行 STGC blocks，並在指定位置攔截
        intermediate_feature = None

        # Block 1
        x = self.backbone.stgc1(x, orig_coords, x_flow, self.backbone.A)
        # Block 2
        x = self.backbone.stgc2(x, orig_coords, x_flow, self.backbone.A)
        # Block 3
        x = self.backbone.stgc3(x, orig_coords, x_flow, self.backbone.A)
        
        # <<< 在這裡攔截第3個 block 的輸出 >>>
        intermediate_feature = x

        # Block 4
        x = self.backbone.stgc4(x, orig_coords, x_flow, self.backbone.A)
        final_feature_map = x
        
        # --- 流程結束 ---
        # 我們不執行後續的池化和分類，只返回特徵圖
        # 讓主模型 ActionRecognitionModel 去決定如何處理這些特徵圖
        
        return intermediate_feature, final_feature_map
# ========================================================================================
# <<< 新版本 MidFusionSTGCN 結束 >>>
# ========================================================================================