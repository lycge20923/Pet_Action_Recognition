import torch.nn as nn
from pytorchvideo.models.hub import x3d_m

class x3d(nn.Module):
    """
    Wrapper for PyTorchVideo X3D-M that returns both penultimate features and logits.
    Args:
        load_pretrained (bool): load Kinetics-pretrained weights if True.
        classes (int): number of output classes.
    """
    def __init__(self, load_pretrained: bool = True, **kwargs):
        super().__init__()
        num_classes = kwargs.get("classes", 11)

        # 1) base model
        self.base = x3d_m(pretrained=load_pretrained)

        # 2) ensure raw logits (no extra activation at head)
        if hasattr(self.base.blocks[-1], "activation"):
            self.base.blocks[-1].activation = nn.Identity()

        # 3) replace classifier to match num_classes
        in_dim = self.base.blocks[-1].proj.in_features  # penultimate channel dim
        self.base.blocks[-1].proj = nn.Linear(in_dim, num_classes)

        # 4) hook: capture proj input and convert to (B, in_dim)
        self._feat = None

        def to_penult_feat(z, in_features):
            # Return (B, in_features), preserving gradients.
            if z.dim() == 2 and z.shape[1] == in_features:
                return z  # already (B, C)
            if z.dim() == 5:
                # Try common layouts
                if z.shape[1] == in_features:          # (B, C, T, H, W)
                    return z.mean(dim=(2, 3, 4))       # -> (B, C)
                if z.shape[-1] == in_features:         # (B, T, H, W, C)
                    return z.mean(dim=(1, 2, 3))       # -> (B, C)
                # Fallback: find axis equal to in_features and average others
                for ax in range(1, z.dim()):
                    if z.shape[ax] == in_features:
                        # move that axis to channel, average the rest T/H/W
                        order = [0, ax] + [i for i in range(1, 5) if i != ax]
                        z = z.permute(*order)          # (B, C, ?, ?, ?)
                        return z.mean(dim=(2, 3, 4))
                # Ultimate fallback: flatten time/space, then reduce by mean to C
                # (rarely used; keeps training stable)
                b = z.size(0)
                return z.view(b, -1, in_features).mean(dim=1)  # -> (B, C)
            # Last resort: flatten to (B, -1); not ideal but safe
            return z.view(z.size(0), -1)

        def grab_penult(mod, inputs):
            x = inputs[0]                   # input to Linear/Conv head
            self._feat = to_penult_feat(x, in_dim)

        self._handle = self.base.blocks[-1].proj.register_forward_pre_hook(grab_penult)

        # Expose dims
        self.feat_dim = in_dim
        self.num_classes = num_classes

    def forward(self, x):
        logits = self.base(x)   # triggers hook; self._feat is set
        feat = self._feat
        self._feat = None
        return feat, logits

    def remove_hooks(self):
        if hasattr(self, "_handle") and self._handle is not None:
            self._handle.remove()
            self._handle = None