import torch.nn as nn
from .stgcn_model import ST_GCN, STGCN_MultiHead
from ..utils.cli_args import TrainingArguments, ModelArguments, DataArguments

class ActionRecognitionModel(nn.Module):
    def __init__(self, 
                 train_params:TrainingArguments,
                 model_params:ModelArguments,
                 data_params:DataArguments,
                 coords):
        super().__init__()
        if not train_params.add_contrastive_loss:
            self.model = ST_GCN(params=model_params, data_params=data_params, coords=coords).to(train_params.device)
        else:
            self.model = STGCN_MultiHead(params=model_params, data_params=data_params, coords=coords).to(train_params.device)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    