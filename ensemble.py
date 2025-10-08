import argparse
import logging
import numpy as np
import os
import torch
import yaml
from collections import defaultdict
import json
import time
from datetime import datetime

from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from src.utils.logging_utils import setup_logger
from src.utils.cli_args import DataArguments, ModelArguments, AugmentationArguments, TrainingArguments, ResultsArguments
from src.utils.common import load_from_wandb, custom_collate, set_seed
from src.dataset.dataset import KpOfDataset
from src.models.model import ActionRecognitionModel

def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble for Action Recognition')
    parser.add_argument('--joints_stream_checkpoint_dir', default=None, help="Skeleton model(joints) checkpoint dir")
    parser.add_argument("--local_flow_stream_checkpoint_dir", default=None, help="Skeleton model(flow) checkpoint dir")
    parser.add_argument("--diff_stream_checkpoint_dir", default=None, help="Skeleton model(diff) checkpoint dir")
    parser.add_argument('--I3D_checkpoint_dir', default=None, help="I3D model checkpoint_dir")
    parser.add_argument('--attgcn_stream', default=None)
    return parser.parse_args()

def load_weights(dir_name:str):
    
    data_params = DataArguments()
    save_adjusted_args_name = train_params.save_adjusted_args_name
    adjusted_params_path = os.path.join(dir_name, save_adjusted_args_name)
    with open(adjusted_params_path, 'r') as f:
        adjusted_params = yaml.safe_load(f)
    
    model_params = load_from_wandb(ModelArguments, adjusted_params)
    
    fold_num = load_from_wandb(DataArguments, adjusted_params).fold_num
    checkpoint_names = [f for f in os.listdir(dir_name) if f.endswith(".pth")]
    checkpoint_name = checkpoint_names[0] if len(checkpoint_names) == 1 else "best.pth"
    checkpoint_path = os.path.join(dir_name, checkpoint_name)
    coords = None
    if model_params.gcn_model_name == "stgcn":
        coord_path = os.path.join(model_params.pretrained_weights_root_dir_name, model_params.gcn_weights_dir_name, model_params.stgcn_coords_file_name)
        coords = np.load(coord_path)
    model = ActionRecognitionModel(model_params, data_params, coords=coords).to(device)
    ckpt = torch.load(checkpoint_path, map_location=train_params.device)
    info = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    logger.info(f"Loading model from '{dir_name}' weights...")
    logger.info(f"Missing keys: {info.missing_keys}")
    logger.info(f"Unexpected keys:{info.unexpected_keys}")
    
    logger.info(f"Loading whole pretrained weights from {checkpoint_path}")
    return model, fold_num

def main():
    # args
    args = parse_args()
    
    # load parameters
    global logger, train_params, data_params, device
    logger = setup_logger(file_path=__file__,level=logging.INFO)
    train_params = TrainingArguments()
    aug_params_eval = AugmentationArguments(augment=False)
    output_params = ResultsArguments()
    data_params = DataArguments()
    device = train_params.device
    
    # set seed 
    set_seed(train_params.seed)
    
    # load model
    model_dirs = [args.joints_stream_checkpoint_dir, args.local_flow_stream_checkpoint_dir, args.diff_stream_checkpoint_dir, args.I3D_checkpoint_dir, args.attgcn_stream]
    models, fold_nums = [], []
    for model_dir in model_dirs:
        if model_dir is not None:
            model, fold_num = load_weights(dir_name=model_dir)
            models.append(model)
            fold_nums.append(fold_num)
    if len(set(fold_nums)) == 1:
        fold_num = fold_nums[0]
    else:
        raise ValueError('The fold number of all model should be the same')
    
    # load dataset
    load_kps = args.joints_stream_checkpoint_dir or args.local_flow_stream_checkpoint_dir or args.diff_stream_checkpoint_dir or args.attgcn_stream
    load_flows = args.local_flow_stream_checkpoint_dir or args.I3D_checkpoint_dir or args.attgcn_stream
    val_dataset = KpOfDataset(data_params, aug_params_eval, load_flows=load_flows, load_kps=load_kps, istrain=False, fold_num=fold_num)
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_params.batch_size,
        shuffle=False, 
        num_workers=train_params.num_workers,
        pin_memory=True if train_params.device == "cuda" else False,
        collate_fn=custom_collate
    )
    
    # set output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_params.output_dir, f"ensemble_{timestamp}_{fold_num}")
    os.makedirs(output_dir, exist_ok=True)
    
    # start eval
    test_correct = 0
    total_samples = 0
    num_classes = len(data_params.actions)
    class_correct = torch.zeros(num_classes).to(device)
    class_total = torch.zeros(num_classes).to(device)
    
    all_trues = []
    all_preds = []
    all_details = defaultdict(lambda: {"preds": [], "ground-trues": []})
    
    with torch.no_grad():
        for kps, flow, label, video_name in val_loader:
            
            kps, flow = kps.to(device) if kps is not None else None, flow.to(device) if flow is not None else None
            label = label.to(device)
            
            total_logits = 0
            for model in models:
                model.eval()
                _, logits = model(kps, flow)
                total_logits += logits
                
            current_batch_size = label.size(0)
            total_samples += current_batch_size
            _, predict = torch.max(total_logits.data, 1)
            
            test_correct += (predict == label).sum().item()
            
            # add batch label & predict to list
            all_trues.extend(label.cpu().numpy())
            all_preds.extend(predict.cpu().numpy())
            
            # collect details
            for i in range(current_batch_size):
                vn = video_name[i]
                gt = int(label[i].item())
                pd = int(predict[i].item())
                all_details[vn]["ground-trues"].append(gt)
                all_details[vn]["preds"].append(pd)
            
            # calculate class-wise accuracy
            corrects = (predict == label)
            for i in range(current_batch_size):
                true_label = label[i].item()       
                class_total[true_label] += 1       
                if corrects[i]:                    
                    class_correct[true_label] += 1 

    cm = confusion_matrix(all_trues, all_preds, labels=list(range(num_classes))) # confusion matrix
    cm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
    
    epoch_acc = test_correct / len(val_loader.dataset)
    
    # for precision, recall, f1
    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        all_trues,
        all_preds,
        labels=list(range(num_classes)),
        zero_division=0
    )
    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(
        all_trues,
        all_preds,
        average='macro',
        zero_division=0
    )
    
    actions=data_params.actions
    log_dict = {}
    for i in range(num_classes):
        action_name = actions[i]
        log_dict[f"val_precision_class_{action_name}"] = float(per_class_prec[i])
        log_dict[f"val_recall_class_{action_name}"] = float(per_class_rec[i])
        log_dict[f"val_f1_class_{action_name}"] = float(per_class_f1[i])
        
    print(f"Overall Acc: {epoch_acc:.4f}, "
          f"Prec: {overall_prec:.4f}, Rec: {overall_rec:.4f}, F1: {overall_f1:.4f}")
    print("Per-class:")
    for i in range(num_classes):
        action_name = actions[i]
        # acc_i = per_class_acc[i].item()
        prec_i = per_class_prec[i]
        rec_i = per_class_rec[i]
        f1_i = per_class_f1[i]
        cor_i = class_correct[i].item()
        tot_i = class_total[i].item()
        # print(f"  Class {actions[i]}, Acc: {acc_i:.4f} ({cor_i}/{tot_i}), P {prec_i:.4f}, R {rec_i:.4f}, F1 {f1_i:.4f}")
        print(f"  Class {actions[i]}, P {prec_i:.4f}, R {rec_i:.4f}({cor_i}/{tot_i}), F1 {f1_i:.4f}")
    
    np.savetxt(os.path.join(output_dir, "confusion_matrix.csv"), cm, fmt="%.2f", delimiter=',')
    
    # pre-process and save predict details
    for video_name, details in all_details.items():
        all_details[video_name]["ground-trues"] = ' '.join(set([data_params.actions[index] for index in all_details[video_name]["ground-trues"]]))
        all_details[video_name]["preds"] = '|'.join([data_params.actions[index] for index in all_details[video_name]["preds"]])
    
    if train_params.save_predict_details_name:
        details_path = os.path.join(output_dir, train_params.save_predict_details_name)
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(all_details, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()