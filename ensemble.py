import argparse
import logging
import numpy as np
import os
import torch
import yaml
from collections import defaultdict
import json
from datetime import datetime
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from src.utils.logging_utils import setup_logger
from src.utils.cli_args import DataArguments, ModelArguments, AugmentationArguments, TrainingArguments, ResultsArguments
from src.utils.common import load_from_wandb, custom_collate, set_seed
from src.dataset.dataset import KpOfDataset
from src.models.model import ActionRecognitionModel

def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble for Action Recognition(include 5-fold validation)')
    
    # for 5-fold validation
    parser.add_argument("--five_fold_val", action="store_true", help="If not 5-fold validation, it would be needed to input the directory path.\
        o.w., You only have to check whether `models` have 5-fold satisfied weight files for validation, and only need to input the stream name")
    parser.add_argument("--joints_stream", action="store_true", help="Use joints stream")
    parser.add_argument("--local_flow_stream", action="store_true", help="Use local flow stream")
    parser.add_argument("--diff_stream", action="store_true", help="Use diff stream")
    parser.add_argument('--attgcn_stream', action="store_true", help="Use attgcn stream")
    parser.add_argument("--I3D_stream", action="store_true", help="Use I3D stream")
    parser.add_argument("--joints_gcn_name", default="degcn", choices=["ctrgcn", "infogcn", "stgcn", "tdgcn", "degcn"], help="(ablation study) choose different gcn name if needed")
    
    # for individual
    parser.add_argument('--joints_stream_checkpoint_dir', default=None, help="Skeleton model(joints) checkpoint dir")
    parser.add_argument("--local_flow_stream_checkpoint_dir", default=None, help="Skeleton model(flow) checkpoint dir")
    parser.add_argument("--diff_stream_checkpoint_dir", default=None, help="Skeleton model(diff) checkpoint dir")
    parser.add_argument('--attgcn_stream_checkpoint_dir', default=None, help="Attgcn checkpoint dir")
    parser.add_argument('--I3D_checkpoint_dir', default=None, help="I3D model checkpoint_dir")
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

def set_models(model_dirs:list):
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
    return models, fold_num

def set_dataloaders(joints, local_flow, diff, attgcn, I3D, fold_num):
    load_kps = joints or local_flow or diff or attgcn
    load_flows = local_flow or attgcn or I3D
    aug_params_eval = AugmentationArguments(augment=False)
    val_dataset = KpOfDataset(data_params, aug_params_eval, load_flows=load_flows, load_kps=load_kps, istrain=False, fold_num=fold_num)
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_params.batch_size,
        shuffle=False, 
        num_workers=train_params.num_workers,
        pin_memory=True if train_params.device == "cuda" else False,
        collate_fn=custom_collate
    )
    return val_loader

def main():
    # args
    args = parse_args()
    
    # load parameters
    global logger, train_params, data_params, device, actions, num_classes
    logger = setup_logger(file_path=__file__,level=logging.INFO)
    
    train_params = TrainingArguments()
    device = train_params.device
    
    output_params = ResultsArguments()
    
    data_params = DataArguments()
    actions = data_params.actions
    num_classes = len(actions)
    
    # set seed 
    set_seed(train_params.seed)
        
    if args.five_fold_val:
        model_params = ModelArguments()
        model_dir, st_dir = model_params.pretrained_weights_root_dir_name, model_params.self_training_weights_dir_name
        subroot_dir = os.path.join(model_dir, st_dir)
        fold_ids = [i for i in range(data_params.num_folds)]
        save_subdir = []
        if args.joints_stream:
            if args.joints_gcn_name == "degcn":
                save_subdir.append("joints")
            else:
                save_subdir.append(os.path.join("supplement", f"joints_{args.joints_gcn_name}"))
        if args.diff_stream:
            if args.joints_gcn_name == "degcn":
                save_subdir.append(os.path.join("supplement", "diff"))
            else:
                save_subdir.append(os.path.join("supplement", f"diff_{args.joints_gcn_name}"))
        if args.local_flow_stream:
            save_subdir.append("local_flow")
        if args.attgcn_stream:
            save_subdir.append("attgcn")
        if args.I3D_stream:
            save_subdir.append("I3D")
        
        all_models = []
        for fold_id in fold_ids:
            model_dirs = [os.path.join(subroot_dir, str(fold_id), sub_dir) for sub_dir in save_subdir]
            models, fold_num = set_models(model_dirs)
            all_models.append(models)
            
        # set output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_params.output_dir, f"ensemble_{timestamp}_all")
        os.makedirs(output_dir, exist_ok=True)
        
    else:
        # set models
        model_dirs = [args.joints_stream_checkpoint_dir, args.local_flow_stream_checkpoint_dir, args.diff_stream_checkpoint_dir, args.I3D_checkpoint_dir, args.attgcn_stream_checkpoint_dir]
        models, fold_num = set_models(model_dirs)
        
        fold_ids = [fold_num]
        
        # load dataset
        all_models = [models]
        
        # set output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_params.output_dir, f"ensemble_{timestamp}_{fold_num}")
        os.makedirs(output_dir, exist_ok=True)
        
    
    # start eval
    test_correct = 0
    total_samples = 0
    
    class_correct = torch.zeros(num_classes).to(device)
    class_total = torch.zeros(num_classes).to(device)
    
    all_trues = []
    all_preds = []
    all_details = defaultdict(lambda: {"preds": [], "ground-trues": []})
    
    count_samples = 0
    for models, fold_id in zip(all_models, fold_ids):
        
        # load data loader
        if args.five_fold_val:
            val_loader = set_dataloaders(args.joints_stream, args.local_flow_stream, args.diff_stream, args.attgcn_stream, args.I3D_stream, fold_id)
        else:
            val_loader = set_dataloaders(args.joints_stream_checkpoint_dir, args.local_flow_stream_checkpoint_dir, args.diff_stream_checkpoint_dir, args.attgcn_stream_checkpoint_dir, args.I3D_checkpoint_dir, fold_id)
        
        count_samples += len(val_loader.dataset)
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
        del val_loader
        
    cm = confusion_matrix(all_trues, all_preds, labels=list(range(num_classes))) # confusion matrix
    cm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    # draw cm
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Average Value', rotation=-90, va="bottom")
    ax.set_title('Average Confusion Matrix (5-fold)')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    tick_marks = np.arange(len(actions))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(actions, rotation=45, ha="right")
    ax.set_yticklabels(actions)
    fmt = ".2f" 
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    save_fig_path = os.path.join(output_dir, "confusion_matrix_plot.png")
    plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    logger.info(f"Confusion matrix image saved to: {save_fig_path}")
    
    epoch_acc = test_correct / count_samples
    
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
        all_details[video_name]["ground-trues"] = ' '.join(set([actions[index] for index in all_details[video_name]["ground-trues"]]))
        all_details[video_name]["preds"] = '|'.join([actions[index] for index in all_details[video_name]["preds"]])
    
    if train_params.save_predict_details_name:
        details_path = os.path.join(output_dir, train_params.save_predict_details_name)
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(all_details, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()