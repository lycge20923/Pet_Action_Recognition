from collections import defaultdict, Counter
import dataclasses
from datetime import datetime
import json
import numpy as np
import os
import random
import wandb
import yaml
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.models.model import ActionRecognitionModel, ContrastiveActionWrapper
from src.models.loss import ContrastiveLoss
from src.dataset.dataset import KpOfDataset, SiameseKpOfDataset
from src.utils.cli_args import DataArguments, ModelArguments, TrainingArguments, AugmentationArguments
from src.utils.common import load_from_wandb, custom_collate, set_seed

def setup_experiments():
    # set wandb
    run = wandb.init(project="Pet_Action_Recognition")
    config = dict(wandb.config)
    
    # change name 
    exec_name = config.get("exec_name")
    fold_num = config.get("fold_num")
    if exec_name and fold_num:
        wandb.run.name = f"{exec_name}_{str(fold_num)}"
    
    # load parameters and add to wandb
    model_params = load_from_wandb(ModelArguments, config)
    train_params = load_from_wandb(TrainingArguments, config)
    aug_params_train = load_from_wandb(AugmentationArguments, config)
    data_params = load_from_wandb(DataArguments, config)
    aug_params_eval = AugmentationArguments(augment=False)
    
    # saving dir
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    saving_dir = os.path.join(os.path.dirname(__file__), train_params.save_root_dir_name, timestamp)
    
    # save parameters
    complete_config_to_save = {
        "dataset_params": dataclasses.asdict(data_params),
        "augmentation_params_train": dataclasses.asdict(aug_params_train),
        "model_params": dataclasses.asdict(model_params),
        "training_params": dataclasses.asdict(train_params),
        "training_completed_at": timestamp 
    }
    return_params = {"run": run,
                     "parameters":
                         {"data": data_params, 
                          "model":model_params, 
                          "train":train_params, 
                          "aug_train": aug_params_train,
                          "aug_val": aug_params_eval},
                    "saving_dir":saving_dir
                    }
    
    if not train_params.for_test:
        os.makedirs(saving_dir, exist_ok=True)
        try:
            with open(os.path.join(saving_dir, train_params.save_complete_args_name), 'w', encoding='utf-8') as f:
                json.dump(complete_config_to_save, f, indent=4, ensure_ascii=False)
            with open(os.path.join(saving_dir, train_params.save_adjusted_args_name), 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"{e}")
    
    return return_params
    
def prepare_dataloaders(data_params:DataArguments, 
                        aug_params_train:AugmentationArguments, 
                        aug_params_eval:AugmentationArguments, 
                        model_params:ModelArguments, 
                        train_params:TrainingArguments):
   
    load_kps = (not (model_params.is_I3D_stream)) or model_params.is_attgcn_stream
    load_flows = model_params.is_local_flow_stream or model_params.is_I3D_stream or model_params.is_attgcn_stream
    val_dataset = KpOfDataset(data_params, aug_params_eval, load_flows=load_flows, load_kps=load_kps, istrain=False, fold_num=data_params.fold_num, for_test=train_params.for_test)
    train_dataset = KpOfDataset(data_params, aug_params_train, load_flows=load_flows, load_kps=load_kps, istrain=True, fold_num=data_params.fold_num, for_test=train_params.for_test)
    
    # calculate the dataset size
    train_size = len(train_dataset)
    val_size   = len(val_dataset)
    print(f"[Dataset sizes] train: {train_size}, val: {val_size}")

    # for calculating weights for cross entropy loss
    print("Start Calculating ratio of each class")
    with open(os.path.join(data_params.data_dir, data_params.trainsplit_dir_name, "annotation_windows_metadata.json"), 'r') as f:
        annotations = json.load(f)
        
    all_labels = []
    for annotation in annotations:
        label = annotation["action_id"]
        all_labels.append(int(label))
    counts = Counter(all_labels)
    num_classes = len(counts)
    total = sum(counts.values())

    # class weights
    weights = [ total / (num_classes * counts[i]) for i in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # for contrastive learning
    if train_params.add_contrastive_loss:
        train_dataset = SiameseKpOfDataset(train_dataset)
    
    if model_params.gcn_model_name == "stgcn":
        coord_path = os.path.join(model_params.pretrained_weights_root_dir_name, model_params.gcn_weights_dir_name, model_params.stgcn_coords_file_name)
        if not os.path.exists(coord_path):
            sample, _ , _ = val_dataset[0]            # sample.shape = (C, T, V)
            if isinstance(sample, torch.Tensor):
                sample = sample.cpu().numpy()
            sample.mean(axis=1)
            coords = sample.mean(axis=1).T 
            np.save(coord_path, coords)
        else:
            coords = np.load(coord_path)
    else:
        coords = None
    
    print("Start to add to Dataloader")
    # build dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params.batch_size,
        shuffle=True,
        num_workers=train_params.num_workers,
        pin_memory=True if train_params.device == "cuda" else False,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_params.batch_size,
        shuffle=False, 
        num_workers=train_params.num_workers,
        pin_memory=True if train_params.device == "cuda" else False,
        collate_fn=custom_collate
    )
    
    return {"data_loader":{"train":train_loader, "val":val_loader}, "center_coords":coords, "class_weights":class_weights}

def build_model_and_optimizer(model_params:ModelArguments, 
                              data_params:DataArguments, 
                              train_params:TrainingArguments, 
                              aug_params:AugmentationArguments,
                              coords,
                              class_weights=None):
    
    # set loss
    if class_weights is not None:
        class_weights = class_weights.to(train_params.device)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        cross_entropy_loss = torch.nn.CrossEntropyLoss() 
    contrastive_loss = ContrastiveLoss()
    
    # set model
    base_model = ActionRecognitionModel(model_params, data_params, aug_params, coords)
    if not train_params.add_contrastive_loss:
        model = base_model
    else:
        model = ContrastiveActionWrapper(base_model, model_params.multihead_emb_dim)
    model = model.to(train_params.device)
    
    # set optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), train_params.learning_rate, momentum=train_params.momentum, weight_decay=train_params.opt_weight_decay)
    
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params.epochs, eta_min=train_params.scheduler_eta_min)
    
    # load pretrained weights
    if train_params.resume_checkpoint_dir != None:
        print(f"Trying load checkpoint from {train_params.resume_checkpoint_dir}")
        checkpoint_names = [f for f in os.listdir(train_params.resume_checkpoint_dir) if f.endswith(".pth")]
        checkpoint_name = checkpoint_names[0] if len(checkpoint_names) == 1 else "best.pt"
        checkpoint_path = os.path.join(train_params.resume_checkpoint_dir, checkpoint_name)
        
        # load weights
        try:
            ckpt = torch.load(checkpoint_path, map_location=train_params.device)
            state_dict = ckpt["model_state_dict"]
            if not train_params.add_contrastive_loss:   
                info = model.load_state_dict(state_dict, strict=False)
            else:
                info = model.backbone.load_state_dict(state_dict, strict=False)
            print("Loading whole pretrained weights...")
            print("Missing keys:", info.missing_keys)
            print("Unexpected keys:", info.unexpected_keys)
        except:
            print(f"Fails to load pre-trained weights in {checkpoint_path}, try to re-train")
       
        # load optimizer
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except:
            print(f"Fails to load pre-trained optimizer in {checkpoint_path}, try to re-set")
    
    return {"loss":{"cross entropy": cross_entropy_loss, "contrastive learning": contrastive_loss}, "model":model, "optimizer":optimizer, "scheduler":scheduler}

def train_one_epoch(model, 
                    loader, 
                    cross_entropy_loss, 
                    optimizer, 
                    epoch,
                    train_params:TrainingArguments,
                    contrastive_loss=None):
    # initial set
    model.train()
    correct = 0
    sum_loss = 0
    total_individual_samples = 0
    device = train_params.device
    
    for batch_idx, batch_data in enumerate(loader):
        optimizer.zero_grad()
        
        batch_loss_total = None
        
        if train_params.add_contrastive_loss and contrastive_loss: 
            (kp1, kp2), (flow1, flow2), (lab1, lab2), y = batch_data
            kp1, kp2 = kp1.to(device) if kp1 is not None else None, kp2.to(device) if kp2 is not None else None
            flow1, flow2 = flow1.to(device) if flow1 is not None else None, flow2.to(device) if flow2 is not None else None
            lab1, lab2 = lab1.to(device), lab2.to(device)
            y = y.to(device)
            emb1, logit1= model(kp1, flow1)
            emb2, logit2 = model(kp2 ,flow2)
            
            # calculate loss
            loss_ce_part1 = cross_entropy_loss(logit1, lab1)
            loss_ce_part2 = cross_entropy_loss(logit2, lab2)
            batch_loss_ce = loss_ce_part1 + loss_ce_part2 
            
            batch_loss_cl = contrastive_loss(emb1, emb2, y) * train_params.contrastive_loss_coefficient
            wandb.log({"train_batch_loss_cl":batch_loss_cl.item()})
            
            batch_loss_total = batch_loss_ce + batch_loss_cl 
            batch_loss_total.backward()
            
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(
            #             name,
            #             "mean:", f"{param.grad.mean().item():.5f}",
            #             "std:",  f"{param.grad.std().item():.5f}",
            #             "norm:", f"{param.grad.norm().item():.5f}"
            #         )
            
            optimizer.step()
            
            batch_size_pairs = lab1.size(0)
            total_individual_samples += batch_size_pairs * 2
            sum_loss += batch_loss_total.item() * batch_size_pairs

            _, pred1 = torch.max(logit1.data, 1)
            _, pred2 = torch.max(logit2.data, 1)
            correct += (pred1 == lab1).sum().item()
            correct += (pred2 == lab2).sum().item()
        else: 
            kps, flow, label, _ = batch_data
            kps, flow = kps.to(device) if kps is not None else None, flow.to(device) if flow is not None else None
            label = label.to(device)
            
            _, output = model(kps, flow)
            batch_loss_ce = cross_entropy_loss(output, label)
            batch_loss_total = batch_loss_ce
            
            batch_loss_total.backward() 
            optimizer.step()

            current_batch_size = label.size(0)
            total_individual_samples += current_batch_size 
            sum_loss += batch_loss_total.item() * current_batch_size 
            _, predict = torch.max(output.data, 1)
            correct += (predict == label).sum().item()
        
    epoch_loss = sum_loss / len(loader.dataset) 
    epoch_acc = correct / total_individual_samples
    
    wandb.log({"epoch": epoch, "train_loss":epoch_loss, "train_acc": epoch_acc})
    return epoch_loss, epoch_acc

def val_one_epoch(model, 
                  loader, 
                  cross_entropy_loss, 
                  epoch, 
                  actions:list, 
                  train_params:TrainingArguments):
    model.eval()
    test_correct = 0
    test_loss = 0
    total_samples = 0
    device = train_params.device
    
    num_classes = len(actions)
    class_correct = torch.zeros(num_classes).to(device)
    class_total = torch.zeros(num_classes).to(device)
    
    # for precision, recall, f1
    all_trues = []
    all_preds = []
    all_details = defaultdict(lambda: {"preds": [], "ground-trues": []})
    
    with torch.no_grad():
        for kps, flow, label, video_name in loader:
            
            kps, flow = kps.to(device) if kps is not None else None, flow.to(device) if flow is not None else None
            label = label.to(device)
            
            if train_params.add_contrastive_loss:
                _, output = model.backbone(kps, flow)
            else:
                _, output = model(kps, flow)
            loss = cross_entropy_loss(output, label)
            current_batch_size = label.size(0)
            total_samples += current_batch_size
            if not torch.isnan(loss).any():
                test_loss += loss.item() * current_batch_size
            _, predict = torch.max(output.data, 1)
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
    
    epoch_acc = test_correct / len(loader.dataset)
    epoch_loss = test_loss / len(loader.dataset)
    
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
    
    log_dict = {"epoch":epoch, "val_loss": epoch_loss, "val_acc":epoch_acc, "val_prec":float(overall_prec), "val_rec":float(overall_rec), "val_f1":float(overall_f1)}
    for i in range(num_classes):
        action_name = actions[i]
        # log_dict[f"val_acc_class_{action_name}"] = per_class_acc[i].item()
        log_dict[f"val_precision_class_{action_name}"] = float(per_class_prec[i])
        log_dict[f"val_recall_class_{action_name}"] = float(per_class_rec[i])
        log_dict[f"val_f1_class_{action_name}"] = float(per_class_f1[i])
        # log_dict[f"val_samples_class_{actions[i]}"] = class_total[i].item()
    wandb.log(log_dict)
    print(f"Overall   Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
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
    
    return epoch_loss, epoch_acc, cm, all_details

def main():
    # initial set
    settings = setup_experiments()
    run = settings["run"]
    params = settings["parameters"]
    data_params, model_params, train_params, aug_params_train, aug_params_eval = \
        params["data"], params["model"], params["train"], params["aug_train"], params["aug_val"]
    saving_dir = settings["saving_dir"]
    
    # set seed
    set_seed(train_params.seed)
    
    # set data loader
    loaders = prepare_dataloaders(data_params, aug_params_train, aug_params_eval, model_params, train_params)
    train_loader, val_loader = loaders["data_loader"]["train"], loaders["data_loader"]["val"]
    coords = loaders["center_coords"]
    class_weights = loaders["class_weights"]
    
    # set model, loss, optimizer, scheduler
    train_objects = build_model_and_optimizer(model_params, data_params, train_params, aug_params_train, coords, class_weights)
    model = train_objects["model"]
    cross_entropy_loss, contrastive_loss = train_objects["loss"]["cross entropy"], train_objects["loss"]["contrastive learning"]
    optimizer = train_objects["optimizer"]
    scheduler = train_objects["scheduler"]
    
    # for stats
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    best_val_acc = 0.0 
    
    # start training 
    patient_count = 0 # for early stopping
    for epoch in range(1, train_params.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, cross_entropy_loss, optimizer, epoch, train_params, contrastive_loss)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        val_loss, val_acc, cm, all_details = val_one_epoch(model, val_loader, cross_entropy_loss, epoch, actions=data_params.actions, train_params=train_params)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # save for plottting
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        scheduler.step()
        
        #  save
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved ({best_val_acc:.4f} --> {val_acc:.4f}). Saving model...")
            best_val_acc = val_acc
            if train_params.add_contrastive_loss:
                save_model = model.backbone
                print("Info: Saving backbone (ActionRecognitionModel) state_dict from ContrastiveActionWrapper.")
            else:
                save_model = model
                print("Info: Saving ActionRecognitionModel state_dict.")
            
            if not train_params.for_test:
                # for one-run model saving
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'accuracy': val_acc,
                    'model_params': model_params, 
                    'dataset_params': data_params, 
                }, os.path.join(saving_dir, 'best.pth'))
                
                # save cofusion matrix
                np.savetxt(os.path.join(saving_dir, "confusion_matrix.csv"), cm, fmt="%.2f", delimiter=',')
                
                # pre-process and save predict details
                for video_name, details in all_details.items():
                    all_details[video_name]["ground-trues"] = ' '.join(set([data_params.actions[index] for index in all_details[video_name]["ground-trues"]]))
                    all_details[video_name]["preds"] = '|'.join([data_params.actions[index] for index in all_details[video_name]["preds"]])
                
                if train_params.save_predict_details_name:
                    details_path = os.path.join(saving_dir, train_params.save_predict_details_name)
                    with open(details_path, "w", encoding="utf-8") as f:
                        json.dump(all_details, f, ensure_ascii=False, indent=2)

            patient_count = 0
        wandb.log({"best_val_acc": best_val_acc})
        # early stopping 
        if patient_count > train_params.patient_epochs:
            break
        patient_count += 1
        
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    
    run.finish()

        
if __name__ == "__main__":
    main()
