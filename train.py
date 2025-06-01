from collections import Counter
import dataclasses
from datetime import datetime
import json
import numpy as np
import os
import random
import wandb
import yaml
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from src.models.model import ActionRecognitionModel, ContrastiveActionWrapper
from src.models.loss import ContrastiveLoss
from src.dataset.dataset import KpOfDataset, SiameseKpOfDataset
from src.utils.cli_args import DataArguments, ModelArguments, TrainingArguments, AugmentationArguments
from src.utils.common import load_from_wandb

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
    aug_params_eval = AugmentationArguments(augment=False)
    data_params = DataArguments()
    
    # saving dir
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    saving_dir = os.path.join(os.path.dirname(__file__), train_params.save_dir_name, timestamp)
    os.makedirs(saving_dir, exist_ok=True)
    
    # save parameters
    complete_config_to_save = {
        "dataset_params": dataclasses.asdict(data_params),
        "augmentation_params_train": dataclasses.asdict(aug_params_train),
        "model_params": dataclasses.asdict(model_params),
        "training_params": dataclasses.asdict(train_params),
        "training_completed_at": timestamp 
    }
    
    try:
        with open(os.path.join(saving_dir, train_params.save_complete_args_name), 'w', encoding='utf-8') as f:
            json.dump(complete_config_to_save, f, indent=4, ensure_ascii=False)
        with open(os.path.join(saving_dir, train_params.save_adjusted_args_name), 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        print(f"{e}")
    
    return {"run": run, 
            "parameters":
                {"data": data_params, 
                 "model":model_params, 
                 "train":train_params, 
                 "aug_train": aug_params_train,
                 "aug_val": aug_params_eval},
            "saving_dir":saving_dir
            }

def custom_collate(batch):
    first_item = batch[0]
    is_siamese = (len(first_item) == 4 and 
                  isinstance(first_item[0], tuple) and len(first_item[0]) == 2 and # (kp1,kp2)
                  isinstance(first_item[1], tuple) and len(first_item[1]) == 2 and # (flow1,flow2)
                  isinstance(first_item[2], tuple) and len(first_item[2]) == 2)   # (lab1,lab2)
    # SiameseKpOfDataset
    if is_siamese:
        kp1_list = [item[0][0] for item in batch] # might have None
        kp2_list = [item[0][1] for item in batch]
        flow1_list = [item[1][0] for item in batch] # might have None
        flow2_list = [item[1][1] for item in batch]
        lab1_list = [item[2][0] for item in batch]
        lab2_list = [item[2][1] for item in batch]
        y_list = [item[3] for item in batch]

        collated_lab1 = default_collate(lab1_list)
        collated_lab2 = default_collate(lab2_list)
        collated_y = default_collate(y_list)

        # special case for optical flow
        collated_kp1 = default_collate(kp1_list) if (kp1_list and kp1_list[0] is not None) else None
        collated_kp2 = default_collate(kp2_list) if (kp2_list and kp2_list[0] is not None) else None
        collated_flow1 = default_collate(flow1_list) if (flow1_list and flow1_list[0] is not None) else None
        collated_flow2 = default_collate(flow2_list) if (flow2_list and flow2_list[0] is not None) else None
        
        return (collated_kp1, collated_kp2), \
               (collated_flow1, collated_flow2), \
               (collated_lab1, collated_lab2), \
               collated_y
    # KpOfDataset
    else: 
        kp_list = [item[0] for item in batch]
        flow_list = [item[1] for item in batch] 
        lab_list = [item[2] for item in batch]
        
        collated_lab = default_collate(lab_list)
        
        # special case for optical flow
        collated_kp = default_collate(kp_list) if (kp_list and kp_list[0] is not None) else None
        collated_flow = default_collate(flow_list) if (flow_list and flow_list[0] is not None) else None
        return collated_kp, collated_flow, collated_lab
    
def prepare_dataloaders(data_params:DataArguments, 
                        aug_params_train:AugmentationArguments, 
                        aug_params_eval:AugmentationArguments, 
                        model_params:ModelArguments, 
                        train_params:TrainingArguments):
    
    val_dataset = KpOfDataset(data_params, aug_params_eval, model_params, istrain=False, fold_num=train_params.fold_num)
    train_dataset = KpOfDataset(data_params, aug_params_train, model_params, istrain=True, fold_num=train_params.fold_num)
    
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
        train_dataset = SiameseKpOfDataset(train_dataset, data_params)
    
    coord_path = os.path.join(model_params.pretrained_weight_dir, model_params.stgcn_coords_file_name)
    if not os.path.exists(coord_path):
        sample, _ , _ = val_dataset[0]            # sample.shape = (C, T, V)
        if isinstance(sample, torch.Tensor):
            sample = sample.cpu().numpy()
        sample.mean(axis=1)
        coords = sample.mean(axis=1).T 
        np.save(coord_path, coords)
    else:
        coords = np.load(coord_path)
    
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
    base_model = ActionRecognitionModel(model_params, data_params, coords)
    if not train_params.add_contrastive_loss:
        model = base_model
    else:
        model = ContrastiveActionWrapper(base_model, model_params.multihead_emb_dim)
    model = model.to(train_params.device)
    
    # set optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), train_params.learning_rate, momentum=train_params.momentum, weight_decay=train_params.opt_weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params.epochs, eta_min=1e-6)
    
    return {"loss":{"cross entropy": cross_entropy_loss, "contrastive learning": contrastive_loss}, "model":model, "optimizer":optimizer, "scheduler":scheduler}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
        batch_loss_ce_val = 0.0 
        
        if train_params.add_contrastive_loss and contrastive_loss: 
            (kp1, kp2), (flow1, flow2), (lab1, lab2), y = batch_data
            kp1, kp2 = kp1.to(device) if kp1 is not None else None, kp2.to(device) if kp2 is not None else None
            flow1, flow2 = flow1.to(device) if flow1 is not None else None, flow2.to(device) if flow2 is not None else None
            lab1, lab2 = lab1.to(device), lab2.to(device)
            y = y.to(device)
            emb1, logit1 = model(kp1, flow1)
            emb2, logit2 = model(kp2 ,flow2)
            
            # calculate loss
            loss_ce_part1 = cross_entropy_loss(logit1, lab1)
            loss_ce_part2 = cross_entropy_loss(logit2, lab2)
            batch_loss_ce_tensor = loss_ce_part1 + loss_ce_part2 
            batch_loss_ce_val = batch_loss_ce_tensor.item() 

            batch_loss_cl = contrastive_loss(emb1, emb2, y)
            batch_loss_total = batch_loss_ce_tensor + train_params.contrastive_loss_coefficient * batch_loss_cl 
            
            batch_loss_total.backward()
            optimizer.step()
            
            batch_size_pairs = lab1.size(0)
            total_individual_samples += batch_size_pairs * 2
            sum_loss += batch_loss_total.item() * batch_size_pairs

            _, pred1 = torch.max(logit1.data, 1)
            _, pred2 = torch.max(logit2.data, 1)
            correct += (pred1 == lab1).sum().item()
            correct += (pred2 == lab2).sum().item()
        else: 
            kps, flow, label = batch_data
            kps, flow = kps.to(device) if kps is not None else None, flow.to(device) if flow is not None else None
            label = label.to(device)
            
            _, output = model(kps, flow)

            batch_loss_ce_tensor = cross_entropy_loss(output, label) 
            batch_loss_ce_val = batch_loss_ce_tensor.item() 
            
            batch_loss_ce_tensor.backward() 
            optimizer.step()

            current_batch_size = label.size(0)
            total_individual_samples += current_batch_size 
            sum_loss += batch_loss_ce_val * current_batch_size 
            _, predict = torch.max(output.data, 1)
            correct += (predict == label).sum().item()
            batch_loss_total = batch_loss_ce_tensor 
        
        wandb.log({"train_batch_loss_total": batch_loss_total.item(), "train_batch_loss_ce": batch_loss_ce_val})
        
    epoch_loss = sum_loss / len(loader.dataset) 
    epoch_acc = correct / total_individual_samples
    
    wandb.log({"epoch": epoch, "train_loss":epoch_loss, "train_acc": epoch_acc})
    return epoch_loss, epoch_acc

def val_one_epoch(model, loader, cross_entropy_loss, epoch, actions:list, train_params:TrainingArguments):
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
    
    with torch.no_grad():
        for kps, flow, label in loader:
            
            kps, flow = kps.to(device) if kps is not None else None, flow.to(device) if flow is not None else None
            label = label.to(device)
            
            if train_params.add_contrastive_loss:
                _, output = model.backbone(kps, flow)
            else:
                _, output = model(kps, flow)
            loss = cross_entropy_loss(output, label)
            current_batch_size = label.size(0)
            total_samples += current_batch_size
            test_loss += loss.item() * current_batch_size
            _, predict = torch.max(output.data, 1)
            test_correct += (predict == label).sum().item()
            
            # add batch label & predict to list
            all_trues.extend(label.cpu().numpy())
            all_preds.extend(predict.cpu().numpy())
            
            # calculate class-wise accuracy
            corrects = (predict == label)
            for i in range(current_batch_size):
                true_label = label[i].item()       
                class_total[true_label] += 1       
                if corrects[i]:                    
                    class_correct[true_label] += 1 

    epoch_acc = test_correct / len(loader.dataset)
    epoch_loss = test_loss / len(loader.dataset)
    per_class_acc = class_correct / class_total.clamp(min=1)
    
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
        log_dict[f"val_acc_class_{action_name}"] = per_class_acc[i].item()
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
        acc_i = per_class_acc[i].item()
        prec_i = per_class_prec[i]
        rec_i = per_class_rec[i]
        f1_i = per_class_f1[i]
        cor_i = class_correct[i].item()
        tot_i = class_total[i].item()
        print(f"  Class {actions[i]}, Acc: {acc_i:.4f} ({cor_i}/{tot_i}), P {prec_i:.4f}, R {rec_i:.4f}, F1 {f1_i:.4f}")
    
    return epoch_loss, epoch_acc

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
    train_objects = build_model_and_optimizer(model_params, data_params, train_params, coords, class_weights)
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
        
        val_loss, val_acc = val_one_epoch(model, val_loader, cross_entropy_loss, epoch, actions=data_params.actions, train_params=train_params)
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
                weights_to_save = model.backbone.state_dict()
                print("Info: Saving backbone (ActionRecognitionModel) state_dict from ContrastiveActionWrapper.")
            else:
                weights_to_save = model.state_dict()
                print("Info: Saving ActionRecognitionModel state_dict.")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': weights_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc,
                'model_params': model_params, 
                'dataset_params': data_params, 
            }, os.path.join(saving_dir, 'best.pth'))
            patient_count = 0
        
        # early stopping 
        if patient_count > train_params.patient_epochs:
            break
        patient_count += 1
        
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    wandb.log({"best_val_acc": best_val_acc})
    
    run.finish()

        
if __name__ == "__main__":
    main()
