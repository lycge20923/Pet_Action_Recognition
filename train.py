import dataclasses
from datetime import datetime
import json
import numpy as np
import os
import random
import wandb

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from src.models.stgcn_model import ST_GCN
from src.data_processing.dataset import JointsDataset
from src.utils.cli_args import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    correct = 0
    sum_loss = 0
    total_samples = 0
    for batch_idx, (data, label) in enumerate(loader):
        data = data.to(device)
        label = label.to(device)
        

        output = model(data)

        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_batch_size = label.size(0)
        total_samples += current_batch_size
        sum_loss += loss.item() * current_batch_size
        _, predict = torch.max(output.data, 1)
        correct += (predict == label).sum().item()
        
        wandb.log({"batch_train_loss": loss.item()})
        
    epoch_acc = (correct / total_samples)
    epoch_loss = (sum_loss/ total_samples)
    
    wandb.log({"epoch": epoch, "train_loss":epoch_loss, "train_acc": epoch_acc})
    return epoch_loss, epoch_acc

def val_one_epoch(model, loader, criterion, device, epoch, actions:list):
    model.eval()
    test_correct = 0
    test_loss = 0
    total_samples = 0
    
    num_classes = len(actions)
    class_correct = torch.zeros(num_classes).to(device)
    class_total = torch.zeros(num_classes).to(device)
    
    with torch.no_grad():
        for data, label in loader:
            
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)
            current_batch_size = label.size(0)
            total_samples += current_batch_size
            test_loss += loss.item() * current_batch_size
            _, predict = torch.max(output.data, 1)
            test_correct += (predict == label).sum().item()
            
            
            # calculate class-wise accuracy
            current_batch_size = label.size(0)
            corrects = (predict == label)
            for i in range(current_batch_size):
                true_label = label[i].item()       
                class_total[true_label] += 1       
                if corrects[i]:                    
                    class_correct[true_label] += 1 

    epoch_acc = test_correct / len(loader.dataset)
    epoch_loss = test_loss / len(loader.dataset)
    
    per_class_acc = class_correct / class_total.clamp(min=1)
    log_dict = {"epoch":epoch, "val_loss": epoch_loss, "val_acc":epoch_acc}
    for i in range(num_classes):
        log_dict[f"val_acc_class_{actions[i]}"] = per_class_acc[i].item()
        log_dict[f"val_samples_class_{actions[i]}"] = class_total[i].item()
    wandb.log(log_dict)
    print("Per-class Validation Accuracy:")
    for i in range(num_classes):
        print(f"  Class {actions[i]}: {per_class_acc[i].item():.4f} ({int(class_correct[i].item())}/{int(class_total[i].item())})")
    
    return epoch_loss, epoch_acc

def main():
    # set wandb
    run = wandb.init(project="st_gcn_adjust")
    
    # load parameters and add to wandb
    d_params = DataArguments()
    model_params = ModelArguments(
        intermediate_channels=wandb.config.intermidiate_channels,
        final_channels=wandb.config.final_channels,
        t_kernel_size=wandb.config.t_kernel_size
    )
    train_params = TrainingArguments(
        learning_rate=wandb.config.learning_rate,
        opt_weight_decay=wandb.config.opt_weight_decay
    )
    aug_params_train = AugmentationArguments(
        augment=wandb.config.augment, 
        rot_max=wandb.config.rot_max,
        scale_min=wandb.config.scale_min,
        scale_max=wandb.config.scale_max,
        joint_drop_prob=wandb.config.joint_drop_prob,
        frame_drop_prob=wandb.config.frame_drop_prob
    )
    aug_params_eval = AugmentationArguments(
        augment=False
    )
    
    # set seed
    set_seed(train_params.seed)
    
    train_dataset = JointsDataset(d_params, aug_params_train, model_params, istrain=True)
    val_dataset = JointsDataset(d_params, aug_params_eval, model_params, istrain=False)

    # build dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params.batch_size,
        shuffle=True,
        num_workers=train_params.num_workers,
        pin_memory=True if train_params.device == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_params.batch_size,
        shuffle=False, 
        num_workers=train_params.num_workers,
        pin_memory=True if train_params.device == "cuda" else False
    )
    
    # initial set
    sample, _ = val_dataset[0]            # sample.shape = (C, T, V)
    if isinstance(sample, torch.Tensor):
        sample = sample.cpu().numpy()
    sample.mean(axis=1)
    coords = sample.mean(axis=1).T 
    
    joint_coords = sample.mean(axis=1).T  # shape (V, C)
    model = ST_GCN(params=model_params, d_params=d_params, coords=coords, dilations=model_params.dilations).to(train_params.device)
    optimizer = torch.optim.SGD(model.parameters(), train_params.learning_rate, momentum=train_params.momentum, weight_decay=train_params.opt_weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params.epochs, eta_min=0)
    
    # save setting
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    saving_dir = os.path.join(os.path.dirname(__file__), train_params.save_dir_name, timestamp)
    os.makedirs(saving_dir, exist_ok=True)
    
    # for stats
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    best_val_acc = 0.0 
    
    # start training 
    patient_count = 0 # for early stopping
    for epoch in range(1, train_params.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, train_params.device, epoch)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        val_loss, val_acc = val_one_epoch(model, val_loader, criterion, train_params.device, epoch, actions=d_params.actions)
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
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc,
                'model_params': model_params, 
                'dataset_params': d_params, 
            }, os.path.join(saving_dir, 'best.pth'))
            patient_count = 0
        
        # early stopping 
        if patient_count > train_params.patient_epochs:
            break
        patient_count += 1
        
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    wandb.log({"best_val_acc": best_val_acc})
    
    # save parameters and plot
    config_to_save = {
        "dataset_params": dataclasses.asdict(d_params),
        "augmentation_params_train": dataclasses.asdict(aug_params_train),
        "model_params": dataclasses.asdict(model_params),
        "training_params": dataclasses.asdict(train_params),
        "training_completed_at": timestamp 
    }
    pars_filename = "args.json"
    try:
        with open(os.path.join(saving_dir, pars_filename), 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"{e}")
    
    run.finish()

        


if __name__ == "__main__":
    main()
