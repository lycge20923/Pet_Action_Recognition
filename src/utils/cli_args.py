'''
The parameters may be a bit messy, but please ensure the following:
1. If you create a new parameter, make sure its name does not conflict with any existing ones, even if they belong to different classes.
2. In addition to creating new parameters for better management, it is recommended to customize your training process using a YAML configuration file similar to those in the configs directory.
'''

from dataclasses import dataclass, field
from typing import Literal, List, Optional
import torch

DEFAULT_ACTIONS_LIST = ["Running", "Walking", "Sniffing", "Standing(on all fours)", "Standing(bipedal)", "Sitting", "Lying", \
           "Coughing", "Seizures", "Vomiting", "Abnormal Movement"]
DEFAULT_SKELETON_LIST = [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [3, 5], [5, 6], [6, 7], [3, 8], [8, 9], [9, 10], [4, 14], [14, 15], [15, 16], [4, 11], [11, 12], [12, 13]]
# DILATIONS_LIST = [1] # [1, 2, 3]
@dataclass
class DataArguments:
    actions: List[str] = field(
        default_factory=lambda: list(DEFAULT_ACTIONS_LIST),
        metadata={"help": "List of action categories to be recognized."}
    )
    ##############################
    ### for data preprocessing ###
    ##############################
    # --- for download and segment ---
    data_dir: str = field(
        default="data/main",
        metadata={"help":"Root directory for storing all main data, including raw and preprocessed versions."}
    )
    raw_dir_name: str = field(
        default="raw",
        metadata={"help":"Subdirectory name for storing the original downloaded videos."}
    )
    seg_dir_name: str = field(
        default="segmented",
        metadata={"help":"Subdirectory name for storing action-segmented videos."}
    )
    metadata_name: str = field(
        default="metadata.csv",
        metadata={"help":"Name of the metadata CSV file to be used."}
    )
    operation: Literal["extend", "reload"] = field(
        default="reload",
        metadata={"help": "Operation to perform: extend (only create data for new videos) or reload the dataset"}
    )
    drive_url: str = field(
        default="https://docs.google.com/spreadsheets/d/10UWZqFRBe5JKn8gc0GOzNIlZilEwLHGP2AjiCc1Hj-I/export?format=csv",
        metadata={"help": "URL of the Excel file stored on Google Drive."}
    )
    # --- for video stabilization ---
    stabilized_dir_name: str = field(
        default="stabilized",
        metadata={"help":"Subdirectory name for storing stabilized videos."}
    )
    # possible parameter for ablation study
    skip_stabilization: bool = field(
        default= False,
        metadata={"help":"Indicates whether the video stabilization step should be skipped."}
    )
    stabilized_crop_percentage: float = field(
        default= 0.9,
        metadata={"help":"Defines the cropping ratio to remove black borders that may appear due to video stabilization."}
    )
    # --- for feature extraction ---
    feature_extract_dir_name: str = field(
        default="feature_extracted",
        metadata={"help": "Subdirectory name for storing the results of keypoint detection and optical flow."}
    )
    # possible parameter for ablation study
    do_crop: bool = field(
        default=True,
        metadata={"help":"Cropping is performed before feeding the segments into the optical flow extractor."}
    )
    plot_pe: bool = field(default=False, metadata={"help": "Plot the results of pose estimation"})
    plot_pe_threshold: float = field(default=0.5, metadata={"help":"Threshold to show on the visualized images/videos"})
    # --- for train-val & 5-fold split ---
    trainsplit_dir_name: str = field(
        default="train_split",
        metadata={"help":"Subdirectory name for storing data after x-fold split."}
    )
    num_folds: int = field(
        default=5,
        metadata={"help":"Number of folds to use for cross-validation."}
    )
    fold_allow_diff: int = field(
        default=200, 
        metadata={"help":"The maximum allowed difference in the number of samples between action classes."}
    )
    min_kp_rate: float = field(
        default=0.5,
        metadata={"help":"Filter out those videos having low keypoint detection rate"}
    )
    window_size: int = field(
        default=64, 
        metadata={"help":"Number of consecutive frames used to form a window segment for sampling."}
    )
    # possible parameter for ablation study
    num_samples: int = field(
        default=32,
        metadata={"help":"Number of sampled frames from each window as input to the model."}
    )
    annotation_file_name: str = field(
        default="annotation.json",
        metadata={"help":"Filename of the annotation file used to store data."}
    )
    num_nodes: int = field(
        default=17,
        metadata={"help":"The defined number of keypoints that can be detected"}
    )
    num_coords: int = field(
        default=3,
        metadata={"help":"The size of each coordination"}
    )

@dataclass
class PoseEstimationArguments:
    model_dir: str = field(
        default="models/pe",
        metadata={"help":"Directory for storing weight of the pose estimation model."}
    )
    model_type: Literal['s', 'b', 'l', 'h'] = field(
        default="h",
        metadata={"help":"Size of the VitPose model (choose from 's', 'b', 'l', 'h')"}
    )
    yolo_type: Literal['s', 'n'] = field(
        default='s',
        metadata={"help":"Size of the Yolo model (choose from 's', 'n')"}
    )
    yolo_size: int = field(
        default=320,
        metadata={"help":"Input size for Yolo model"}
    )
    pretrained_dataset: Literal["coco", "coco_25", "wholebody", "mpii", "ap10k", "apt36k", "aic", "custom"] = field(
        default="ap10k",
        metadata={"help":"Model pretrained on the specific dataset"}
    )

@dataclass
class OpticalFlowArguments:
    model_url_name: str =field(
        default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
        metadata={"help":"Model identifier or path used for optical flow inference."}
    )
    cfg: str = field(
        default="features/SEA_RAFT/config/eval/spring-M.json",
        metadata={"help":"Configuration file to initialize the optical flow model."}
    )
    input_model_size: List[int] = field(
        default_factory=lambda: list([256, 256]),
        metadata={"help":"Input resolution for the 3D CNN model (height, width)."}
    )

@dataclass
class OutputArguments:
    output_dir: str = field(
        default="output",
        metadata={"help":"Dir storing experimental results"}
    )

@dataclass
class ModelArguments:
    hop_size: int = field(
        default=1,
        metadata={"help":"Define what is 'neighbor'"}
    )
    in_channels: int = field(
        default=3,
        metadata={"help":"Input channel size in st-gcn"}
    )
    base_channels: int = field(
        default=64, 
        metadata={"help":"Base channel size in gcn."}
    )
    # intermediate_channels: int = field(
    #     default=32,
    #     metadata={"help":"Intermediate channel size in st-gcn"}
    # )
    # final_channels: int = field(
    #     default=128,
    #     metadata={"help":"Final channel size in st-gcn"}
    # )
    t_kernel_size: int =field(
        default=13,
        metadata={'help':"refer total t kernel size in temporal conv"}
    )
    num_classes: int = field(
        default=len(DEFAULT_ACTIONS_LIST),
        metadata={"help":"Total label to predict"}
    )
    neighbor_base: List[str] = field(
        default_factory=lambda: list(DEFAULT_SKELETON_LIST),
        metadata={"help": "List of skeleton (node pairs) to be recognized."}
    )
    dilation: int = field(
        default=1,
        metadata={"help":"Dilation for ST-GCN"}
    )
    multihead_emb_dim: int = field(
        default=128, 
        metadata={"help":"Multihead ST GCN embedding size(for contrastive learning)"}
    )
    add_optical_flow: bool = field(
        default=True,
        metadata={"help":"Whether adding optical flow"}
    )
    pretrained_weight_dir: str = field(
        default="models",
        metadata={"help":"Root directory to store pretrained weights"}
    )
    gcn_weights_dir_name: str = field(
        default="GCN",
        metadata={"help":"Sub directory to store pretrained weights for ST_GCN(self-training)"}
    )
    gcn_model_name: Literal["stgcn", "tdgcn", "degcn"] = field(
        default="tdgcn",
        metadata={"help": "The model for training and predict. Now we only have ST-GCN & TD-GCN"}
    )
    gcn_weights_file_template: str = field(
        default="best_{}.pth",
        metadata={"help":"Weights to store pretrained weights for GCN(self-training)\
                          If you are using the training dataset whose fold num is not 0, \
                          stronly recommended to retrained the gcn"}
    )
    @property
    def gcn_weights_file_name(self) -> Optional[str]:
        return self.gcn_weights_file_template.format(self.gcn_model_name)
    
    gcn_include_blocks: List[int] = field(
        default_factory=lambda: [1, 5, 8, 10],
        metadata={"help": "Which GCN blocks to include (1~10)"}
    )
    degcn_num_streams: int = field(
        default=2, 
        metadata= {"help": "The number of streams used in DE-GCN"}
    )
    
    load_gcn_weights: bool = field(
        default=False, 
        metadata={"help":"Whether continue training using pre-trained ST-GCN"}
    )
    stgcn_coords_file_name: str = field(
        default="coords_stgcn.npy",
        metadata={"help":"ST-GCN needs one center coordinate, thus it could be accessed in the file"}
    )
    I3D_weights_dir_name: str = field(
        default="I3D",
        metadata={"help":"Dir to store pretrained weights for I3D"}
    )
    I3D_weights_file_name: str = field(
        default="flow_imagenet.pt",
        metadata={"help":"I3D pretrained weights file name"}
    )
    I3D_raw_feat_dim: int = field(
        default=4096,
        metadata={"help":"Raw feature dimension of I3D"}
    )
    I3D_project_dim: int = field(
        default=128,
        metadata={"help":"Project to have the similar size with ST-GCN"}
    )
    only_optical_flow: bool = field(
        default=False, 
        metadata={"help":"Only use optical flow and I3D to conduct action recognition"}
    )
    
    
@dataclass
class AugmentationArguments:
    augment: bool = field(
        default=True,
        metadata={"help": "Whether to apply data augmentation."}
    )
    rot_max: float = field(
        default=33.0,
        metadata={"help": "Maximum rotation angle for augmentation (in degrees)."}
    )
    scale_min: float = field(
        default=0.8,
        metadata={"help": "Minimum scaling factor for augmentation."}
    )
    scale_max: float = field(
        default=1.1,
        metadata={"help": "Maximum scaling factor for augmentation."}
    )
    trans_max: float = field(
        default=0.05,
        metadata={"help": "Maximum translation factor (relative to pose size) for augmentation."}
    )
    noise_std: float = field(
        default=0.01,
        metadata={"help": "Standard deviation of Gaussian noise to add to keypoints."}
    )
    joint_drop_prob: float = field(
        default=0.2,
        metadata={"help": "Probability of dropping individual joints."}
    )
    frame_drop_prob: float = field(
        default=0.02,
        metadata={"help": "Probability of dropping keypoints from an entire frame."}
    )
    shear_max: float = field(
        default=0.08,
        metadata={"help": "Maximum shear intensity or angle for augmentation."}
    )
    temporal_jitter_prob: float = field(
        default=0.12,
        metadata={"help": "Probability of applying temporal jittering."}
    )
    valid_kpt_confidence_thresh: float = field(
        default=0.1, # Example threshold, adjust as needed
        metadata={"help": "Confidence threshold above which a keypoint is considered valid for geometric augmentations."}
    )

@dataclass
class TrainingArguments:
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to use for training (e.g., 'cuda', 'cpu')."}
    )
    fold_num: int = field(
        default=0,
        metadata={"help": "Fold number for k-fold cross-validation in the training step."}
    )
    epochs: int = field(
        default=100,
        metadata={"help": "Total number of training epochs."}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for training and evaluation."}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Initial learning rate for the optimizer."}
    )
    scheduler_eta_min: float = field(
        default=1e-5,
        metadata={"help":"Scheduler's eta_min"}
    )
    train_ratio: float = field(
        default=0.8,
        metadata={"help": "Ratio of the dataset to use for training (the rest for validation)."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of worker processes for data loading."}
    )
    opt_weight_decay: float = field(
        default=1e-5,
        metadata={"help": "Weight decay (L2 penalty) for the optimizer."}
    )
    momentum: float = field(
        default=0.9,
        metadata={"help": "Momentum factor for SGD optimizer (if used)."}
    )
    save_dir_name: str = field(
        default="runs",
        metadata={"help":"Saving dir for training"}
    )
    patient_epochs: int = field(
        default= 1000, 
        metadata={"help":"If the performance is not good for a long time, terminate it!"}
    )
    # for ablation study
    add_contrastive_loss: bool = field(
        default=True,
        metadata={"help":"Add contrastive loss"}
    )
    contrastive_loss_coefficient: float = field(
        default=0.4,
        metadata={"help":"The coefficient for adding contrastive loss"}
    )
    save_complete_args_name: str = field(
        default="args_complete.json",
        metadata={"help":"The file name for saving complete args"}
    )
    save_adjusted_args_name: str = field(
        default="args_adjusted.yaml",
        metadata={"help":"The file name for saving adjusted args, those would be used in prediction"}
    )
    save_gcn_weights: bool = field(
        default=False,
        metadata={"help": "Whether cover the best model weight of STGCN."}
    )
    use_multiplie_learning_rates: bool = field(
        default=False,
        metadata={"help": "Since we use multiple branch, \
            thus we experiment for different learning rate for different branches"}
    )
    branch_stgcn_learning_rate: float = field(
        default=3e-5,
        metadata={"help":"If use 'use_multiplie_learning_rates', then this \
            indicate the learning rate for stgcn branch"}
    )
    branch_I3D_learning_rate: float = field(
        default=1e-4,
        metadata={"help":"If use 'use_multiplie_learning_rates', then this \
            indicate the learning rate for I3D branch"}
    )
    branch_fuse_head_learning_rate: float = field(
        default=1e-4,
        metadata={"help":"If use 'use_multiplie_learning_rates', then this \
            indicate the learning rate for fusing branch"}
    )