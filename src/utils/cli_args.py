'''
The parameters may be a bit messy, but please ensure the following:
1. If you create a new parameter, make sure its name does not conflict with any existing ones, even if they belong to different classes.
2. In addition to creating new parameters for better management, it is recommended to customize your training process using a YAML configuration file similar to those in the configs directory.
'''

from dataclasses import dataclass, field
from typing import Literal, List, Optional, Tuple
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
    fold_num: int = field(
        default=0,
        metadata={"help": "Fold number for k-fold cross-validation in the training step."}
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
class ResultsArguments:
    output_dir: str = field(
        default="output",
        metadata={"help":"Dir storing experimental results"}
    )

@dataclass
class ModelArguments:
    _num_coords: int = field(
        default=3,
        metadata={"help": "The size of each coordination"}
    )
    add_flow_coords: bool = field(
        default=True,
        metadata={"help":"Delete confidence, and extend optical flow information (m_x, m_y, std_x, std_y) for gcn"}
    )
    flow_statistic_mode: Literal["mean+max+std", "mean+max", "mean"] = field(
        default="mean+max+std",
        metadata={"help":"What kinds of statistic for sending the flow information."}
    )
    flow_block_size:int = field(
        default=5,
        metadata={"help":"The patch size of the flow for each keypoints"}
    )
    @property
    def in_channels(self) -> int:
        if self.add_flow_coords:
            if self.flow_statistic_mode == "mean+max+std":
                return 8
            elif self.flow_statistic_mode == "mean+max": 
                return 6
            else:
                return 4
        return self._num_coords
    base_channels: int = field(
        default=64, 
        metadata={"help":"Base channel size in gcn."}
    )
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
    multihead_emb_dim: int = field(
        default=128, 
        metadata={"help":"Multihead ST GCN embedding size(for contrastive learning)"}
    )
    pretrained_weights_root_dir_name: str = field(
        default="models",
        metadata={"help": "Root directory to store pretrained weights"}
    )
    self_training_weights_dir_name: str = field(
        default="self_training",
        metadata={"help": "Directory name for saving self-training weights"}
    )
    gcn_weights_dir_name: str = field(
        default="GCN",
        metadata={"help":"Sub directory to store pretrained weights for ST_GCN(self-training)"}
    )
    gcn_model_name: Literal["stgcn", "tdgcn", "degcn"] = field(
        default="degcn",
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
    # for ablation study
    add_flow_adjacency: bool = field(
        default=True,
        metadata={"help":"Whether add leanable adjacency matrix from optical flow in the architecture in DE-GCN"}
    )
    load_gcn_weights: bool = field(
        default=False, 
        metadata={"help":"Whether continue training using pre-trained GCN model weights"}
    )
    # stgcn related parameters
    stgcn_hop_size: int = field(
        default=1,
        metadata={"help":"Define what is 'neighbor'"}
    )
    stgcn_dilation: int = field(
        default=1,
        metadata={"help":"Dilation for ST-GCN"}
    )
    stgcn_coords_file_name: str = field(
        default="coords_stgcn.npy",
        metadata={"help":"ST-GCN needs one center coordinate, thus it could be accessed in the file"}
    )
    # I3D related parameters
    add_I3D_branch: bool = field(
        default=True,
        metadata={"help":"Whether add parallel I3D branch in the model for the prediction"}
    )
    only_I3D_branch: bool = field(
        default=False,
        metadata={"help":"Whether only use I3D branch(exclude GCN branch) in the model for the prediction"}
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
    # optical‐flow specific augmentations
    flow_noise_std: float = field(
        default=0.01,
        metadata={"help": "Standard deviation for Gaussian noise added to optical flow."}
    )
    flow_scale_range: Tuple[float, float] = field(
        default=(0.8, 1.2),
        metadata={"help": "Range (min, max) for random scaling of optical flow magnitude."}
    )
    flow_hflip_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of horizontally flipping the optical flow."}
    )
    flow_occl_ratio: float = field(
        default=0.2,
        metadata={"help": "Occlusion block size ratio (relative to H and W) for optical flow."}
    )
    flow_occl_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of applying random occlusion to optical flow."}
    )
    flow_blur_ksize: int = field(
        default=5,
        metadata={"help": "Gaussian blur kernel size applied to optical flow."}
    )
    flow_blur_sigma: float = field(
        default=1.0,
        metadata={"help": "Sigma (standard deviation) for Gaussian blur on optical flow."}
    )
    flow_blur_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of applying Gaussian blur to optical flow."}
    )

@dataclass
class TrainingArguments:
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to use for training (e.g., 'cuda', 'cpu')."}
    )

    epochs: int = field(
        default=100,
        metadata={"help": "Total number of training epochs. For only training GCN, it is recommended to change to at least 1000"}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for training and evaluation. For only training GCN, it should change to 32"}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Initial learning rate for the optimizer."}
    )
    scheduler_eta_min: float = field(
        default=1e-4,
        metadata={"help":"Scheduler's eta_min"}
    )
    resume_checkpoint_dir: str = field(
        default=None,
        metadata={"help":"If there is no value, retrain; if there is a value, load the pretrained weights."}
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
    save_root_dir_name: str = field(
        default="runs",
        metadata={"help":"The root directory for saving files while training"}
    )
    patient_epochs: int = field(
        default= 500, 
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
    save_best_gcn_weights: bool = field(
        default=False,
        metadata={"help": "Whether cover the best model weight of STGCN."}
    )
    save_predict_details_name: str = field(
        default="details.json",
        metadata={"help": "The file name for saving predict details(could set None not to save)"}
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