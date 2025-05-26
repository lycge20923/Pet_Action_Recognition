from dataclasses import dataclass, field
from typing import Literal, List
import torch

DEFAULT_ACTIONS_LIST = ["Running", "Walking", "Sniffing", "Standing(on all fours)", "Standing(bipedal)", "Sitting", "Lying", \
           "Coughing", "Seizures", "Vomiting", "Abnormal Movement"]
DEFAULT_SKELETON_LIST = [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [3, 5], [5, 6], [6, 7], [3, 8], [8, 9], [9, 10], [4, 14], [14, 15], [15, 16], [4, 11], [11, 12], [12, 13]]
DILATIONS_LIST = [1] # [1, 2, 3]
@dataclass
class DataArguments:
    actions: List[str] = field(
        default_factory=lambda: list(DEFAULT_ACTIONS_LIST),
        metadata={"help": "List of action categories to be recognized."}
    )
    data_dir: str = field(
        default="data",
        metadata={"help": "Destination path of downloaded videos"}
    )
    raw_dir_name: str = field(
        default="raw",
        metadata={"help": "Original dataset dir name"}
    )
    seg_dir_name: str = field(
        default="segmented",
        metadata={"help": "Segmented dataset dir name"}
    )
    stabilized_dir_name: str = field(
        default="stabilized",
        metadata={"help":"Stabilized dataset dir name"}
    )
    skip_stabilization: bool = field(
        default= False,
        metadata={"help":"Whether it would skip the step for video stabilization"}
    )
    stabilized_crop_percentage: float = field(
        default= 0.9,
        metadata={"help":"The cropping ratio for video stabilization"}
    )
    feature_extract_dir_name: str = field(
        default="feature_extracted",
        metadata={"help": "Pose_estimation dataset dir name"}
    )
    optical_flow_sub_dir_name: str = field(
        default="optical_flow",
        metadata={"help":"Sub dir name for storing numpy for each video"}
    )
    trainsplit_dir_name: str = field(
        default="train_split",
        metadata={"help":"After 5-fold split"}
    )
    metadata_name: str = field(
        default="metadata.csv",
        metadata={"help": "Name of the metadata file"}
    )
    operation: Literal["extend", "reload"] = field(
        default="reload",
        metadata={"help": "Operation to perform: extend (only create data for new videos) or reload the dataset"}
    )
    drive_url: str = field(
        default="https://docs.google.com/spreadsheets/d/10UWZqFRBe5JKn8gc0GOzNIlZilEwLHGP2AjiCc1Hj-I/export?format=csv",
        metadata={"help": "Website url of public google drive"}
    )
    annotation_file_name: str = field(
        default="annotation.json",
        metadata={"help":"Annotation name for storing data"}
    )
    min_kp_rate: float = field(
        default=0.5,
        metadata={"help":"Filter out those videos having low keypoint detection rate"}
    )
    window_size: int = field(
        default=64, 
        metadata={"help":"Frame number for splitting"}
    )
    num_samples: int = field(
        default=32,
        metadata={"help":"Sampling number for inputing to model"}
    )
    num_nodes: int = field(
        default=17,
        metadata={"help":"Number of nodes"}
    )
    num_coords: int = field(
        default=3,
        metadata={"help":"The size of each coordination"}
    )
    plot_pe: bool = field(default=False, metadata={"help": "Plot the results of pose estimation"})
    plot_pe_threshold: float = field(default=0.5, metadata={"help":"Threshold to show on the visualized images/videos"})


@dataclass
class PoseEstimationArguments:
    model_dir: str = field(
        default="models/pe",
        metadata={"help":"Dir storing models"}
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
        metadata={"help":"model name for optical flow"}
    )
    cfg: str = field(
        default="features/SEA_RAFT/config/eval/spring-M.json",
        metadata={"help":"The config for initiate optical flow model"}
    )
    input_model_size: List[int] = field(
        default_factory=lambda: list([256, 256]),
        metadata={"help":"Input size for 3D CNN"}
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
        default=2,
        metadata={"help":"Define what is 'neighbor'"}
    )
    in_channels: int = field(
        default=3,
        metadata={"help":"Input channel size in st-gcn"}
    )
    intermediate_channels: int = field(
        default=32,
        metadata={"help":"Intermediate channel size in st-gcn"}
    )
    final_channels: int = field(
        default=64,
        metadata={"help":"Final channel size in st-gcn"}
    )
    t_kernel_size: int =field(
        default=9,
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
    dilations: List[int] = field(
        default_factory=lambda: list(DILATIONS_LIST),
        metadata={"help":"List for dilations(branches)"}
    )
    multihead_emb_dim: int = field(
        default=128, 
        metadata={"help":"Multihead ST GCN embedding size(for contrastive learning)"}
    )
    add_velocity: bool = field(
        default=False,
        metadata={"help":"Add (x_diff, y_diff) in information of keypoints"}
    )
    
@dataclass
class AugmentationArguments:
    augment: bool = field(
        default=False,
        metadata={"help": "Whether to apply data augmentation."}
    )
    rot_max: float = field(
        default=20.0,
        metadata={"help": "Maximum rotation angle for augmentation (in degrees)."}
    )
    scale_min: float = field(
        default=0.9,
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
        default=0.1,
        metadata={"help": "Probability of dropping individual joints."}
    )
    frame_drop_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping keypoints from an entire frame."}
    )
    shear_max: float = field(
        default=0.1,
        metadata={"help": "Maximum shear intensity or angle for augmentation."}
    )
    temporal_jitter_prob: float = field(
        default=0.1,
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
    epochs: int = field(
        default=3000,
        metadata={"help": "Total number of training epochs."}
    )
    batch_size: int = field(
        default=64,
        metadata={"help": "Batch size for training and evaluation."}
    )
    learning_rate: float = field(
        default=0.0003,
        metadata={"help": "Initial learning rate for the optimizer."}
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
        default=1e-4,
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
    add_contrastive_loss: bool = field(
        default=False,
        metadata={"help":"Add contrastive loss"}
    )
    contrastive_loss_coefficient: float = field(
        default=0.5,
        metadata={"help":"The coefficient for adding contrastive loss"}
    )