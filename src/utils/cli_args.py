from dataclasses import dataclass, field
from typing import Literal, List

DEFAULT_ACTIONS_LIST = ["Running", "Walking", "Sniffing", "Standing(on all fours)", "Standing(bipedal)", "Sitting", "Lying", \
           "Coughing", "Seizures", "Vomiting", "Abnormal Movement"]

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
    pe_dir_name: str = field(
        default="pose_estimation",
        metadata={"help": "Pose_estimation dataset dir name"}
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
class OutputArguments:
    output_dir: str = field(
        default="output",
        metadata={"help":"Dir storing experimental results"}
    )