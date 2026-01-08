# Pet Action Recognition

This is the project for **A Multi-Stream Framework Integrating Joint and Optical-Flow Representations for Pet Action Recognition**

## Dataset Preparation

### Notes: It is necessary to Monitor disk capacity in real time with `watch df -h`, not to let it exceed the limitation!!!

### Set Up Environment

* Create new virtual environment: ```conda create --name PAR python=3.10 -y```

* Go to the environment: ```conda activate PAR```

* Install related packages: 

    ```
    pip install -r requirements.txt
    cd features/easy_ViTPose
    pip install -e .
    pip install -r requirements.txt
    cd ../..
    ```

* Troubleshooting: If facing `cannot import name 'Sentinel' from 'typing_extensions'`, run `pip install -U "typing-extensions>=4.14.0"`

### Dataset Check List

* Before running the step of training/testing, or prediction, go check the folders in `data/main` (for **PetAction**) or `data/others/KABR` (for **KABR**) to determine what steps to run next.

    <!-- | `raw` | `segmented` | `stabilized` | `feature_extracted` | `train_split` |   -->

    <table>
        <thead>
            <tr>
                <th rowspan="2">Step</th>
                <th colspan="5">Folders</th>
                <th colspan="5">Necessary Preprocessing(s)</th>
                <th rowspan="2">Notes</th>
            </tr>
            <tr>
                <th><code>raw</code></th>
                <th><code>segmented</code></th>
                <th><code>stabilized</code></th>
                <th><code>feature_extracted</code></th>
                <th><code>train_split</code></th>
                <th>1. download</th>
                <th>2. segmentation</th>
                <th>3. stabilization</th>
                <th>4. feature extraction</th>
                <th>5. dataset splitting</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="6"><b>Train/Test</b></td>
                <td>✅/❌</td>
                <td>✅/❌</td>
                <td>✅/❌</td>
                <td>✅/❌</td>
                <td>✅</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>Nothing to do in this step, please skip <code>## Dataset Preparation</code></td>
            </tr>
            <tr>
                <td>✅</td>
                <td>✅</td>
                <td>✅</td>
                <td>✅</td>
                <td>❌</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>⭕</td>
                <td></td>
            </tr>
            <tr>
                <td>✅</td>
                <td>✅</td>
                <td>✅</td>
                <td>❌</td>
                <td>❌</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>⭕</td>
                <td>⭕</td>
                <td></td>
            </tr>
            <tr>
                <td>✅</td>
                <td>✅</td>
                <td>❌</td>
                <td>❌</td>
                <td>❌</td>
                <td>-</td>
                <td>-</td>
                <td>△</td>
                <td>⭕</td>
                <td>⭕</td>
                <td>It could be skipped. More details see in <code>### Stabilization</code></td>
            </tr>
            <tr>
                <td>✅</td>
                <td>❌</td>
                <td>❌</td>
                <td>❌</td>
                <td>❌</td>
                <td>-</td>
                <td>△</td>
                <td>⭕</td>
                <td>⭕</td>
                <td>⭕</td>
                <td>For PetAction: you need to conduct segmentation. For KABR, it is not necessary</td>
            </tr>
            <tr>
                <td>❌</td>
                <td>❌</td>
                <td>❌</td>
                <td>❌</td>
                <td>❌</td>
                <td>⭕</td>
                <td>△</td>
                <td>⭕</td>
                <td>⭕</td>
                <td>⭕</td>
                <td></td>
            </tr>
            <tr>
                <td rowspan="3"><b>Prediction</b></td>
                <td>✅/❌</td>
                <td>✅</td>
                <td>✅/❌</td>
                <td>✅/❌</td>
                <td>✅/❌</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>Nothing to do in this step, please skip <code>## Dataset Preparation</code></td>
            </tr>
            <tr>
                <td>✅</td>
                <td>❌</td>
                <td>✅/❌</td>
                <td>✅/❌</td>
                <td>✅/❌</td>
                <td>-</td>
                <td>⭕</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td></td>
            </tr>
            <tr>
                <td>❌</td>
                <td>❌</td>
                <td>✅/❌</td>
                <td>✅/❌</td>
                <td>✅/❌</td>
                <td>⭕</td>
                <td>⭕</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td></td>
            </tr>
        </tbody>
    </table>

### Dataset Download

#### PetAction

- First, check whether the required metadata file exists. There are two possible locations. If neither exists, contact someone:

    1. Google Sheet: [link](https://docs.google.com/spreadsheets/d/10UWZqFRBe5JKn8gc0GOzNIlZilEwLHGP2AjiCc1Hj-I/edit?gid=0#gid=0)
    
    2. `data/main/metadata.csv`

- Run the following command:

    ```bash
    python -m src.data_processing.download
    ```
    
    * If file in Google Sheet exists and you just want to download the videos that have not been downloaded yet, you could modify `operation` parameter in `src/utils/cli_args.py` to choose `extend`. 

#### KABR

- Conduct:

    ```bash
    python -m src.data_processing.ref_download --dataset_name KABR
    ```

### Segmentation

#### PetAction

- Conduct:

    ```bash
    python -m src.data_processing.segment
    ```

#### KABR

- (skipped)

### Stabilization

#### PetAction

- Conduct:

    ```bash
    python -m src.data_processing.stabilization
    ```

#### KABR

- Conduct:

    ```bash
    python -m src.data_processing.stabilization --for_comparison --dataset_name KABR
    ```

### Feature Extraction

- You could add `CUDA_VISIBLE_DEVICES=<GPU device ID>` to specify the GPU device ID.

- If you want to use the unstabilized video from `data/main/segmented` instead of `data/main/stabilized`, you could modify `skip_stabilization` parameter in `src/utils/cli_args.py` to choose `True`. 

#### PetAction

- Conduct:

    ```bash
    python -m src.data_processing.feature_extraction
    ```
#### KABR

- Conduct:

    ```bash
    python -m src.data_processing.feature_extraction --for_comparison --dataset_name KABR
    ```
### Dataset Splitting

- After conducting the commands described in the following paragraph, if you want to conduct training/testing of **CTR-GCN**, **InfoGCN** or **TD-GCN**, you have to additionally conduct the following command:

    ```bash
    python -m comparison.convert # for PetAction
    python -m comparison.convert --other_dataset_name KABR # for KABR
    ```

#### PetAction

- Conduct:

    ```bash
    python -m src.data_processing.train_split
    ```

#### KABR

- Conduct:

    ```bash
    python -m src.data_processing.train_split --for_comparison --dataset_name KABR
    ```

## Train

### Notes: It is necessary to Monitor memory usage in real time with `htop`, not to let it exceed the limitation!!!

### Wandb & Config Introduction

- Before training, you have to build a `wandb` account.

- For `wandb` training, it must use yaml to declare training parameters.

- The yaml could be adjusted for parameter changes, e.g. only want to train on `fold_num` == 1, then set/modify the following code on your config file:
    
    ```
    fold_num:
        values: [1]
    ```

- For more parameters to choose, you could refer `src/utils/cli_args.py`

- There are some config files in the `configs`:

    1. Main, for our proposed architecture:

        * `configs/sweep_joints.yaml`: **Joints** Stream

        * `configs/sweep_joints.yaml`: **Local Flow** Stream

        * `configs/sweep_i3dgcn.yaml`: **I3DGCN** Stream

    2. Other but share the same training architecture

        * `configs/sweep_I3D.yaml`: **I3D** Stream, the `I3D_mode` could be adjusted to `rgb` to use rgb-based I3D

        * `configs/sweep_X3D.yaml`: **X3D** Stream

        * `configs/sweep_stgcn.yaml`: **ST-GCN** Stream
    
    3. KABR
        
        * `configs/sweep_KABR.yaml`: It could be used for training or reference for setting in KABR
    
    4. Experiment/Testing

        * `configs/sweep_experiment.yaml`: Just for experiment

        * `configs/sweep_test.yaml`: For simple and quick testing 
    
- You might see that also some configs in `configs/comp`, just skip those files in the folder, we would introduce in the next section

### Start Training

- Before training, make sure your configs are set right

- Two types of training: Main & Others(for **CTR-GCN**, **InfoGCN**, **TD-GCN**)

#### Main

- Main training(except for model **CTR-GCN**, **InfoGCN**, **TD-GCN**): You could choose two scripts for training:

    1. `scripts/train_with_wandb.sh`

        ```bash
        bash scripts/train_with_wandb.sh <GPU device ID> <Sweep File path>
        ```

        * e.g. `bash scripts/train_with_wandb.sh 2 configs/sweep_local_flow.yaml`

    2. `scripts/onestep_experiment.sh`: This would directly use the config file `configs/sweep_experiment.yaml`

        ```bash
        bash scripts/onestep_experiment.sh <GPU device ID> 
        ```

        * e.g. `bash scripts/onestep_experiment.sh 2

- When successfully conducting training, the folder for training would be created under `runs` folder, and the naming of the folder would be `<Name of exec_name in config>_<Time>`. For example, `formal:X3D stream(no-pretrained)_20251112_113230`, where `formal:X3D stream(no-pretrained)` is the `exec_name` in config file and `20251112_113230` would be near the time start training. In the folder, there would be several files:

    1. `args_adjusted.yaml` & `args_complete.json`: Training parameters for specifically and total declaration, separately

    2. `best.pth`, `confusion_matrix.csv`, `details.json`: Weights, confusion matrix, prediction details for the best accuracy during training

#### Others

- This is for model **CTR-GCN**, **InfoGCN**, **TD-GCN** training. Since those are 4-streams(Joint, Joint Velocity, Bone, Bone Velocity), while training one model, you have to conduct four times command with different setting for each time 

    1. **CTR-GCN**, **TD-GCN**

        - Those are corresponded to `CTRGCN_<dataset name>.yaml` or `tdgcn_<dataset name>.yaml` in `configs/comp`

        - Training scripts are corresponded to `scripts/comparison/comp_CTRGCN.sh` and `scripts/comparison/comp_tdgcn.sh`. The steps are:

            1. Change the boolean value of `bone` & `vel` in the yaml file. Those are label `# modify`. The stream are:

                |`bone`|`vel`|Stream|
                |------|-----|------|
                | False|False|Joint|
                | False| True| Joint Velocity|
                | True |False| Bone|
                | True | True| Bone Velocity|
            
            2. Conduct the bash scripts:

                ```bash
                bash scripts/comparison/comp_<Model Name>.sh <GPU device ID> 
                ```

            3. Go back to **Step 1** until you have conduct the four streams training.

            * For example, when you want to train **CTR_GCN** in **PetAction** in one fold(not all five folds), then you should first modify the boolean value of `bone` = False & `vel` = False in `configs/comp/CTRGCN_PetAction.yaml`, and then conduct `bash scripts/comparison/comp_CTRGCN.sh` for training **Joint Stream**. Afterwards, you could modify the boolean value of `bone` = False & `vel` = True, and then conduct `bash scripts/comparison/comp_CTRGCN.sh` for training **Joint Velocity Stream**, and so on.
        
        - When successfully conducting training, the folder for training would be created under `runs` folder, and the naming of the folder would be `comp_<model name>_<fold num>_<is bone?>_<is velocity>_<time>`.For example, `comp_CTRGCN_4_True_True_20251115123334` means conduct training **Bone Veocity** Stream for **CTRGCN** on fold 4.(sorry but, we don't additionally add dataset in the folder name) 

    2. **InfoGCN**

        - You should conduct 

            ```bash
            CUDA_VISIBLE_DEVICES=<GPU device ID> python comparison/infogcn/main.py --use_vel <True or False> --mode <joint or bone> --fold_num <fold_num> # PetAction

            CUDA_VISIBLE_DEVICES=<GPU device ID> python comparison/infogcn/main.py --use_vel <True or False> --mode <joint or bone> --dataset KABR --num_class 8 # KABR
            ```
        
        - Similar to **CTR-GCN** or **TD-GCN**, you should conduct four times while each time has a different setting. For example, if you want to train **InfoGCN** in **KABR**, you should conduct `python comparison/infogcn/main.py --use_vel False --mode joint --dataset KABR --num_class 8` & `python comparison/infogcn/main.py --use_vel True --mode joint --dataset KABR --num_class 8` and so on.

        - When successfully conducting training, the folder for training would be created under `runs` folder, and the naming of the folder would be `comp_infogcn_<fold num>_<mode>(_vel)_<time>`.For example, `comp_infogcn_0_bone_vel_20251116012407` means conduct training **Bone Veocity** Stream on fold 0, and `comp_infogcn_4_joint_20251116012617` means conduct training **Joint** Stream on fold 4.

## Test/Ensemble

- Since training is splitted to two types of training: Main & Others(for **CTR-GCN**, **InfoGCN**, **TD-GCN**), we would explain separately.

#### Main

- The main file is `ensemble.py`. There are 

- There are two ways to test the result: **individual-fold** or **five-fold** testing

    1. **Individual-fold**: You could refer to `# for individual` part for deciding which streams you want to add. For example, if I want to check the results of **Joint** and **Local Flow** Strreams, then I could conduct 

        ```bash
        CUDA_VISIBLE_DEVICES=<GPU device ID> python ensemble.py --joints_stream_checkpoint_dir <training dir of joints stream> --local_flow_stream_checkpoint_dir <training dir of local flow stream> 
        ```

        * The `<training dir>` would be complete directory path, e.g.`"runs/exp: joints stream_20251130_015542"`
    
    2. **Five-fold**: You could refer to `# for 5-fold validation` part for deciding which streams you want to add. However, quite different to the **Individual-fold**, you have to follow the rules storing the directories. More specificity, the directories containing weights should be placed in `models/self_training`. For more details, you could trace the first time of `if args.five_fold_val:` appearing and read the below codes of it. For the best result of the paper, you could conduct:

        ```bash 
        python ensemble.py --five_fold_val --joints_stream --local_flow_stream --i3dgcn_stream
        ```

#### Others

- Sorry! Since no additional code has been written to simplify the 5-fold split process, please run the evaluation for the five folds sequentially, and then take the average of the five accuracies obtained. 

- This is for model **CTR-GCN**, **InfoGCN**, **TD-GCN** training. There are two ways to test/ensemble:

    1. **CTR-GCN**, **TD-GCN**: Conduct the following command

        ```bash
        bash scripts/comparison/ensemble_<model name>.sh <GPU device ID> <Fold num> <weights dir of joint> <weights dir of joint vel> <weights dir of bone> <weights dir of bone vel>
        ```

        * For example, `bash scripts/comparison/ensemble_CTRGCN.sh 4 4 models/self_training/4/supplement/ctrgcn/joint models/self_training/4/supplement/ctrgcn/joint_vel models/self_training/4/supplement/ctrgcn/bone models/self_training/4/supplement/ctrgcn/bone_vel`
    
    2. **InfoGCN**: Conduct the following command

        ```bash
        CUDA_VISIBLE_DEVICES=<GPU device ID> python -m comparison.infogcn.ensemble --dataset PetAction --position_ckpts <pkls of joint and bone> --motion_ckpts <pkls of joint vel and bone vel> --fold_num <fold num> # for PetAction
        
        CUDA_VISIBLE_DEVICES=<GPU device ID> python -m comparison.infogcn.ensemble --dataset KABR --position_ckpts <pkls of joint and bone> --motion_ckpts <pkls of joint vel and bone vel> # for KABR
        ```

        * For example, `CUDA_VISIBLE_DEVICES=0 python -m comparison.infogcn.ensemble --dataset PetAction --position_ckpts models/self_training/4/supplement/infogcn/bone/best_score.pkl models/self_training/4/supplement/infogcn/joint/best_score.pkl --motion_ckpts models/self_training/4/supplement/infogcn/bone_vel/best_score.pkl models/self_training/4/supplement/infogcn/joint_vel/best_score.pkl --fold_num 4`
    
## Prediction



        
