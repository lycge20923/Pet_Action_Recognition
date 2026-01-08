# Pet Action Recognition

This is the project for **A Multi-Stream Framework Integrating Joint and Optical-Flow Representations for Pet Action Recognition**

## Dataset Preparation

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
                <td>⭕</td>
                <td>⭕</td>
                <td>⭕</td>
                <td>⭕</td>
                <td></td>
            </tr>
            <tr>
                <td>❌</td>
                <td>❌</td>
                <td>❌</td>
                <td>❌</td>
                <td>❌</td>
                <td>⭕</td>
                <td>⭕</td>
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



<!-- 
This project aims to recognize and classify various actions performed by pets (i.e., cats and dogs) from video footage using deep learning techniques. It is developed as part of a graduation thesis. The system can identify normal and abnormal actions, such as **Walking**, **Running**, **Seizures**, ... 
<!-- TODO: Add the table to list the actions -->
<!-- TODO: Push the paper finally -->
<!-- TODO: add some visualized videos on it -->

## Table of Contents (目錄 - 可選但推薦)
* [Setup Environment](#set-up-environment)
* [Training / Validation](#trainingvalidation)
* [Prediction](#prediction)

## Set Up Environment

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

## Training/Validation

### Download Dataset 

* To download the necessary dataset for this project, run the following command from the root directory of the project

    ```
    python -m src.data_processing.download
    ```

* Important Notes

    * This download process can take several hours to complete. It is highly recommended to run this command within a persistent terminal session (e.g., using tmux or screen) to ensure it continues running even if your connection drops.

    * During the process, the script might prompt you to enter a verification code or an API key. Please follow the on-screen instructions carefully.
 
### Segment Dataset

* After downloading the full videos, this step extracts the specific annotated action segments, conduct:

    ```
    python -m src.data_processing.segment
    ```

* Important Notes:
    
    * Processing Time: Depending on the total duration of the videos and the number of segments to be extracted, this process might also take a significant amount of time.

### Video Stabilization(Optional)

* In this step, we would refer [vidgear](https://github.com/abhiTronix/vidgear) to conduct video stabilization:

    ```
    python -m src.data_processing.stabilization
    ```

* If you don't conduct video stabilization, make sure to set ```skip_stabilization = True``` in ```src/utils/cli_args.py```

### Obtain Information/Feature(keypoints and optical flows)

* In this step, we would refer the following two repositories to extract keypoints and optical flows of the segmented videos:

    1. [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose): extract skeleton information. 

    2. [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT): extract optical flow information.

    ```
    python -m src.data_processing.feature_extraction
    ```

### Train-Val Split

* In this step, it reads annotations, performs a 5-fold split by source video, samples fixed-size frame windows, balances class samples, and outputs train/validation JSON files. Conduct:

    ```
    python -m src.data_processing.train_split
    ```

### Training
* Finally, we could start to train. To utilize ```wandb``` to help us to find the best parameters, please follow the below steps: 

    1. Go check ```src/utils/cli_args.py``` to see the default values. If you want to change the values, it is recommended not to directly modify the values in it. Instead, you should write a config file like any files in ```configs```, and specify which parameters you want to change. For example, to disable ```add_contrastive_loss```, you could write:

        ```
        ...
        parameters:
          add_contrastive_loss: # this is the parameters you want to change
            values: [False] # or you could specifiy multiple values like [True, False]
        ```

    2. Conduct the following commands

        ```
        wandb sweep configs/sweep_config.yaml # this would include data augmentation and no data augmentation
        ```

    3. Then it would show a command like ```wandb agent <path>```, copy and run it 

    4. After finishing training(or you terminated on the way), you could go to check ```run``` directory and find the info of arguments(```.json```, ```.yaml```) and checkpoint (```.pth```) in it 

## Prediction

* You could use the following command to make predictions on a video, it would output the intermediate results(including optical flows and keypoints) and the final results:

    ```
    python predict.py --input_path <input video path> --checkpoint_dir <checkpoint dir>
    ```

* Important Notes

    * When finishing the prediction, you could find the results in the ```output``` directory -->