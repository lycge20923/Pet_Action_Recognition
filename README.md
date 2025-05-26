# Pet_Action_Recognition

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

### Obtain Pose Estimation Dataset

* In this step, we would refer [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) to extract skeleton information. Conduct: 

    ```
    python -m src.data_processing.feature_extraction
    ```

### Train-Val Split

* In this step, it reads and filters keypoint annotations, performs a 5-fold split by source video, samples fixed-size frame windows, normalizes keypoints within bounding boxes, balances class samples, and outputs train/validation JSON files. Conduct:

    ```
    python -m src.data_processing.train_split
    ```

### Training
* Finally, we could start to train. To utilize ```wandb``` to help us to find the best parameters, please follow the below steps: 

    1. Conduct the following commands

        ```
        wandb sweep configs/sweep_config_original.yaml # this would include data augmentation and no data augmentation
        ```

    2. Then it would show a command like ```wandb agent <path>```, copy and run it 

## Prediction

