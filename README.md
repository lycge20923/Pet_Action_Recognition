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

* Install related packages: ```pip install -r requirements.txt```

## Training/Validation

### Download Dataset 

* To download the necessary dataset for this project, run the following command from the root directory of the project

    ```
    python -m src.data_processing.download_dataset
    ```

* Important Notes

    * This download process can take several hours to complete. It is highly recommended to run this command within a persistent terminal session (e.g., using tmux or screen) to ensure it continues running even if your connection drops.
    
    * During the process, the script might prompt you to enter a verification code or an API key. Please follow the on-screen instructions carefully.
 

## Prediction

