## Installation

1. Download data

    [Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).
    
    Directory structure: 
    ```shell script
    H3.6m
    |-- S1
    |-- S5
    |-- S6
    |-- ...
    `-- S11
    ```

    [AMASS](https://amass.is.tue.mpg.de/en) from their official website.

    Directory structure:
    ```shell script
    amass
    |-- ACCAD
    |-- BioMotionLab_NTroje
    |-- CMU
    |-- ...
    `-- Transitions_mocap
    ```
    [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website.

    Directory structure: 
    ```shell script
    3dpw
    |-- imageFiles
    |   |-- courtyard_arguing_00
    |   |-- courtyard_backpack_00
    |   |-- ...
    `-- sequenceFiles
        |-- test
        |-- train
        `-- validation
    ```
    Put the all downloaded datasets in ../datasets directory.


2. Create the environment
    ```bash
    conda env create -f environment.yml
    ```


3. Activate the environment
    ```bash
    conda activate dlproject
    ```


4. Submit the train task to GPU with the following command (indicated time necessary to reproduce results)
    ```bash
    cd src/
    ```

    Original STSGCN paper results:
        ```bash
        bsub -n 6 -W 24:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python run.py --data_dir DATA_DIR --dataset DATASET --output_n OUTPUT_N --gen_clip_grad null
        ```
    

    STSGCN + MotionDisc results:
        Amass:
        ```bash
        bsub -n 6 -W 24:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python run.py --data_dir DATA_DIR --dataset amass_3d --output_n OUTPUT_N --gen_clip_grad 10 --use_disc
        ```

        H36M:
        ```bash
        bsub -n 6 -W 24:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python run.py --data_dir DATA_DIR --dataset h36m_3d --output_n OUTPUT_N --use_disc
        ```

    DATA_DIR should be the directory where the datasets are located.
    DATASET should be amass_3d or h36m_3d
    OUTPUT_N should be 10 or 25
    GEN_CLIP_GRAD should be null or 10

Note: This repository borrows code from https://github.com/FraLuca/STSGCN