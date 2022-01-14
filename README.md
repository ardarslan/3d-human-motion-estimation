
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

    Put all the downloaded datasets in ../datasets directory.
2. Create the environment
    ```bash
    conda env create -f environment.yml
    ```
3. Activate the environment
    ```bash
    conda activate dlproject
    ```
4. Get into src folder
    ```bash
    cd src/
    ```
    
    ### Training Scripts
5. Submit the train task to GPU with the following command (indicated time necessary to reproduce results)

    - Original STSGCN + TCN:
    ```bash
    bsub -n 6 -W 24:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py --data_dir DATA_DIR --dataset DATASET --output_n OUTPUT_N
    ```	
    - Simple RNN/GRU/LSTM:
	```bash
	bsub -n 6 -W 24:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py --data_dir DATA_DIR --dataset DATASET --output_n OUTPUT_N --gen_model simple_rnn --recurrent_cell RECURRENT_CELL --gen_clip_grad 1.0 --gen_lr 0.001
	```
    - STSGCNEncoder + AttentionDecoder:
    ```bash
    bsub -n 6 -W 24:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py --data_dir DATA_DIR --dataset DATASET --output_n OUTPUT_N --gen_model stsgcn_transformer --gen_milestones 5 15 25 35 --gen_clip_grad 1.0
    ```
    - STSGCNEncoder + RNNDecoder:
    ```bash
    bsub -n 6 -W 24:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py --data_dir DATA_DIR --dataset DATASET --output_n OUTPUT_N --gen_model rnn_stsE --batch_size 64 --gen_lr 0.001 --early_stop_patience 5 --gen_clip_grad 1.0 --recurrent_cell RECURRENT_CELL
    ```
    - STSGCN + MotionDisc:
        - Amass:
        ```bash
        bsub -n 6 -W 24:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py --data_dir DATA_DIR --dataset amass_3d --output_n OUTPUT_N --gen_model stsgcn --gen_clip_grad 10 --use_disc
        ```
        - H36M:
        ```bash
        bsub -n 6 -W 24:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py --data_dir DATA_DIR --dataset h36m_3d --output_n OUTPUT_N --gen_model stsgcn --use_disc
        ```

    - STSGCNEncoder + RNNDecoder + MotionDisc:
    ```bash
    bsub -n 6 -W 24:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py --data_dir DATA_DIR --dataset DATASET --output_n OUTPUT_N --gen_model rnn_stsE --batch_size 64 --gen_lr 0.001 --early_stop_patience 10 --gen_clip_grad 1.0 --recurrent_cell RECURRENT_CELL --use_disc
    ```
    - STSGCNEncoder + AttentionDecoder + MotionDisc:
    ```bash
    bsub -n 6 -W 24:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py --data_dir DATA_DIR --dataset DATASET --output_n OUTPUT_N --gen_model stsgcn_transformer --gen_milestones 5 15 25 35 --use_disc
    ```
    DATA_DIR should be the directory where the datasets are located

    DATASET should be amass_3d or h36m_3d

    OUTPUT_N should be 10 or 25

    RECURRENT_CELL should be lstm or gru or rnn
   
   ### Testing Scripts
6. Submit the test task to GPU with the following command

    - Original STSGCN + TCN:
    ```bash
    bsub -n 6 -W 4:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python test.py --data_dir DATA_DIR --dataset DATASET --output_n OUTPUT_N --model_loc MODEL_LOCATION
    ```	
    - Simple RNN/GRU/LSTM:
	```bash
	bsub -n 6 -W 4:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python test.py --data_dir DATA_DIR --dataset DATASET --output_n OUTPUT_N --gen_model simple_rnn --recurrent_cell RECURRENT_CELL --model_loc MODEL_LOCATION
	```
    - STSGCNEncoder + AttentionDecoder:
    ```bash
    bsub -n 6 -W 4:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python test.py --data_dir DATA_DIR --dataset DATASET --output_n OUTPUT_N --gen_model stsgcn_transformer --model_loc MODEL_LOCATION
    ```
    - STSGCNEncoder + RNNDecoder:
    ```bash
    bsub -n 6 -W 4:00 -o output -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python test.py --data_dir DATA_DIR --dataset DATASET --output_n OUTPUT_N --gen_model rnn_stsE --batch_size 64 --recurrent_cell RECURRENT_CELL --model_loc MODEL_LOCATION
    ```
    MODEL_LOCATION is the location of stored best_model after training
    
Note: This repository borrows code from https://github.com/FraLuca/STSGCN
