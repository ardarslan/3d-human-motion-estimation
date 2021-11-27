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


4. Submit the train/test/visualization task to GPU with the following command (indicated time necessary to reproduce results)
    ```bash
    cd src/
    bsub -n 6 -W 4:00 -o ../output -R "rusage[mem=4096, ngpus_excl_p=1]" python run.py
    ```
