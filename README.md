
# ECOLE: Learning to call copy number variants on whole exome sequencing data



> ECOLE is a deep learning based software that performs CNV call predictions on WES data using read depth sequences.

> The manuscript can be found here:  <a href="https://www.biorxiv.org/content/10.1101/2022.11.17.516880v1" target="_blank">**ECOLE: Learning to call copy number variants on whole exome sequencing data**</a>

> The repository with the necessary data and scripts to reproduce the results in the paper can be found here: <a href="https://zenodo.org/record/7317266#.Y3F0jS8w1hE" target="_blank">**ECOLE results reproduction**</a>

> <a href="https://en.wikipedia.org/wiki/Deep_learning" target="_blank">**Deep Learning**</a>, <a href="https://en.wikipedia.org/wiki/Copy-number_variation" target="_blank">**Copy Number Variation**</a>, <a href="https://en.wikipedia.org/wiki/Exome_sequencing" target="_blank">**Whole Exome Sequencing**</a>


---

## Authors

Berk Mandiracioglu, Furkan Ozden, Gun Kaynar, M. Alper Yilmaz, Can Alkan, A. Ercument Cicek

---

## Questions & comments 

[firstauthorname].[firstauthorsurname]@epfl.ch

---



## Table of Contents 

> Warning: Please note that ECOLE software is completely free for academic usage. However it is licenced for commercial usage. Please first refer to the [License](#license) section for more info.

- [Installation](#installation)
- [Features](#features)
- [Instructions Manual](#instructions-manual)
- [Usage Examples](#usage-examples)
- [Citations](#citations)
- [License](#license)


---

## Installation

- ECOLE is a python3 script and it is easy to run after the required packages are installed.

### Requirements

For easy requirement handling, you can use ECOLE_environment.yml files to initialize conda environment with requirements installed:

```shell
$ conda env create --name ecole_env -f ECOLE_environment.yml
$ conda activate ecole_env
```

Note that the provided environment yml file is for Linux systems. For MacOS users, the corresponding versions of the packages might need to be changed.
---

## Features

- ECOLE provides GPU support optionally. See [GPU Support](#gpu-support) section.


## Instructions Manual for ECOLE
Important notice: Please call the ECOLE_call.py script from the scripts directory.

### Required Arguments

#### -m, --model
- Pretrained models of the paper, one of the following: (1) ecole, (2) ecole-ft-expert, (3) ecole-ft-somatic. 


#### -bs, --batch_size
- Batch size to be used to perform CNV call on the samples. 

#### -i, --input
- Relative or direct path for are the processed WES samples, including read depth data. 

#### -o, --output
- Relative or direct output directory path to write ECOLE output file.

### -c, --cnv
- Level of resolution you desire, choose one of the options: (1) exonlevel, (2) merged.


### -n, --normalize
- Relative or direct path for mean&std stats of read depth values to normalize. These values are obtained precalculated from the training dataset before the pretraining.


### Optional Arguments

#### -g, --gpu
- Set to PCI BUS ID of the gpu in your system.
- You can check, PCI BUS IDs of the gpus in your system with various ways. Using gpustat tool check IDs of the gpus in your system like below:

#### -v, --version
-Check the version of ECOLE.

#### -h, --help
-See help page.



## Usage Example

> Usage of ECOLE is very simple!

### Step-0: Install conda package management

- This project uses conda package management software to create virtual environment and facilitate reproducability.

- For Linux users:
 - Please take a look at the <a href="https://repo.anaconda.com/archive/" target="_blank">**Anaconda repo archive page**</a>, and select an appropriate version that you'd like to install.
 - Replace this `Anaconda3-version.num-Linux-x86_64.sh` with your choice

```shell
$ wget -c https://repo.continuum.io/archive/Anaconda3-vers.num-Linux-x86_64.sh
$ bash Anaconda3-version.num-Linux-x86_64.sh
```


### Step-1: Set Up your environment.

- It is important to set up the conda environment which includes the necessary dependencies.
- Please run the following lines to create and activate the environment:

```shell
$ conda env create --name ecole_env -f ECOLE_environment.yml
$ conda activate ecole_env
```

### Step-2: Run the preprocessing script.

- It is necessary to perform preprocessing on WES data samples to obtain read depth and other meta data and make them ready for CNV calling.
- Please run the following line:

```shell
$ source preprocess_samples.sh
```

### Step-3: Run ECOLE on data obtained in Step-2

- Here, we demonstrate an example to run ECOLE on gpu device 0, and obtain exon-level CNV call.
- Please run the following script:

```shell
$ source ecole_call.sh
```
 You can change the argument parameters within the script to run it on cpu and/or to obtain merged CNV calls.

### Output file of ECOLE
- At the end of the CNV calling procedure, ECOLE will write its output file to the directory given with -o option. In this tutorial it is ./ecole_calls_output
- Output file of ECOLE is a tab-delimited .bed like format. 
- Columns in the output file of ECOLE are the following with order: 1. Sample Name, 2. Chromosome, 3. CNV Start Index, 4. CNV End Index, 5. ECOLE Prediction 
- Following figure is an example of ECOLE output file.


<img src="./example_output.png"   class="center">

## Instructions Manual for Finetuning ECOLE
Important notice: Please call the ECOLE_finetune.py script from the scripts directory.

### Required Arguments

#### -bs, --batch_size
- Batch size to be used to perform CNV call on the samples. 

#### -i, --input
- Relative or direct path for are the processed WES samples, including read depth data. 

#### -o, --output
- Relative or direct output directory path to write ECOLE output file.

### -n, --normalize
- Relative or direct path for mean&std stats of read depth values to normalize. These values are obtained precalculated from the training dataset before the pretraining.

### -e, --epochs
- The number of epochs the finetuning will be performed.

### -lr, --learning_rate
- The learning rate to be used in finetuning

### -lmp, --load_model_path
- The path for the pretrained model weights to be loaded for finetuning

### Optional Arguments

#### -g, --gpu
- Set to PCI BUS ID of the gpu in your system.
- You can check, PCI BUS IDs of the gpus in your system with various ways. Using gpustat tool check IDs of the gpus in your system like below:

#### -v, --version
-Check the version of ECOLE.

#### -h, --help
-See help page.


## Finetune Example

> We provide an ECOLE Finetuning example with WES sample of NA12891 using only chromosome 21.
> Step-0 and Step-1 are the same as the ECOLE call example.

### Step-0: Install conda package management

- This project uses conda package management software to create virtual environment and facilitate reproducability.

- For Linux users:
 - Please take a look at the <a href="https://repo.anaconda.com/archive/" target="_blank">**Anaconda repo archive page**</a>, and select an appropriate version that you'd like to install.
 - Replace this `Anaconda3-version.num-Linux-x86_64.sh` with your choice

```shell
$ wget -c https://repo.continuum.io/archive/Anaconda3-vers.num-Linux-x86_64.sh
$ bash Anaconda3-version.num-Linux-x86_64.sh
```


### Step-1: Set Up your environment.

- It is important to set up the conda environment which includes the necessary dependencies.
- Please run the following lines to create and activate the environment:

```shell
$ conda env create --name ecole_env -f ECOLE_environment.yml
$ conda activate ecole_env
```

### Step-2: Run the preprocessing script for preparing the samples for finetuning.

- It is necessary to perform preprocessing on WES data samples to obtain read depth and other meta data and make them ready for ECOLE finetuning.
- ECOLE Finetuning requires .bam and ground truth calls as provided under /finetune_example_data. Please see the below image for a sample ground truths format.
- <img src="./finetune_ground_truths.png"   class="center">
- Please run the following line:

```shell
$ source finetune_preprocess_samples.sh
```

### Step-3: Start ECOLE Finetuning on data obtained in Step-2

- Here, we demonstrate an example to run ECOLE Finetuning on gpu device 0.
- Please run the following script:

```shell
$ source ecole_finetune.sh
```
 You can change the argument parameters within the script to run it on cpu.

### Output file of ECOLE
- At the end of ECOLE Finetuning, the script will save its model weights file to the directory given with -o option. In this tutorial it is ./ecole_finetuned_model_weights


---



## Citations

---

## License


- **[CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/)**
- Copyright 2022 © ECOLE.
- For commercial usage, please contact.
