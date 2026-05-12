# **MSc-Thesis:** Evaluating Temporal Dimensionality Reduction Methods for ERP-Structured EEG Data
**Author:** *Benedikt Ehinger*

**Supervisor(s):** *Vladimir Mikheev*

**Year:** *2026*

## Project Description
> Electroencephalographic data is a very high-dimensional, complex, and noisy time-series record of 
neural activity. These signals contain meaningful information about how the brain states change over 
time, but their complexity makes it a challenge to analyze and visualize them directly. The information 
contained within this massive dataset is encoded in the continuous temporal flow of brain states. Such 
high-dimensionality and complexity, including the various features, present a fundamental barrier to 
robust visualization and analysis. 

To study and identify the patterns of neural activity, also known as 
manifolds, dimensionality reduction methods are used to simplify the data while keeping the important 
information about how brain states evolve over time (Perich et al., 2025).
More recent approaches, such as PHATE (Potential of Heat-diffusion for Affinity-based Transition 
Embedding), T-PHATE (Temporal-PHATE), and BCNE (Brain-dynamic Convolutional-Network-based Embedding) are designed to better preserve the temporal and structural relationships 
in neural data. 

In this research, we will compare these advanced, time-aware algorithms along with 
standard dimensionality reduction approaches like PCA (Principal Component Analysis) and t-SNE. 
The goal is to identify which method can provide the most accurate data of brain states.

## Zotero Library Path
>Please provide the link to the Zotero group here or include a `Bib`-File in the `report` folder

## Instruction for a new student
>Simulate data using Unfoldsim in julia (refer Simulatedata.pynb file)
>Approach: Trial level projection
1. Approach: Trial level projection
The BCNE model is trained on the grand average ERP of all trials to obtain a clean and stable reference trajectory. 
The resulting embedding space is then used for projecting individual trials.

Steps:
1. Include the simulated data file in the data folder.
   a. Data can be simulated in Julia using Unfoldsim package, refer simulateData.jl file.
   b. Replace the csv filename with your file in bcne_train.py
For BCNE (non linear dimensionality reduction):
2. Run python bcne_train.py for training global average of all trials.
3. Run python trial_analysis for projecting individual trials through the trained model.
4. For visualisation purpose, trial_analysis will project 200 trials to have clean visuals and to avoid clutter.
For PCA (linear dimensionality reduction)
5. Run pca.py for comparison.
>
2. Approach: Average By condition


## Overview of Folder Structure 

```
│projectdir          <- Project's main folder. It is initialized as a Git
│                       repository with a reasonable .gitignore file.
│
├── report           <- **Immutable and add-only!**
│   ├── proposal     <- Proposal PDF
│   ├── thesis       <- Final Thesis PDF
│   ├── talks        <- PDFs (and optionally pptx etc) of the Intro,
|   |                   Midterm & Final-Talk
|
├── _research        <- WIP scripts, code, notes, comments,
│   |                   to-dos and anything in an alpha state.
│
├── plots            <- All exported plots go here, best in date folders.
|   |                   Note that to ensure reproducibility it is required that all plots can be
|   |                   recreated using the plotting scripts in the scripts folder.
|
├── notebooks        <- Pluto, Jupyter, Weave or any other mixed media notebooks.*
│
├── scripts          <- Various scripts, e.g. simulations, plotting, analysis,
│   │                   The scripts use the `src` folder for their base code.
│
├── src              <- Source code for use in this project. Contains functions,
│                       structures and modules that are used throughout
│                       the project and in multiple scripts.
│
├── test             <- Folder containing tests for `src`.
│   └── runtests.jl  <- Main test file
│   └── setup.jl     <- Setup test environment
│
├── README.md        <- Top-level README. A fellow student needs to be able to
|   |                   continue your project. Think about her!!
|
├── .gitignore       <- focused on Julia, but some Matlab things as well
│
├── (Manifest.toml)  <- Contains full list of exact package versions used currently.
|── (Project.toml)   <- Main project file, allows activation and installation.
└── (Requirements.txt)<- in case of python project - can also be an anaconda file, MakeFile etc.
                        
```

\*Instead of having a separate *notebooks* folder, you can also delete it and integrate your notebooks in the scripts folder. However, notebooks should always be marked by adding `nb_` in front of the file name.