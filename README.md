# **MSc-Thesis:** Evaluating Temporal Dimensionality Reduction Methods for ERP-Structured EEG Data
**Author:** *Priyanka Ahuja*

**Supervisor(s):** *Vladimir Mikheev* , *Benedikt Ehinger*

**Year:** *2026*

## Project Description
> EEG data recorded during cognitive experiments produces short, stimulus-locked epochs (ERPs) that repeat across hundreds of trials. Standard dimensionality reduction methods like PCA and t-SNE treat each time point as independent, grouping them by amplitude similarity rather than temporal order, losing the sequential structure that makes ERP data meaningful.
This project adapts two time-aware methods, T-PHATE and BCNE, to ERP-structured EEG data and evaluates whether encoding temporal autocorrelation into the embedding produces more faithful representations than time-agnostic approaches. ERP components were simulated using UnfoldSim with four components (P100, N170, P300, N400) across categorical conditions and a continuous variable.
Two pipelines are implemented. The first averages trials per condition group and compares all six methods (PCA, t-SNE, UMAP, PHATE, T-PHATE, BCNE) on the clean ERP input. The second trains BCNE on the grand average and projects every individual trial through the fixed coordinate system, enabling trial-level visualisation, continuous effect recovery, and trajectory-based outlier detection without retraining.

## Zotero Library Path
>Please provide the link to the Zotero group here or include a `Bib`-File in the `report` folder

## Instruction for a new student
Simulate data using Unfoldsim in Julia (refer to DataSimulation_UnfoldSim.pynb file). 

>Approach: Average By condition
Data Simulation:
 Include the simulated data file in the data folder.
   a. Data can be simulated in Julia using the UnfoldSim package.
   b. Replace the CSV filename with your file in main.py.
Steps:
1. Run python main.py to train the model for condition-continuous groups.
2. Run python compare_methods.py for comparing the model with PCA, t-SNE, UMAP, PHATE, TPHATE.

Download the zip file in the src in the Average-By-Condition approach (https://github.com/s-ccs/2025_MSc_temporal-DR/tree/main/src) and run python main.py for BCNE training of ERP groups and run python compare_methods.py for comparison across algorithms to get two-dimensional embeddings of PCA, t-SNE, UMAP, PHATE, T-PHATE and BCNE.

>Approach: Trial-level projection
The BCNE model is trained on the grand average ERP of all trials to obtain a clean and stable reference trajectory. 
The resulting embedding space is then used for projecting individual trials.

Data Simulation:
Include the simulated data file in the data folder. (Same as in the above approach, the same file can be used)
   a. Data can be simulated in Julia using Unfoldsim package, refer DataSimulation_UnfoldSim.jl file.
   b. Replace the csv filename with your file in main.py
   
Download the zip file Projection approach in src (https://github.com/s-ccs/2025_MSc_temporal-DR/tree/main/src) 
Steps:
1. Run python main.py for BCNE training of grand average and individual trial analysis
2. Run python pca_projection.py for comparison across with PCA.
3. Run outlier_test.py to test how flat and noise outlier trials appear in the BCNE embedding space



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
