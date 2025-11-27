#import "utils/general-utils.typ": * 
#import "template/styles.typ": *

#set document(title: "Thesis Proposal") // Note: this title is visible in the PDF viewer

#show: styles

#set align(center)
#text(
  heading("A Master Thesis Proposal", numbering: none, outlined: false), size: 1.15em
)
#v(14pt)

#text("MSc Computer Science
Priyanka Ahuja
November 2025")
\


#line(length: 100%, stroke: gray)

#set align(left)
#set heading(numbering: "1.")


= Introduction

Electroencephalographic data is a very high-dimensional, complex, and noisy time-series record of 
neural activity. These signals contain meaningful information about how the brain states change over 
time, but their complexity makes it a challenge to analyze and visualize them directly. The information 
contained within this massive dataset is encoded in the continuous temporal flow of brain states. Such 
high-dimensionality and complexity, including the various features, present a fundamental barrier to 
robust visualization and analysis. To study and identify the patterns of neural activity, also known as manifolds, dimensionality reduction methods are used to simplify the data while keeping the important 
information about how brain states evolve over time #cite(<Manifold>).
\
More recent approaches, such as PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding), T-PHATE (Temporal-PHATE), and CEBRA.AI (Contrastive Embedding for Behavior and 
Representation Analysis) are designed to better preserve the temporal and structural relationships in neural data. In this research, we will compare these advanced, time-aware algorithms along with 
standard dimensionality reduction approaches like PCA (Principal Component Analysis) and t-SNE. 
The goal is to identify which method can provide the most accurate data of brain states.

== Motivation
Traditional dimensionality reduction methods such as Principal Component Analysis (PCA), t-distributed Stochastic Neighbor Embedding (t-SNE), and Uniform Manifold Approximation and Projection (UMAP) have limitations when used for EEG data visualization. Since EEG is a time-series dataset, the most important information lies in the continuous trajectory of brain states. But these methods neglect the time-series information, resulting in disconnected and irrelevant clusters.
PHATE, T-PHATE and CEBRA.AI are recent advanced time-focused algorithms. They reveal transitions and temporal patterns of brain activity, such as the existence of ring or point attractors in brain states that are otherwise not shown in traditional methods.
\

#table(
columns: 2,

[*Time Agnostic Methods*], [*Time Preserving Methods*], 
[PCA],[TPHATE], 
[UMAP], [CEBRA.AI],
[PHATE], [BCNE], 
[],[Dynamic t-SNE],

)
Table 1: Time Agnostic vs Time Preserving   Methods 

== Related work

- PHATE is a dimensionality reduction technique designed for high-dimensional data where the preservation of continuous developmental trajectories is important #cite(<PHATE>). It was demonstrated on artificial and biological datasets like human germ-layer differentiation. PHATE’s methodology is based on diffusion geometry to focus on global structure and effectively denoise the data. The process first captures local similarities between data points in high-dimensional data; it then uses a diffusion process to transform the local similarities into probabilities that measure the probability of transitioning from one data point to another to find the relative path and transition probabilities, which smooths out noise and reveals the continuous path (geodesic distance) along the manifold. The final embedding is then based on potential distance information into low dimensions for the visualization. The performance was validated based on the DeMAP metric. The paper concludes that PHATE captures the true structure of high-dimensional data more accurately than traditional visualization methods like t-SNE and UMAP.
\
-  T-PHATE is a technique in manifold learning for analyzing time-series brain activity data, such as fMRI #cite(<TPHATE>). T-PHATE is built on PHATE but has a temporal view by data’s autocorrelation structure, indicating how a signal is related to its past timestamp. It uses an approach of dual-view diffusion: one based on the patterns of brain activity (the PHATE view) and one based on the data’s autocorrelation structure (the temporal view). It calculates the similarity between local time points in order to link nearby data points along the temporal sequence. The paper concludes that T-PHATE shows ordered brain state time trajectories by preserving the temporal continuity, which is otherwise lost.
\
-  CEBRA is a supervised (hypothesis-driven) and self-supervised (discovery-driven) learning algorithm that can be used for decoding and generating low latent space maps #cite(<cebra>). The algorithm defines “positive” (similar) and “negative” (dissimilar) points between neural activity points based on time or behavioral labels. A deep neural network is trained using a contrastive loss function. The network then learns the low-dimensional embedding by continuously minimizing the loss function, pulling “positive” pairs closer together and “negative” pairs farther apart. The paper concludes that CEBRA produces consistent embeddings, meaning the low-dimensional map is stable and reliable, and the resulting manifolds exhibited high decoding accuracy for both time and behavioral labels.
\
- The PHATE algorithm is also applied on the EEG dataset containing information about groups: schizophrenia (SZ) patients, Clinical High-Risk (CHR) (who show early symptoms), and Healthy Relatives (RSs) (who have a genetic risk) to detect the early stages of SZ disorder #cite(<schizophrenia>). The methodology starts with the initial dimensionality reduction of the EEG data using Principal Component Analysis (PCA) and then use this cleaned data in the PHATE algorithm. PHATE calculates the 3 dimensions that capture the most variation: PHATE1, PHATE2, PHATE3 and compares them among different groups. The study concludes that PHATE3 values are related to the positive and negative symptoms of SZ and can be used as the identifier.  It successfully differentiates between the healthy group and the CHR group, hence distinguishing who is most likely to progress to SZ.
\
- Techniques like UMAP, t-SNE, and PHATE treat brain activity as isolated snapshots, failing to generate the continuous temporal flow of information. A framework, Brain-dynamic Convolutional-Network-based Embedding (BCNE), has been proposed to visualize and understand the fMRI and EEG data #cite(<BCNE>).
  BCNE methodology starts with temporal processing, focusing on temporal dependencies by calculating an autocorrelation-based affinity matrix. Then it moves to the spatial-channel modeling and image construction phase that calculates pairwise channel relationships and reorganizes the channel responses into a 2D image for each timepoint. These images are then processed through a convolutional neural network (CNN) that learns a low-dimensional embedding by minimizing the KL divergence between the high-dimensional and low-dimensional probability matrices.
  A recursive refinement is implemented to improve the trajectory outcome by replacing the initial high-dimensional matrices with feature vectors extracted from the dense layers of the CNN.
  
  The study states that BCNE generates clear time trajectories for visualization of brain activity in comparison to PCA, UMAP, PHATE, T-PHATE and CEBRA.
\
- T-SNE dimensionality reduction works by constructing a probability distribution over pairs of high-dimensional data points in such a way that similar points are assigned a higher probability while dissimilar points are assigned a lower probability. Then, t-SNE maps a similar probability distribution in the low-dimensional map, and it minimizes the Kullback–Leibler divergence between the two distributions with respect to the locations of the points in the map. T-SNE thus arranges points in the low-dimensional map such that their spatial relationships reflect the similarity structure present in the original high-dimensional data.
  But comparing each data point with every other data point will take a lot of time to compute. Hence, Barnes-Hut t-SNE method #cite(<barnes>) is introduced for faster computation by grouping distant points and treating each group as a single aggregated point, instead of treating the points individually. This drastically reduces the number of pairwise interactions that need to be computed. To do this grouping, the map is divided quad-trees, so the algorithm can quickly decide which groups are distant. This approach can be applied to EEG data visualization, where large amounts of high-dimensional time-series data are transformed into low-dimensional maps.
  \
- Using t-SNE on time-series data can result in visualizations that introduce unnecessary variability, which does not accurately reflect real patterns of change. Dynamic t-SNE  #cite(<dynamictsne>) is an approach that is an adaptation of t-SNE, a controllable trade-off between temporal coherence for accurate representation of the data's structure at the current time step and projection reliability to preserve data structure at a particular time step.  It works by finding a map that is both an accurate representation of the data currently (projection reliability) and one that is smoothly connected to the map from the previous time step (temporal coherence). The dynamic t-SNE cost tries to preserve the neighborhoods for each time step but also penalizes each point for unnecessarily moving between time steps. Hence, a hyperparameter is used as a penalty to choose how much to prioritize stability or smoothness over the projections. The dynamic t-SNE could be used to show the ERP components visualisation over time.
 

= Planned Project
== Research Questions

- What is the best way to assess dimensionality reduction methods for EEG data (Table 1)? 

- Which dimensionality reduction method produces the most accurate low-dimensional manifold for revealing the temporal state of EEG data? 

== Novelty of work 
- We systematically evaluate time-preserving vs. time-agnostic dimensionality reduction methods as mentioned in Table 1, specifically for simulated ERP data, which has not been thoroughly assessed in prior research.
- We demonstrate how different algorithms influence the interpretability of ERP components.
    

== Goals
To assess data simulation using the UnfoldSim package in Julia and use Python packages to implement methods (see Table 1).

=== Main Goals <mainGoals>
#v(0.3em)
#set enum(numbering: "A.")


#[
  #show figure: set align(left) 
   + #goal("Simulate EEG data using UnfoldSim.jl.") <goal1>  
  + #goal("Implement dimensionality reduction and visualization techniques on simulated EEG data in Python.") <goal2>  
  
  + #goal("Understand and compare the methods based on the ability to  preserve the structure of data.") <goal3>  
  + #goal("Review the research papers that use PHATE, T-PHATE, Cebra.AI, BCNE and various methods for EEG datasets (see Table 1).") <goa14>  
  
  + #goal("Discover hidden patterns in each method and state which method provides the clear low-dimensional visualization.") <goal5>
  + #goal("Documenting the steps and procedure for easy understanding.") <goal6>
]




== Approach <approach>
*Phase 1 *
- Data Simulation: Use UnfoldSim.jl, a Julia package for simulating  time series data #cite(<unfoldsim>) focusing on EEG and event-related potentials (ERPs). 
- Simulate EEG data with 32 channels using the predef_eeg function, generating realistic signals based on 4 ERP components: N170, P300, N1, and P1.
- Simulate 5 trials for 2 conditions and deactivate components one by one to see the differences and effects on the data.
- Expected geometry after dimensionality reduction (PHATE, TPHATE, BCNE): Branch trajectory structure for 4 different components such as P300, N170, P1 and N1 starting from same initial time point. There might be space-related differences as there are 32 channels for each ERP component.


*Phase 2*
 - Implementation: Use dual approach:  Julia for data simulation along with Python for its libraries to perform analysis and run algorithms as mentioned in table 1. Validate the low-dimensional maps by using metrics on how accurate the maps are.

- Different studies have used different assessment criteria such as DeMAP, K-nearest neighbor (KNN), Representational Similarity Analysis, Pearson correlation. 

- Understand the metrics performance across different methods and identify the methods for preserving temporal structure in EEG data that can be used to compare the dimensionality reduction methods. 

*Phase 3*
- Visually compare all low-dimensional maps to identify which method gives the most relevant information.
- Apply evaluation metrics:
 - Pearson correlation and Representational Similarity Analysis (RSA) for comparing pairwise distances between embeddings.
 - Trustworthiness and Continuity metrics to understand preservation of local and global structure.
 - Direction accuracy and stage accuracy.

*Phase 4*
- Document the process and their visualization maps along with the findings for EEG data. 

= Plan

- Month 1: Understand the data simulation and dimensionality reduction methods.

- Month 2 and 3: Literature review and implementation will be carried out simultaneously to understand the EEG dataset as techniques like PHATE, T-PHATE and BCNE are recent developments. 
    Refer to insights from the literature to understand the implementation pipeline and analysis procedure and start documenting.
- Month 4 and 5: Test and understand the evaluation metrics.
- Month 6: Review and documentation.




#timeliney.timeline(
  show-grid: true,
  {
    import timeliney: *
      
    headerline(group([*Nov*],[*Dec*],[*Jan*],[*Feb*],[*Mar*],[*Apr*],[*May*],[*June*]))
    
    task("Literature review", (0, 3), style: (stroke: 2pt + gray))
    task("Writing Proposal", (0.5, 1), style: (stroke: 2pt + gray))
    task("Implementation of algorithms", (1, 5.5), style: (stroke: 2pt + gray))
    task("Metrics evaluation", (3, 5.5), style: (stroke: 2pt + gray))
    task("Review and documentation", (4.5,7), style: (stroke: 2pt + gray))
     task("Final submission", (7,8), style: (stroke: 2pt + gray))

    milestone(
      at: 6.5,
      style: (stroke: (dash: "dashed")),
      align(center, [
        *Main goal completion*\
        June 2026
      ])
    )
  }
)




#line(length: 100%, stroke: gray)

#bibliography("refs.bib", style: "american-psychological-association")
