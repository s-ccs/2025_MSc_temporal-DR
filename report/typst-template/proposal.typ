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



== Related work

- PHATE is a dimensionality reduction technique designed for high-dimensional data where the preservation of continuous developmental trajectories is important #cite(<PHATE>). It was demonstrated on artificial and biological datasets like human germ-layer differentiation. PHATE’s methodology is based on diffusion geometry to focus on global structure and effectively denoise the data. The process first captures local similarities between data points in high-dimensional data; it then uses a diffusion process to transform the local similarities into probabilities that measure the probability of transitioning from one data point to another to find the relative path and transition probabilities, which smooths out noise and reveals the continuous path (geodesic distance) along the manifold. The final embedding is then based on potential distance information into low dimensions for the visualization. The performance was validated based on the DeMAP metric. The paper concludes that PHATE captures the true structure of high-dimensional data more accurately than traditional visualization methods like t-SNE and UMAP.
\
-  T-PHATE is a technique in manifold learning for analyzing time-series brain activity data, such as fMRI #cite(<TPHATE>). T-PHATE is built on PHATE but has a temporal view by data’s autocorrelation structure, indicating how a signal is related to its past timestamp. It uses an approach of dual-view diffusion: one based on the patterns of brain activity (the PHATE view) and one based on the data’s autocorrelation structure (the temporal view). It calculates the similarity between local time points in order to link nearby data points along the temporal sequence. The paper concludes that T-PHATE shows ordered brain state time trajectories by preserving the temporal continuity, which is otherwise lost.
\
-  CEBRA is a supervised (hypothesis-driven) and self-supervised (discovery-driven) learning algorithm that can be used for decoding and generating low latent space maps #cite(<cebra>). The algorithm defines “positive” (similar) and “negative” (dissimilar) points between neural activity points based on time or behavioral labels. A deep neural network is trained using a contrastive loss function. The network then learns the low-dimensional embedding by continuously minimizing the loss function, pulling “positive” pairs closer together and “negative” pairs farther apart. The paper concludes that CEBRA produces consistent embeddings, meaning the low-dimensional map is stable and reliable, and the resulting manifolds exhibited high decoding accuracy for both time and behavioral labels.
\
- The PHATE algorithm is also applied on the EEG dataset containing information about groups: schizophrenia (SZ) patients, Clinical High-Risk (CHR) (who show early symptoms), and Healthy Relatives (RSs) (who have a genetic risk) to detect the early stages of SZ disorder #cite(<schizophrenia>). The methodology starts with the initial dimensionality reduction of the EEG data using Principal Component Analysis (PCA) and then use this cleaned data in the PHATE algorithm. PHATE calculates the 3 dimensions that capture the most variation: PHATE1, PHATE2, PHATE3 and compares them among different groups. The study concludes that PHATE3 values are related to the positive and negative symptoms of SZ and can be used as the identifier.  It successfully differentiates between the healthy group and the CHR group, hence distinguishing who is most likely to progress to SZ.
\
- Techniques like UMAP, t-SNE, and PHATE treat brain activity as isolated snapshots, failing to generate the continuous temporal flow of information. A framework, Brain-dynamic Convolutional-Network-based Embedding (BCNE), is proposed to visualize and understand fMRI and EEG data #cite(<BCNE>).
  It combines the strength of T-PHATE in temporal signal processing with deep neural network-based techniques, CEBRA to generate 2D maps that can be used in the detection and interpretation of cognitive and behavioral patterns.
  The study states that BCNE generates clear time trajectories for visualization of brain activity in comparison to PCA, UMAP, PHATE, T-PHATE and CEBRA.

= Planned Project
== Research Questions

- What is the best way to assess dimensionality reduction methods for EEG data among PCA, t-SNE, UMAP, PHATE, T-PHATE, and CEBRA.AI  ? 

- Which dimensionality reduction method produces the most accurate low-dimensional manifold for revealing the temporal state of EEG data? 
Novelty of work: Test the most advanced, time-focused algorithms: PHATE, T-PHATE and CEBRA.AI on EEG data against standard, time-ignoring methods like t-SNE and UMAP.
    

== Goals
To assess data simulation using the UnfoldSim package in Julia and use Python packages to implement PHATE, T-PHATE and CEBRA.AI.

=== Main Goals <mainGoals>
#v(0.3em)
#set enum(numbering: "A.")


#[
  #show figure: set align(left) 
   + #goal("Simulate EEG data using UnfoldSim.jl.") <goal1>  
  + #goal("Implement dimensionality reduction and visualization techniques on simulated EEG data in Python.") <goal2>  
  
  + #goal("To understand and compare the methods based on the ability to  preserve the structure of data.") <goal3>  
  + #goal("To review the research papers that use PHATE, T-PHATE and Cebra.AI methods for EEG datasets.") <goa14>  
  
  + #goal("To discover hidden patterns in each method and state which method provides the clear low-dimensional visualization.") <goal5>
  + #goal("Documenting the steps and procedure for easy understanding.") <goal6>
]


=== Stretch Goals <stretchGoals>
#v(0.3em)
#set enum(numbering: "A.", start: 1) // continue the numbering from where the main goals left off. Adjust `start` depending on how many main goals you have.
#[
  #show figure: set align(left) 
  + #goal("Understand and implement Brain-dynamic Convolutional-Network-based Embedding (BCNE).") <goal7>
] 

== Approach <approach>
*Phase 1 *
- Data Simulation: Use UnfoldSim.jl, a Julia package for simulating  timeseries data #cite(<unfoldsim>) focusing on EEG and event-related potentials (ERPs). 
\
*Phase 2*
 - Implementation: We will use dual approach,  Julia for data simulation along with Python for its libraries to perform analysis and run algorithms: PCA, t-SNE, UMAP, PHATE, T-PHATE and Cebra.AI and validate the low-dimensional maps by using metrics on how accurate the maps are.

- Different maps might have different metrics such as DeMAP, K-nearest neighbor (KNN) , support vector machine (SVM) and Spearman correlation. 

- Understand the metrics performance across different methods and identify the best method for preserving temporal structure in EEG data. 
\
*Phase 3*
- Compare all low-dimensional maps to identify which method gives the most relevant information.
\
*Phase 4*
- Document the process and their visualization maps along with the findings for EEG data. 
\

= Plan

- Literature review and implementation will be carried out simultaneously to understand the EEG dataset as techniques like PHATE, T-PHATE are recent developments. 

- Refer to insights from the literature to understand the implementation pipeline and analysis procedure.



#timeliney.timeline(
  show-grid: true,
  {
    import timeliney: *
      
    headerline(group([*Nov*],[*Dec*],[*Jan*],[*Feb*],[*Mar*],[*Apr*],[*May*]))
    
    task("Literature review", (0, 3), style: (stroke: 2pt + gray))
    task("Writing Proposal", (0.5, 1), style: (stroke: 2pt + gray))
    task("Thesis work ", (1, 5.5), style: (stroke: 2pt + gray))
    task("Review", (4.5,7), style: (stroke: 2pt + gray))

    milestone(
      at: 6.5,
      style: (stroke: (dash: "dashed")),
      align(center, [
        *Main goal completion*\
        May 2025
      ])
    )
  }
)




#line(length: 100%, stroke: gray)

#bibliography("refs.bib", style: "american-psychological-association")
