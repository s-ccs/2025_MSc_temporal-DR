// A central place where libraries are imported (or macros are defined)
// which are used within all the chapters:
#import "utils/global.typ": *
#import "utils/caption.typ": dynamic-caption


#let abstract = [Electroencephalographic (EEG) data is a high-dimensional, noisy time series capturing the continuous evolution of neural response patterns. Dimensionality reduction methods such as @PCA and @tSNE have been used to analyze such data; however, they group time points by amplitude similarity rather than temporal order, making them unable to preserve the sequential structure. Existing time-aware approaches, such as @T-PHATE and @BCNE, address this by explicitly encoding temporal autocorrelation into the embedding, yet both have only been demonstrated on continuous recordings such as fMRI and not on epoched #[ERP]-structured @EEG data.

This study addresses this research gap and evaluates time-aware dimensionality reduction 
methods on @ERP structured @EEG data, investigating whether temporal awareness yields more 
meaningful embeddings than time-agnostic approaches. @ERP components were simulated using 
the UnfoldSim package, and @T-PHATE and @BCNE were evaluated along with standard methods 
under two approaches: condition-averaged input and single-trial projection via grand 
average. The results show that @T-PHATE and @BCNE recover temporally ordered trajectories 
with condition-specific divergences, while time-agnostic methods yield fragmented 
embeddings. These findings suggest that time-aware methods offer a more faithful 
representation of @ERP data and can be used for exploratory analysis of complex 
experimental designs.

]

// Fill me with acknowledgments
#let acknowledgements = [I would like to thank my supervisors, Dr. Benedikt Ehinger and Mr. Vladimir Mikheev, for their constant support and guidance throughout my thesis.]


// Declaration regarding own work / AI use: adapted from the guidelines of the Computer Science Department, Faculty 5, Uni-Stuttgart 

#let declaration = [
 #include "declaration.typ"
]

// if you have appendices, add them here
#let appendix = [

= Additional Figures

This section contains additional figures that support the results presented but were not shown to maintain narrative focus.

   #figure(
  image("template/demo/figures/neuromaps_erp.png", width: 60%),
  caption: dynamic-caption(
    [Neuromaps illustrating how channel-correlation patterns evolve across @ERP peak latencies for 64 channels represented by 8 x 8 grid. Red and blue regions reflect changes in channel activity over time.],
    [Neuromaps 8 x 8 grid at @ERP peak latencies.]
  ),
) <fig:second_additional_figure>


#figure(
  image("template/demo/figures/AvgByCondition/2conditions-AvgByCondition/bcne_condition_recursive_2cond.png", width: 60%),
  caption: dynamic-caption(
    [@BCNE condition separation for the two-condition design shown across 
    all four recursive stages m1 to m4, illustrating how separation becomes 
    more compact with each stage.],
    [@BCNE condition separation across recursive stages, two-condition design.]
  ),
) <fig:bcne_condition_recursive_2cond>

#figure(
  image("template/demo/figures/AvgByCondition/4conditions-AvgByCondition/fig2_condition.png", width: 90%),
  caption: dynamic-caption(
    [@BCNE condition separation for the two-condition design shown across 
    all four recursive stages m1 to m4, illustrating how separation becomes 
    more compact with each stage.],
    [Condition-averaged embeddings for the four-condition design.]
  ),
) <fig:fig_4condition>

#figure(
  image("template/demo/figures/AvgByCondition/4conditions-AvgByCondition/fig3_time_car.png", width: 90%),
  caption: dynamic-caption(
    [Points are coloured by time from stimulus onset (0 to 590 ms). @T-PHATE and @BCNE preserve smooth temporal trajectories with temporal order maintained as a continuous path.],
    [Conditional variable and predictor in four-condition design.]
  ),
) <fig:fig3_time_animal_4cond>

The four-condition design evaluated whether the two-condition finding scales to a more complex categorical structure. As shown in @fig:fig_4condition, 
@T-PHATE and @BCNE maintained distinct condition trajectories while preserving temporal organization across all four conditions (@fig:fig3_time_animal_4cond). 

Hence, increasing the number of conditions did not substantially reduce separation quality.


 #include "appendix.typ"

]



// Put your abbreviations/acronyms here.
// 'key' is what you will reference in the typst code
// 'short' is the abbreviation (what will be shown in the pdf on all references except the first)
// 'long' is the full acronym expansion (what will be shown in the first reference of the document)
//
// In the text, call @eeg or @uniS to reference  the shortcode
#let abbreviations = (
  (
    key: "EEG",
    short: "EEG",
    long: "Electroencephalography",
  ),
  (
    key: "ERP",
    short: "ERP",
    long: "Event-related Potentials",
  ),
  (
    key: "uniS",
    short: "UoS",
    long: "University of Stuttgart",
  ),
  (
    key: "PCA",
    short: "PCA",
    long: "Principal component analysis",
  ),
  (
    key: "tSNE",
    short: "tSNE",
    long: "T-distributed Stochastic Neighbor Embedding",
  ),
  (
    key: "UMAP",
    short: "UMAP",
    long: "Uniform Manifold Approximation and Projection",
  ),
  (
    key: "PHATE",
    short: "PHATE",
    long: "Potential of Heat-diffusion for Affinity-based Trajectory Embedding",
  ),
  (
    key: "T-PHATE",
    short: "T-PHATE",
    long: "Temporal-Potential of Heat-diffusion for Affinity-based Trajectory Embedding ",
  ),
   (
    key: "BCNE",
    short: "BCNE",
    long: "Brain-dynamic Convolutional-Network-based Embedding",
  ),
   (
    key: "CEBRA",
    short: "CEBRA",
    long: "Consistent Embeddings through Contrastive Learning",
  ),
    (
    key: "CNN",
    short: "CNN",
    long: "Convolutional Neural Network",
  ),
   (
    key: "KNN",
    short: "KNN",
    long: "K-nearest neighbour",
  ),
   (
    key: "DeMAP",
    short: "DeMAP",
    long: "Denoised Manifold Affinity Preservation",
  ),
  (
    key: "GW",
    short: "GW",
    long: "Gromov-Wasserstein",
  ),
)

#show: thesis.with(
  author: "Priyanka Sanjeevkumar Ahuja",
  title: "Evaluating Temporal Dimensionality Reduction Methods for ERP-Structured EEG Data",
  degree: "Masters",
  faculty: "Faculty of Electrical Engineering and Computer Science
Computational Cognitive Science",
  department: "",
  major: "Computer Science",
  supervisors: (
    (
      title: "Main Supervisor",
      name: "Benedikt Ehinger",
      affiliation: [Computational Cognitive Science \
        Faculty of Electrical Engineering and Computer Science, \
        Department of Computer Science
      ],
    ),
    (
      title: "Second Supervisor",
      name: "Vladimir Mikheev",
      affiliation: [Computational Cognitive Science \
        Faculty of Electrical Engineering and Computer Science, \
        Department of Computer Science
      ],
    ),
  ),
  epigraph: none,
  abstract: abstract,
  appendix: appendix,
  acknowledgements: acknowledgements,
  preface: none,
  figure-index: true,
  table-index: true,
  listing-index: true,
  abbreviations: abbreviations,
  date: datetime(year: 2026, month: 6, day: 3),
  bibliography: bibliography("refs.bib", title: "Bibliography", style: "american-psychological-association"),
  declaration: declaration
  
)

// Code blocks
#codly(
  languages: (
    julia: (
      name: "julia",
      color: rgb("#CE412B"),
    ),
    // NOTE: Hacky, but 'fs' doesn't syntax highlight
    fsi: (
      name: "F#",
      color: rgb("#6a0dad"),
    ),
  ),
)

// If you wish to use lining figures rather than old-style figures, uncomment this line.
// #set text(number-type: "lining")

// import custom utilities
#import "utils/general-utils.typ": *

// Main Content starts here
= Introduction <chp:introduction>

The brain generates electrical signals that can be recorded at the scalp using @EEG. #[@ERP]s are time-locked brain responses to specific events that capture the sequence of neural activity following a stimulus, revealing how the brain processes information over time.

@ERP data has several dimensions: channels, time, conditions, and subjects. As experimental designs become more complex, with multiple conditions, continuous stimulus parameters, and overlapping neural responses, visualizing and interpreting this structure becomes increasingly challenging @mikheev2023art. Dimensionality reduction addresses this problem by compressing the high-dimensional structure into a low, interpretable representation.

Standard dimensionality reduction approaches, such as @PCA and @tSNE, share a common limitation when applied to @ERP data. They treat each time point as an independent observation, without explicitly modeling the relationship between adjacent time points @polivcar. This assumption ignores the most important property of @ERP data, temporal continuity. Such methods embed time points solely based on amplitude similarity, producing fragmented structures in which the chronological flow of brain states is often lost.

Neural activity follows a trajectory in the neural state space, with the structure geometry representing the stages of neural activity @cunningham2014dimensionality. Nonlinear dimensionality reduction methods like @tSNE, @UMAP, and @PHATE typically assume simple underlying manifolds and may not be able to represent complex topologies @chung2021neural. Because these approaches do not encode temporal order, they cannot accurately reconstruct the sequential structure of @ERP data.

Time-aware dimensionality reduction methods address this by incorporating temporal structure directly into the embedding. @T-PHATE and @BCNE methods are two examples that produce temporally ordered embeddings on continuous neural recordings.

== Exploratory analysis for complex EEG designs
Modern EEG experiments increasingly involve multiple categorical conditions, continuous stimulus levels, and overlapping neural responses, making direct visualization of the data difficult @ehinger2019unfold. As experimental complexity increases, standard visualizations such as ERP waveforms and topoplot series typically show only one aspect of the data at a time, and important patterns across multiple conditions and channels can be easily missed.

Dimensionality reduction can support exploratory analysis of multidimensional data structures. Instead of relying on manual selection of which dimensions to display, it can summarise structure across channels, conditions, trials, and time in a single, compact representation. This provides an initial assessment of the data's salient features, guiding subsequent analyses. It also serves as a sanity check to determine whether trials from the same experimental condition are more similar to each other than trials from different conditions. Dimensionality reduction, therefore, acts as a powerful exploratory tool in neural data analysis and can generate the hypotheses that formal statistical tests later may evaluate @cunningham2014dimensionality.


== Research Gap

Time-aware methods such as @T-PHATE and @BCNE have been demonstrated on continuous neural recordings, including fMRI BOLD signals, rat hippocampal spike trains, and macaque motor cortex activity. However, neither method has been applied to @ERP structured @EEG data, which presents a fundamentally different analytical challenge.

@ERP are short, stimulus-locked epochs that are repeated across many trials rather than long continuous time series. Conventional analysis averages across trials within a condition to suppress noise, but this discards the trial-level variability that may carry meaningful information. Adapting these methods to @ERP data, therefore, requires careful handling of the epoch structure and a way to recover information that is otherwise lost in averaging.


== Research questions
RQ1. To what extent do time-aware dimensionality reduction methods (@BCNE and @T-PHATE) better preserve temporal structure in structured EEG data compared to time-agnostic methods (@PCA, @tSNE, @UMAP, @PHATE)?

RQ2. Which metrics provide the most informative assessment of temporal preservation in dimensionality reduction?


= Literature Review 
== Event-Related Potential (ERP) Components 

@ERP waveforms consist of a sequence of positive and negative voltage deflections. Within these waveforms, @ERP components are commonly characterised by their latency, polarity, and scalp distribution @woodman2010brief. They are computed by averaging the continuous @EEG across many trials aligned to a common stimulus onset, to cancel out noise while preserving the stimulus-locked signal. The temporal sequence of @ERP components can provide an indirect measure of the different stages of information processing. The challenge of visualizing these sequential components across multiple conditions and trials encourages the application of dimensionality reduction methods that explicitly preserve temporal order.



== Time-Agnostic Dimensionality Reduction Methods


=== Principal Component Analysis

@PCA is a linear dimensionality reduction technique that finds a low-dimensional representation of high-dimensional data by identifying the directions of maximum variance, known as principal components, along which the data vary the most. Applied to @EEG data, @PCA identifies the directions of maximum variance across channels and time points. The first few components can sometimes align with known @ERP components, making it a common preprocessing step in @EEG analysis @kayser2003optimizing.

However, @PCA has important limitations for the visualization of @ERP trajectories. As a linear method, it cannot identify nonlinear manifold organization in neural recordings. Further, @PCA does not explicitly encode temporal continuity and therefore cannot preserve the sequential order of @ERP states in the embedding space. An important advantage of @PCA is that it can learn a fixed linear mapping from the training data, which can be applied to new observations without recomputation.


=== T-distributed Stochastic Neighbor Embedding

@tSNE is a non-linear dimensionality reduction method that preserves local neighborhood structure. It models pairwise similarities between data points in the high-dimensional space using a Gaussian kernel. It constructs a low-dimensional embedding that preserves these neighborhood relationships by minimizing the Kullback-Leibler divergence between the high-dimensional and low-dimensional similarity distributions @anuragi2024mitigating.

@tSNE has two basic limitations for the @ERP trajectory visualization. First, it optimizes for local neighborhood structure and then tends to reveal global structure to some extent, meaning that nearby clusters in the embedding do not necessarily reflect true distances between neural response patterns @barnes. Second, @tSNE treats each time point as an independent observation, unaware of its sequential order. Time points with similar amplitudes are grouped together regardless of when they occur, so that the sequential flow of brain states is not recoverable from the embedding. Although this local clustering can incidentally separate experimental conditions, it does so without any temporal meaning, reflecting amplitude similarity rather than the progression of neural activity over time @van2008visualizing.


=== Dynamic t-SNE

 Using @tSNE on time-series data can yield visualizations that introduce unnecessary variability and fail to accurately reflect real patterns of change. Dynamic @tSNE is an adaptation of @tSNE that balances temporal coherence for accurate representation of the data’s structure at the current time step with projection reliability to preserve data structure at a particular time step. The dynamic @tSNE has a cost function that tries to preserve neighborhoods at each time step, while penalizing each point for unnecessarily moving between time steps. Hence, a hyperparameter is used as a penalty to choose how much to prioritize stability or smoothness over the projections @Rauber2016VisualizingTD. The dynamic @tSNE could be used to visualize @ERP components over time.

=== Uniform Manifold Approximation and Projection

@UMAP approximates the manifold structure of high-dimensional data by constructing a weighted graph of nearest-neighbor relationships and optimizes a low-dimensional embedding that preserves this graph structure. Compared to @tSNE, @UMAP is computationally faster and better preserves global structure alongside local neighborhoods. However, it shares the fundamental limitation of treating time points as independent observations, with no mechanism for encoding sequential order @anuragi2024mitigating.

Unlike @tSNE, @UMAP provides a transform method that allows new data points to be projected into a previously learned embedding space @sainburg2021parametric.

=== Potential of Heat-diffusion for Affinity-based Trajectory Embedding


@PHATE is a diffusion geometry-based dimensionality reduction method designed to preserve continuous trajectories and branching structure within high-dimensional biological data @refPHATE. It constructs a Markov diffusion operator from local similarities between data points, computes information-geometric potential distances from resulting transition probabilities and embeds these distances into a low-dimensional space. @PHATE has been shown to outperform @tSNE on datasets with branching trajectories, while preserving both local and global structure, as measured by the @DeMAP.


== Time-Aware Dimensionality Reduction Methods

=== Temporal PHATE

@T-PHATE extends @PHATE by adding an explicit temporal view
derived from the autocorrelation structure of the signal.
It is based on a dual-view diffusion method, with the @PHATE view based on brain activity patterns and the temporal view based on the autocorrelation structure of the data.
The first view is the standard
@PHATE diffusion operator, which captures the geometry of brain activity patterns.
The second view is a temporal diffusion operator constructed from the
autocorrelation matrix, which encodes the temporal continuity of the signal. The
two views are combined using a multi-view diffusion framework that integrates
information from both the activity patterns and the temporal structure.

The result is an embedding that preserves both the similarity
structure of brain states and their sequential ordering. @T-PHATE was originally
demonstrated on the Sherlock fMRI dataset.
The authors concluded that @T-PHATE recovered temporally ordered brain-state trajectories from continuous neural recordings @refTPHATE.



=== Brain-dynamic Convolutional-Network-based Embedding


The BCNE method uses a convolutional neural network to analyze time-series patterns of neural activity @refBCNE. It is an unsupervised learning approach where each time point is treated independently as an image for recursive @CNN training. Data undergoes temporal autocorrelation and spatial projection before being fed to the @CNN model.

#[CNN]s have been widely applied to EEG data for feature extraction and classification, with convolutional layers to effectively capture spatial and temporal patterns in neural signals @li2020deep.
The dynamic data are first processed through temporal autocorrelation smoothing, which amplifies the @ERP signal and lowers noise by averaging each time point with its temporal neighbors.

Second, spatial mapping generates a structured two-dimensional representation from the multi-channel signal at each time point. The temporally processed signals are used to estimate the inter-channel interaction matrix, which is then aligned onto a grid using the Gromov-Wasserstein optimal transport. To encode their relationships in the image's spatial context, channels with similar activity are close together in the grid @islam2023cartography.

Third, the image sequence is processed by a convolutional neural network trained without any labels to minimize the KL divergence between pairwise similarity distributions in the high-dimensional image space and the two-dimensional embedding space. To further improve the resultant trajectory of the brain dynamic data, the initial HD probability matrices used for network optimization are replaced by feature vectors extracted from the first dense layer of BCNE.  As iterations increase, this approach uses feature vectors from deeper, dense layers of BCNE to generate progressively refined temporal trajectory representations of dynamic brain data in two-dimensional or 3-dimensional space.

@BCNE was demonstrated on fMRI Sherlock movie, rat hippocampal, and macaque motor cortex datasets, achieving the highest classification accuracy and representational similarity scores among all compared methods @refBCNE. A key property distinguishing @BCNE from @T-PHATE and all other methods evaluated here is that the trained network defines a fixed coordinate system to which new data can be projected without retraining, enabling single-trial analysis and generalization to unseen conditions.

=== Consistent Embeddings through Contrastive Learning (CEBRA)

@CEBRA is a dimensionality reduction approach that combines supervised (hypothesis-driven) and self-supervised (discovery-driven) analysis, as well as the visualization and decoding of brain data. It defines “positive” (similar) and “negative” (dissimilar) sample pairings based on time or behavioral labels, and then trains a deep neural network with a contrastive loss function @cebra. The network learns a low-dimensional embedding by minimizing this loss, which pulls positive pairs closer together and pushes negative pairs apart.

@CEBRA was demonstrated on rat hippocampal, primate motor cortex, and mouse visual cortex recordings. The authors report that it produces consistent, stable embeddings that are reproducible across repeated runs. These embeddings support high-accuracy decoding of behavior.

@CEBRA was not included in the present evaluation but is described here as a closely related contrastive approach to time-aware embedding.


#pagebreak()
= Data Simulation

== UnfoldSim

The #[EEG]-structured @ERP dataset was generated using UnfoldSim @refunfoldsim, a Julia package for simulation of realistic EEG signals.
The simulated data consisted of four ERP components (P100, N170, P300, and N400), simulated using predefined basis functions. Both categorical and continuous (stimulus intensity) effects were generated for data variability across 32 and 64 channels @Harmening_2022.
The resulting data was epoched into trials and transformed into a long-format representation, where each row corresponds to a time point with associated trial, condition and continuous labels.


#figure(
  table(
    columns: (auto, auto),
    inset: 7pt,
    align: (left, left),
    [*Parameter*], [*Value*],
    [Number of channels], [32 (selected from BioSemi 32 montage)],
    
    [Number of trials], [2000],
    [Categorical conditions], [2 (car, face)],
    [Continuous levels], [10 (range −5.0 to +5.0)],
    [Sampling frequency], [200 Hz],
    [Experimental design repetition], [50],
    [Noise level], [1],
    
  ),
  caption: "Example of simulation parameters to generate the ERP dataset."
) <tbl:simulation_params>
#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 7pt,
    align: (left, left, left, left),
    [*Component*], [*Latency*], [*Affected by*], [*Cortical source*],
    [P100], [100 ms], [None (intercept only)], [Right Occipital Pole],
    [N170], [170 ms], [Condition], [Left Postcentral Gyrus],
    [P300], [300 ms], [Continuous levels], [Left Superior Frontal Gyrus],
    [N400], [400 ms], [Condition], [Right Subcallosal Cortex],
  ),
  caption: "Simulated ERP components, their latencies, the experimental factors affecting them, and their assigned cortical sources."
) <tbl:erp_components>

== Simulated ERP Components Example 
#show: codly-init.with()
#figure(

```julia
design = SingleSubjectDesign(
    conditions = Dict(
        :condition => ["car", "face"],
        :continuous => range(-5, 5, length = 10),
    ),
    event_order_function = shuffle,
) |> x -> RepeatDesign(x, 100)

```,
caption: "Julia code defining the experimental design."

)<lst:SimulatedCodeDesign>

#figure(

```julia
p1 = (p100(; sfreq=sfreq), @formula(0 ~ 1),[5.0],Dict(),0)
n1 = (n170(; sfreq=sfreq), @formula(0 ~ 1 + condition),[5.0, 3.0],  Dict(), 0)
p3 = (p300(; sfreq=sfreq), @formula(0 ~ 1 + continuous),  [5.0, 1.0],  Dict(), 0)
n4 = (n400(; sfreq=sfreq), @formula(0 ~ 1 + condition),   [5.0, 5.0],  Dict(), 0)
```,
caption: "Julia code defining ERP components"

)<lst:SimulatedERPComponents>


#figure(
  image("template/demo/figures/butterfly_plot.png", width: 100%),
  caption: dynamic-caption(
    [Butterfly plot of the simulated ERP data showing the grand average across all 32 channels (top), and per condition for car (bottom left) and face (bottom right). Each coloured line represents one channel; vertical dotted lines mark P100, N170, P300, and N400 peak latencies.],
    [Butterfly plot of the simulated ERP data.]
  ),
) <fig:butterfly_conditions>

The simulated data was visualized as topoplot series using UnfoldMakie.jl @mikheev2025unfoldmakie.


#figure(
  image("template/demo/figures/ERP_Topoplot_Series_Mean_32ch.png", width: 100%),
  caption: dynamic-caption(
    [Topoplot series of the trial-averaged simulated ERP data, illustrating the evolution of scalp voltage distributions across time, visualized using UnfoldMakie.jl.],
    [Topoplot series of the simulated ERP data.],
  ),
) <fig:topoplot>


@tbl:datasets presents the three datasets used in the evaluation, each containing different numbers of categorical conditions and continuous variables.

#figure(
  table(
    columns: (auto, auto, auto, 2.7cm, 2.8cm , auto),
    inset: 7pt,
    align: (left, left, left, left, left, left),
    [*Dataset*], [*Channels*], [*Trials*], [*Conditions*], [*Continuous levels*], [*Noise level*],
    [1],  [32], [2000], [2],  [10 ], [1],
    [2],  [64], [1200], [4], [6 ],  [1],
    [3],  [64], [1200], [4 ],   [6 ],  [7],
    [4],  [32], [2000], [10 ], [2], [2],
  ),
  caption: "Dataset configurations used across the three simulated designs. All datasets share the same time window (0–595 ms)."
) <tbl:datasets>

= Implementation 

== Average by condition

The average-by-condition approach adapts the original @BCNE pipeline proposed in @refBCNE while tailoring it to an @ERP -structured @EEG dataset.

The pipeline was implemented in four stages:

1. Grouping: The data was grouped by condition and continuous levels. Within each group, trials are averaged across the trial dimension to produce one clean @ERP per group, resulting in one averaged signal per condition-level combination.

2. Temporal smoothing: To reduce high-frequency noise while preserving temporal structure, the temporal autocorrelation of each EEG channel was computed to denoise the data. This smoothing reduces trial-level noise while preserving the underlying ERP waveform shape.

3. Spatial mapping: At each time point of averaged @ERP signals, the 32-channel vector is converted into a 6 x 6 grid, with channels with similar activity patterns arranged closer together, using Gromov-Wasserstein optimal transport @islam2023cartography. The generated grid image sequences, known as neuromaps, are input for the @CNN model.


#figure(
  image("template/demo/figures/neuromaps_erp_peaks_32ch.png", width: 70%),
  caption: dynamic-caption(
    [Neuromaps illustrating how channel-correlation patterns evolve across @ERP peak latencies for 32 channels represented by 6 x 6 grid. Red and blue regions reflect changes in channel activity over time; ERP components therefore correspond to distinct channel activations.],
    [Neuromaps at @ERP peak latencies.],
  ),
) <fig:neuromaps_erp_peaks_32ch>


The neuromaps in @fig:neuromaps_erp_peaks_32ch show that distinct regions of the grid corresponded to different scalp topographies, consistent with the simulated activity patterns. Parietal and central channels were active during the N170 component, while frontal and central channels dominated during the P300 component.


4. @CNN training: A convolutional neural network was trained in an unsupervised way over four recursive optimization stages (m1–m4) following the @BCNE recursive refinement strategy. The architecture comprised four convolutional layers (filters: 3, 16, 32, 64) followed by dense layers of 1024, 512, 256, and 8 units, with a final two-dimensional output.
  
At the first recursive stage m1, the target similarity distribution was computed directly from the neuromap images. At each subsequent stage, the model was reloaded, and the similarity target was recomputed from progressively deeper dense layer features, allowing the embedding to incorporate increasingly global representations of the temporal structure @refBCNE. The two-dimensional outputs at each stage were saved separately to enable comparison of the manifold structure across multiple recursion stages.

#figure(
  table(
    columns: (auto, auto),
    inset: 9pt,
    align: (left, left),
    [*Hyperparameter*], [*Value*],
    [Optimiser], [Adam, learning rate 5e-4],
     [NeuroMap grid], [6 × 6],
    [Loss function], [KL divergence],
    [Epochs per stage], [Max 100, early stopping (patience 20)],
    [Recursive stages], [4],
    [Convolutional layers], [4 layers, filters (3, 16, 32, 64), kernel 3×3],
    [Dense layers], [1024, 512, 256, 8 ],
    [Output], [two-dimensional embedding],
   
  ),
caption: "BCNE training hyperparameters used in the average-by-condition approach for 32-channel @EEG dataset."

) <tbl:hyperparameters_BCNE_condition>



== Trial-level projection 

The BCNE model was trained in an unsupervised manner on the winsorized grand average, without using condition or continuous labels.
Winsorization replaces extreme amplitude values with the boundary values at the chosen percentiles rather than removing them entirely,  to prevent extreme amplitude from distorting the average value @dixon1974trimming. 
Condition labels were used only post-hoc to compute condition-averaged embeddings for visualization, and played no role in model optimization.

1. Grand-average @ERP computation: All trials are averaged to produce a single (channel, time) @ERP. This signal is the low-noise representation of the event-locked component structure for the simulated dataset.
2. @BCNE training on the grand-average: Temporal smoothing, spatial mapping, and recursive CNN training proceed exactly as in the Average by Condition pipeline, but with a single input @ERP instead of condition-continuous groups. The trained @BCNE model defines a fixed coordinate system based on the grand average. 

  All @ERP trials, regardless of condition, follow the same basic temporal sequence: P100 at 100 ms, N170 at 170 ms, P300 at 300 ms, and N400 at 400 ms @KAPPENMAN2021117465. This temporal structure is captured by the grand average, and @BCNE learns a low-dimensional coordinate system to represent it. Before the model trains on any data, the channel amplitudes are first temporally smoothed using an autocorrelation map, which weights each time point by how strongly it relates to nearby time points. The smoothed amplitudes are then remapped onto a two-dimensional grid using optimal transport (@refBCNE), which arranges channels so that channels with similar activity end up as neighbors on the grid. This grid is fixed once from the grand average.
    
  Every trial passes through this same fixed transformation. The model then learns a coordinate system where each time point gets a two-dimensional position based on the spatial pattern of channel amplitudes on this grid at that moment. When an individual trial is projected, its amplitude pattern on the grid at each time point determines its two-dimensional position.

3.  Trial projection: Each trial is then passed through the trained @CNN model to generate its two-dimensional coordinates at each time point.

The architecture comprised three convolutional layers (filters: 3, 16, 32) followed by dense layers (256, 128, 64, 8) with a final two-dimensional output.

#figure(
  table(
    columns: (auto, auto),
    inset: 9pt,
    align: (left, left),

    [*Hyperparameter*], [*Value*],

    [Optimiser], [Adam, learning rate 5e-4],

    [NeuroMap grid], [6 × 6],

    [Loss function], [KL divergence],

    [Epochs per stage],
    [Max 100, early stopping (patience 20)],

    [Recursive stages],
    [3 (m1, m2, m3)],

    [Convolutional layers],
    [3 layers, filters (3, 16, 32), kernel 3×3],

    [Dense layers],
    [(256, 128, 64, 8)],

    [Output],
    [two-dimensional embedding],
  ),
  caption: "BCNE training hyperparameters used in the trial-level projection approach for 32-channel @EEG dataset."
) <tbl:hyperparameters_BCNE>

=== Comparison with @PCA
The projected trajectories were visualized against the reference grand-average trajectory and compared with a linear baseline (@PCA) trained on the same grand-average and used in the same projection mode. The directions of maximum variance were computed from the grand average, and then each trial was projected onto the two principal components with maximum variance @kayser2003optimizing. Principal components were learned from the grand average and applied as a fixed projection to each trial.

In this study, only @BCNE and @PCA as baseline comparisons were used for single-trial projection, as both define an explicit fixed mapping learned from the grand average that can be applied to new data. The out-of-sample projection capabilities of tSNE, UMAP, PHATE, and T-PHATE were not evaluated @anuragi2024mitigating.


=== Outlier Detection 

The automated pipeline was implemented on top of the trial-level @BCNE projections to assess whether @BCNE projection method could differentiate between outlier trials. Before any trial was projected, the data were winsorized by clipping amplitude values to the 2nd–98th percentile range of the normal trial distribution. 

The grand average used for training was computed by first winsorizing all individual trials at the 2nd and 98th percentiles of the amplitude distribution, then averaging across trials. This replaces extreme amplitude values with boundary values rather than removing them, ensuring that amplitude spikes do not distort the reference coordinate system used to project individual trials.

After projecting all individual trials without winsorization onto the fixed @BCNE coordinate system, each trial received an outlier score, calculated as the mean Euclidean distance between its two-dimensional trajectory and its condition's mean trajectory. Trials with scores exceeding three standard deviations above the mean were flagged as outliers, following a statistical approach based on a threshold frequently used to detect unusually deviant EEG epochs @delorme2001automatic. The outlier score reflects the overall geometric deviation of a trial from its expected temporal path in the embedding space.




== Rationale for two approaches
The two approaches address different aspects of data. The average-by-condition approach evaluated whether time-aware methods appear to recover
meaningful temporal and condition structure from clean @ERP signals, providing a
direct comparison across six methods.

The trial-level projection approach addressed a practical limitation of the
condition-averaged strategy. By averaging across trials before embedding, trial-level variability is lost. Training @BCNE on the grand average and projecting individual trials through the fixed coordinate system preserved this
variability while still recovering condition separation and enabling outlier detection that was not possible with condition averaging alone.
 
= Results 


To assess the interpretability of the proposed approach, simulated @ERP datasets were used for experimental designs of increasing complexity, spanning 2, 4, and 10 categorical conditions, combined with a continuous variable. This progressive design assessed how well the approach distinguishes different conditions in the embedding space as experimental complexity increased.

Across all four datasets, @T-PHATE and @BCNE consistently recovered temporally ordered trajectories with condition divergences at the correct @ERP latencies, while time-agnostic methods produced fragmented embeddings. The following sections document this pattern across increasing design complexity. 

== Average by Condition
=== Condition Separation

#figure(
  image("template/demo/figures/AvgByCondition/2conditions-AvgByCondition/fig_condition_2cond.png", width: 100%),
  caption: dynamic-caption(
    [Two-dimensional embeddings produced by all six methods for the two-condition design. @T-PHATE and @BCNE show the categorical separation, with each condition occupying a distinct region of the embedding space.],
    [Two-dimensional embeddings for the two-condition design.],
  ),
) <fig:fig_condition_2cond>



As shown in @fig:fig_condition_2cond, the car and face condition trajectories revealed a clear difference between time-aware and time-agnostic methods.
In @T-PHATE, the two conditions followed a shared temporal progression through the early P100 window before diverging into two distinct loops at the N170 and N400 time points, consistent with the condition effects encoded in the simulation. The loop structure in @T-PHATE arises because of the diffusion operator that connects each time point to its temporal neighbours, causing the embedding to trace a continuous path through time. Shared @ERP components such as P100 appear as a single overlapping path, while condition-specific components cause the trajectories to diverge into separate loops. @T-PHATE trajectories diverge at exactly the simulated @ERP latencies, indicating that the embedding recovers genuine temporal structure, given that the method received only trial-averaged EEG as input with no condition labels.


@BCNE also produced condition-separated embeddings. The separation became more compact through each recursive stage, as each 
stage recomputes the pairwise similarity target from progressively deeper 
network features rather than the raw images, causing the model to focus 
on an increasingly global structure and pull the embedding tighter.

=== Temporal Structure Preservation

#figure(
  image("template/demo/figures/AvgByCondition/2conditions-AvgByCondition/fig_time_car_2cond.png", width: 100%),
  caption: dynamic-caption(
    [Conditional variable and predictor for the car condition in the two-condition design. Points are coloured by time from stimulus onset (0 to 590 ms). @BCNE and @T-PHATE produce directed trajectories in which temporal order is preserved as a continuous path through the embedding space.],
    [Conditional variable and predictor for the car condition.],
  ),
) <fig:fig_time_car_2cond>


As shown in @fig:fig_time_car_2cond, each point represents a single time point, with the 32-channel voltage vector reduced to a two-dimensional coordinate, colored by time from stimulus onset (0 to 590 ms). @T-PHATE and @BCNE both produced directed trajectories in which temporal 
order was preserved as a continuous path from early to late time points. In 
@BCNE, the trajectory became more compact across recursive stages, indicating the progressive refinement of feature embeddings.

Time-agnostic methods produced fragmented or unordered embeddings without 
meaningful condition separation, consistent with their inability to encode 
temporal continuity. @PHATE and @T-PHATE share the same diffusion-geometry core, yet only 
@T-PHATE produced ordered trajectories. @PHATE produced overlapping 
structures across both conditions, confirming that the temporal 
encoding added by @T-PHATE is the key factor driving trajectory 
preservation, not the diffusion geometry alone.


#figure(
  image("template/demo/figures/AvgByCondition/2conditions-AvgByCondition/fig_continuous_all_2cond.png", width: 100%),
  caption: dynamic-caption(
    [Two-dimensional embeddings for the six continuous levels ranging from −5.0 to +5.0. @T-PHATE and @BCNE trajectories fan out at the P300 time point (~300 ms), forming a smooth gradient ordered by continuous value. The gradient is absent at P100, N170, and N400, reflecting the simulation design in which only P300 amplitude depends on the continuous variable.],
    [Continuous-level embeddings for the two-condition design.]
  ),
) <fig:fig_continuous_all_2cond>

As shown in @fig:fig_continuous_all_2cond, @T-PHATE produced a distinct 
continuous gradient, with each of the six levels occupying a separate loop 
smoothly ordered from −5.0 to +5.0. @BCNE recursive stages m1 and m4 also 
fanned out around P300 latency with continuous levels separating into 
distinguishable trajectories, whereas other methods produced no visible gradient.




=== Quantitative Analysis

The following standard quantitative metrics were computed:
1. *@KNN* accuracy measures local condition separability
2. *Trustworthiness and continuity* measure how faithfully local and global neighborhood structure is preserved between the high-dimensional and two-dimensional space @refBCNE. 


#figure(
  image("template/demo/figures/AvgByCondition/4conditions-AvgByCondition/fig0_metrics_table.png", width: 100%),
  caption: dynamic-caption(
    [Quantitative comparison of all six methods on the condition-averaged embeddings for simulated data with four categorical effects and six continuous levels. @KNN accuracy was highest for the time-aware methods, with @BCNE and @T-PHATE outperforming all others. Trustworthiness and continuity showed little difference across algorithms.],
    [Quantitative comparison of all six methods.]
  ),
) <fig0_metrics_table-4cond>


@fig0_metrics_table-4cond indicated that standard dimensionality 
reduction metrics measure neighborhood preservation, but are not 
sensitive to temporal-order preservation, exposing a limitation 
in quantitatively evaluating temporal dimensionality reduction. The recovery of temporal structure is instead evident qualitatively, in the ordered progression of the embedding trajectories from early to late time points across the epoch.

=== Robustness to Noise

Dataset 3 used a higher noise level (7)
allowing assessment of how increased noise affects the embedding quality across recursive stages.

#figure(
  image("template/demo/figures/AvgByCondition/4conditions-noise7/4conditions_noise7_64ch_recursive.png", width: 70%),
   caption: dynamic-caption(
    [@BCNE condition separation for the four-condition design at noise level 7 shown across 
    all four recursive stages, m1 to m4. Condition separation remains visible 
    across stages despite increased noise.],
    [@BCNE condition separation under high noise, noise level 7.]
  ),
) <fig:bcne_noise_4cond>

As shown in @fig:bcne_noise_4cond, @BCNE maintained condition-separated 
embeddings under the higher noise level across all four recursive stages. 
The four conditions remained distinguishable in the embedding space, 
though with greater overlap compared to the lower-noise design 
in the Appendix. The recursive refinement from m1 to m4 continued to 
improve separation progressively, suggesting the model learns increasingly 
discriminative features at each stage.




#figure(
  image("template/demo/figures/AvgByCondition/4conditions-noise7/fig_condition_4cond_noise7.png", width: 100%),
  caption: dynamic-caption(
    [Condition separation embeddings produced by all six methods for the 
    four-condition design (animal, car, face, house) under high noise 
    conditions (noise level 7). @T-PHATE and @BCNE maintain visible 
    condition separation despite increased noise, while time-agnostic 
    methods show further fragmentation.],
    [Condition separation for the four-condition design, noise level 7.]
  ),
)<fig:fig_condition_4cond_noise7>


#figure(
  image("template/demo/figures/AvgByCondition/4conditions-noise7/metrics_table_4cond_noise7.png", width: 100%),
  caption: dynamic-caption(
    [Quantitative comparison of all six methods under higher noise conditions (noise level 7). 
    @BCNE m4 achieved the highest @KNN accuracy (0.599), outperforming all other methods. 
    Trustworthiness and continuity remained high across all methods, consistent with the 
    lower-noise results.],
    [Quantitative metrics under higher noise conditions.]
  ),
) <fig:metrics_noise_4cond_noise7>


As shown in @fig:metrics_noise_4cond_noise7 and @fig:fig_condition_4cond_noise7, @BCNE recursive stage m4 achieved the highest 
@KNN accuracy of 0.599 under the higher noise level, outperforming all 
other methods, including @T-PHATE (0.48). Compared to the lower-noise results in @fig0_metrics_table-4cond, @KNN accuracy decreased across all methods under higher noise, while 
trustworthiness and continuity remained consistently high, indicating 
that these standard metrics are not able to distinguish between the conditions in high-noise data.


#pagebreak()


== Trial level analysis

After training on the grand average, each condition's mean trajectories were projected onto the fixed coordinate system to verify that the embedding captured condition-relevant structure before proceeding to single-trial analysis. 

As shown in @fig:bcne_condition_2cond, the car and face condition mean trajectories diverged at N170 and N400, the two latencies where the simulation introduced condition effects. This confirmed that the trained coordinate system revealed differences in condition.
#figure(
  image("template/demo/figures/Projection/Projection-2conditions/bcne_condition.png", width: 100%),
  caption: dynamic-caption(
    [@BCNE trial-level projection for the two-condition design. Condition mean trajectories for car and face diverge at the @ERP components where condition effects were simulated, following a shared path through the P100 window where no condition effect was present.],
    [@BCNE trial-level projection for the two-condition design.],
  ),
) <fig:bcne_condition_2cond>

Car and face trials share identical amplitude distributions at 100 ms and therefore occupy the same region in two-dimensional space. At N170, a condition effect of 3.0 was added on top of the baseline response of 5.0 for the face condition, producing stronger amplitudes across the electrode grid and therefore different two-dimensional positions in the embedding. The condition structure was recovered from the learned coordinate system. The model was trained only on the winsorized grand average. The condition separation emerged from amplitude differences 
already present in the data, without condition labels being provided to the model during training.

#figure(
  image("template/demo/figures/Projection/Projection-2conditions/bcne_continuous.png", width: 85%),
  caption: dynamic-caption(
    [@BCNE continuous-level embeddings for the two-condition design, showing trajectory fanning at P300 with levels ordered smoothly from −5.0 to +5.0. The distance plot confirms the effect peaks selectively at P300.],
    [@BCNE continuous-level embeddings for the two-condition design.],
  ),
) <fig:bcne_continuous_2cond>

As shown in @fig:bcne_continuous_2cond, the continuous-level embeddings produced a fanned structure across P300, with trajectories ordered smoothly from the lowest to the highest continuous value. The distance plot confirmed that the mean Euclidean distance from the reference level peaked at P300 and was absent at all other latencies, capturing the graded continuous effect selectively at the correct component.

#figure(
  image("template/demo/figures/Projection/Projection-2conditions/bcne_trial_conditions.png", width: 100%),
  caption: dynamic-caption(
    [@BCNE single-trial projection for the two-condition design. The grand average panel shows 200 randomly sampled individual trial trajectories as faint grey traces. Condition panels show individual trial clouds for car and face alongside the condition mean trajectory, with the grand average shown as a reference.],
    [@BCNE single-trial projection for the two-condition design.],
  ),
) <fig:bcne_trial_conditions_2cond>

As shown in @fig:bcne_trial_conditions_2cond, individual trials confirmed that temporal structure was preserved at the trial level. Car trials formed a compact cloud around the car mean trajectory, while face trials showed a wider spread particularly at N170, consistent with the larger amplitude variability of the face condition. The condition mean trajectories remained distinguishable within the trial clouds.

#figure(
  image("template/demo/figures/Projection/Projection-2conditions/bcne_trial_continuous.png", width: 100%),
  caption: dynamic-caption(
    [@BCNE single-trial continuous projection for the two-condition design. Continuous levels separate visible at P300.],
    [@BCNE single-trial continuous projection for the two-condition design.],
  ),
) <fig:bcne_trial_continuous_2cond>

As shown in @fig:bcne_trial_continuous_2cond, the single-trial continuous projection confirmed that the fanning at P300 observed in the condition-averaged embeddings was also present at the trial level. Continuous levels separated smoothly from −5.0 to +5.0 at P300, with the distance from the reference level peaking at 300 ms and returning to baseline at all other latencies.

#figure(
  image("template/demo/figures/Projection/Projection-4conditions/bcne_trial_conditions_4cond.png", width: 100%),
  caption: dynamic-caption(
    [@BCNE single-trial projection for the four-condition design (animal, car, face, house). Conditions diverge into distinct trajectories at the simulated @ERP component latencies; condition averages are the mean of per-trial two-dimensional embeddings.],
    [@BCNE single-trial projection for the four-condition design.],
  ),
) <fig:bcne_trial_conditions_4cond>

#figure(
  image("template/demo/figures/Projection/Projection-10conditions/bcne_trial_conditions_10conditions.png", width: 100%),
  caption: dynamic-caption(
    [@BCNE single-trial projection for the ten-condition design. All ten conditions form distinct trajectories diverging at the correct @ERP latencies, demonstrating that BCNE scales to complex experimental designs.],
    [@BCNE single-trial projection for the ten-condition design.]
  ),
) <fig:bcne_trial_conditions_10conditions>


#figure(
  image("template/demo/figures/Projection/Projection-10conditions/bcne_trial_continuous_10conditions.png", width: 110%),
  caption: dynamic-caption(
    [@BCNE continuous trial trajectories for the ten-condition design, showing fanning at P300 ordered by continuous value. No fanning is visible at other component latencies, confirming selective recovery of the continuous effect.],
    [@BCNE continuous trial trajectories for the ten-condition design.],
  ),
) <fig:bcne_trial_continuous_10conditions>

As shown in @fig:bcne_trial_conditions_4cond, @fig:bcne_trial_conditions_10conditions, and @fig:bcne_trial_continuous_10conditions, these findings extended to the four and ten condition designs, suggesting that @BCNE generalises faithfully across increasing experimental complexity. 
Continuous level fanning at P300 was preserved across both designs, with 
levels separating in smooth, orderly steps consistent with the simulation.


#figure(
  image("template/demo/figures/pca/pca_conditions.png", width: 90%),
  caption: dynamic-caption(
    [@PCA trial-level projection for the four-condition design (animal, car, face, house). Condition mean trajectories appear separated at the correct @ERP latencies, reflecting the maximum variance of the grand average rather than explicit temporal modelling.],
    [@PCA trial-level projection for the four-condition design.]
  ),
) <fig:pca_conditions>


#figure(
  image("template/demo/figures/pca/pca_continuous.png", width: 90%),
  caption: dynamic-caption(
    [@PCA continuous trial trajectories for the four-condition design, showing trajectory fanning at P300. Compared to @BCNE, the trajectories are more angular and the trial cloud more dispersed.],
    [@PCA continuous trial trajectories for the four-condition design.]
  ),
) <fig:pca_continuous>

As a linear baseline, @PCA was trained on the same grand average and projected in the same way as @BCNE, as shown in @fig:pca_conditions and @fig:pca_continuous. The condition trajectories appeared visually separated at the correct latencies, but this should not be interpreted as temporal structure preservation by @PCA. As a variance-driven method, @PCA does not explicitly model temporal order. The apparent separation arose because the grand average already contained temporal and condition-related structure, which @PCA projected onto its principal components. Compared to @BCNE, the @PCA trajectories were more angular and the trial cloud more dispersed, suggesting that @BCNE produced a more compact and temporally consistent embedding.



=== Outlier Analysis

To analyze whether the @BCNE embedding space could detect trial-level artifacts, two types of synthetic outlier trials were constructed and projected through the fixed coordinate system trained on the grand average.
The first type was a flat trial, in which all 32 channels were set to zero throughout the epoch, simulating complete signal loss.

#figure(
  image("template/demo/figures/outliers/outlier_test_flat.png", width: 80%),
  caption: dynamic-caption(
    [Flat outlier trial (red dot) collapses to a single stationary point in @BCNE embedding space, well outside the normal face trial cloud (blue lines). Raw @EEG confirms all 32 channels of the outlier are zero throughout the epoch.],
    [Flat outlier trial in @BCNE embedding space.]
  ),
) <fig:outlier_test_flat>

 As shown in @fig:outlier_test_flat, this trial collapsed to a single stationary point in the embedding space, well separated from the normal trial cloud. The raw @EEG confirmed that the signal was flat across all channels.

#figure(
  image("template/demo/figures/outliers/outlier_test_noise.png", width: 80%),
  caption: dynamic-caption(
    [Noise outlier trajectory (red dots) preserves the @ERP loop shape but is displaced outside the normal trial cloud. Raw @EEG shows broadband noise across all 32 channels with no identifiable @ERP components.],
    [Noise outlier trial in @BCNE embedding space.]
  ),
) <fig:outlier_test_noise>

The second type was a noise trial, in which the signal was replaced by Gaussian noise with a standard deviation of 4 µV across all 32 channels. As shown in @fig:outlier_test_noise, the resulting trajectory preserved the general loop shape of the grand average but was displaced outside the normal trial cloud. The raw @EEG showed broadband noise with no identifiable @ERP components, indicating that noise-corrupted trials occupied a geometrically distinct region of the embedding space.

Together, these results confirmed that the @BCNE projection approach captures the trial's underlying nature directly from its two-dimensional embedding space, without requiring labels or manual inspection. A collapsed point indicates signal loss, while irregularly scattered embedding indicates noise corruption. 

#figure(
  image("template/demo/figures/Projection/Projection-2conditions/bcne_outliers.png", width: 100%),
  caption: dynamic-caption(
    [An outlier trial and a typical trial projected into the two-dimensional @BCNE embedding space, alongside the outlier score distribution for the ten-condition design. Trials exceeding the detection threshold of 2.677 are flagged as outliers.],
      [Outlier score distribution for 10-condition dataset]

  ),
) <fig:bcne_outliers_2conditions>

To analyze outlier trials detection beyond visual inspection, an outlier scoring pipeline was implemented. 
Each trial was projected through the fixed @BCNE coordinate system after winsorization, and its deviation from the condition mean trajectory was calculated as the mean Euclidean distance across all time points in the two-dimensional embedding space. Trials exceeding the mean plus three standard deviations were flagged as outliers. As shown in @fig:bcne_outliers_2conditions, the outlier score distribution for the ten-condition design showed that 11 trials exceeded the detection threshold of 2.677. The worst outlier, with a score of 3.362, produced a trajectory that deviated substantially from the condition mean, while a typical trial closely followed the expected loop. The majority of trials clustered below 2.0, confirming that geometrically deviant trials formed a distinct tail in the distribution and were identifiable without manual inspection.


#pagebreak()
= Evaluation

#let yes = text(fill: green, [✓])
#let no = text(fill: red, [✗])

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 8pt,
    align: center,
    [*Method*], [*Temporal order*], [*Condition separation*], [*Smooth trajectory*], [*Single trial projection*],
    [PCA],     [#no],      [#no], [#no],  [#yes],
    [t-SNE],   [#no],      [#no], [#no],  [#no],
    [UMAP],    [#no],      [#no], [#no],  [#yes],
    [PHATE],   [#no],      [#no], [#no],  [#no],
    [T-PHATE], [#yes],     [#yes],     [#yes], [#no],
    [BCNE],    [#yes],     [#yes],     [#yes], [#yes],
  ),
  caption: "Qualitative comparison of all dimensionality reduction methods."
) <tbl:comparison>




== Qualitative Comparison
=== Continuity and smoothness of trajectories:
As shown in @tbl:comparison, BCNE and T-PHATE are the only methods that
preserve all three properties: temporal order, condition separation, and
trajectory smoothness.


=== Condition separation
The ability to distinguish between different experimental states in a low-dimensional space is another critical metric for evaluating dimensionality reduction techniques. The condition separation visible in @fig:fig_condition_2cond was structurally
meaningful and the divergence at N170 and N400 corresponded exactly to the
condition effects in the simulation, confirming that BCNE and T-PHATE recovered known neural differences without being given condition labels.


In contrast, @PCA and @tSNE failed to provide clear comparative visualizations. In @PCA , the branches corresponding to different conditions overlapped into the same regions of global variance, making it impossible to detect time evolution or condition differences. UMAP and t-SNE, while capable of forming clusters, often separate identical conditions into various clusters.
@PHATE and @T-PHATE share the same diffusion-geometry 
core mechanism, but only @T-PHATE recovers temporal 
trajectories, confirming that the temporal diffusion is the key factor distinguishing ordered from 
fragmented embeddings.


== Quantitative Comparison

Quantitative evaluation of dimensionality reduction embeddings is an open challenge due to the lack of neutral temporal structure preservation.

The metric proposed for @T-PHATE in @refTPHATE is optimized for diffusion-based embeddings, while the metric used in @refBCNE targets condition separability. Hence, neither was designed to evaluate temporal structure preservation as a standalone property.
Specifically, @T-PHATE used the @DeMAP metric, which measures how well the low-dimensional embedding preserves diffusion-based affinities from the high-dimensional space @refPHATE. Since @T-PHATE optimized directly for these affinities, @DeMAP favored it by design and therefore is not well-suited as a neutral comparison across all methods.

@BCNE (@refBCNE) used @KNN classification accuracy in the embedding
space for structure preservation, where the number of neighbors is determined by grid search. However, @KNN accuracy measures condition separability rather than temporal structure, making it unsuitable 
for evaluating chronological ordering.

Hence, this study evaluates embeddings qualitatively against the known ground truth of the simulated data, where condition separation at N170 and N400 and a continuous gradient at P300 serve as the reference criteria.


#pagebreak()
= Discussion

== Interpretation


This study demonstrated that time-aware dimensionality reduction methods produced qualitatively different and more meaningful embeddings than time-agnostic methods when applied to @ERP structured @EEG data, directly addressing RQ1. @T-PHATE and @BCNE both recovered temporally ordered trajectories with condition-specific divergences at the correct @ERP latencies, while @PCA, @tSNE, @UMAP, and @PHATE yielded fragmented or unordered embeddings that did not reflect the sequential structure of the data.

Both methods explicitly encode temporal autocorrelation, but differ in their core mechanism. @T-PHATE maintains temporal structure throughout the embedding via its dual-view diffusion 
framework, meaning temporal order continuously shapes the geometry of the final output. @BCNE, by contrast, encodes temporal autocorrelation as a preprocessing transformation applied once before the CNN training stage, after which the model optimizes for similarity preservation without an explicit temporal ordering constraint. @T-PHATE therefore produces geometrically cleaner, more structured loops, while @BCNE produces more compact trajectories.

The unsupervised condition separation at N170 and N400 is consistent with the neuroscientific literature, where N170 reflects face-selective processing and N400 reflects semantic processing @KAPPENMAN2021117465. Since @BCNE computes channel correlations at each time point, it captures these topographic differences, which explains why condition separation emerged at exactly these latencies without condition labels being provided.

The comparison between @BCNE and @T-PHATE revealed a genuine trade-off. @T-PHATE is preferable for exploratory visualization of condition-averaged data, producing geometrically distinct loops that clearly separate conditions. @BCNE, however, supports single-trial projection, outlier analysis, and generalization to unseen conditions through its fixed coordinate system, 
without requiring retraining.

Addressing RQ2, standard metrics such as trustworthiness and continuity were found to score consistently high across all approaches, regardless of whether trajectories were temporally ordered. The quality of embedding was assessed by @KNN accuracy, which provided more information about condition separability. This shows that qualitative evaluation against known ground truth is currently the most reliable verification.

== Application

The outlier detection results demonstrated a practical use case that is unique
to @BCNE among the six methods evaluated. Flat trials and noisy trials produced qualitatively distinct signatures in the embedding space: a flat trial collapsed to a stationary point, while a noise trial preserved the general loop shape but was displaced outside the normal trial cloud. This geometric readability means the nature of the trial data is directly interpretable from the trajectory position.

Standard @EEG artifact rejection uses amplitude 
threshold @Zhang2024, which may fail to distinguish between a 
large brain response and a muscle artifact 
of the same size. @BCNE can detect such artifacts through 
trajectory shape rather than amplitude.

== Limitations
Epoch-based input: @T-PHATE and @BCNE were designed for continuous time series
data. Applying them to epoch-based EEG requires averaging across trials before
projection, which discards trial-level information in the condition-averaged strategy.
An equivalent trial-level extension of @UMAP was not evaluated due to time constraints.


No neutral temporal quantitative metric: The metrics used in the reference papers @refBCNE and @refTPHATE were not designed to directly evaluate temporal structure preservation, making a neutral comparison unavailable. A dedicated metric for temporal structure preservation would strengthen the conclusions of future work.

== Future Scope
A validation on real EEG recordings would be a promising next step. The simulated data provided a controlled environment with known condition effects, but testing these methods on real experimental data would extend their practical value for exploratory EEG analysis.

Additionally, a dedicated metric for temporal structure preservation would strengthen future comparisons, as existing metrics measure neighborhood preservation rather than chronological ordering.

Finally, the trial-level projection approach can be extended to @UMAP for comparison and analysis for temporal-structured data.


#pagebreak()
= Summary

Time-aware dimensionality reduction methods produced more meaningful trajectories in two-dimensional embeddings than time-agnostic methods when applied to @ERP structured @EEG data.

A simulated @ERP dataset was generated using UnfoldSim @refunfoldsim, containing four components (P100, N170, P300, N400) with both categorical and continuous condition effects across 32 channels. Six dimensionality reduction methods were evaluated: @PCA, @tSNE, @UMAP, @PHATE, @T-PHATE, and @BCNE, under two approaches: condition-averaged input and single-trial projection via grand average.

The results showed that @T-PHATE and @BCNE recovered temporally ordered trajectories with condition-specific divergences at the simulated @ERP latencies. The comparison between @PHATE and @T-PHATE identified temporal autocorrelation encoding as the key mechanism underlying temporal trajectory preservation. In addition, a single-trial projection framework was developed in which @BCNE model was trained on the grand-average @ERP and subsequently used to project individual trials into a fixed embedding space. This approach supported trial-level condition separation, recovery of continuous effects, and trajectory-based outlier detection without retraining.


The results demonstrated that the @BCNE method provided a more interpretable representation, combining temporal structure preservation with single-trial projections, whereas @T-PHATE produced cleaner condition-averaged embeddings. Both methods could provide a practical basis for exploratory analysis of complex EEG experimental designs.

//#include "template/demo/main.typ"
//#include "template/demo/main.typ"