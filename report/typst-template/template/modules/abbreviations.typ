#import "@preview/glossarium:0.5.6": print-glossary

// Only print short and long, disregard rest
#let custom-print-title(entry) = {
  let short = entry.at("short")
  let long = entry.at("long", default: "")
  [#strong(short) #h(0.5em) #long]
}

#let abbreviations-page(abbreviations) = {
  // --- List of Abbreviations ---
  align(left)[
    = List of Abbreviations

    #let abbreviations = (
  (
    key: "EEG",
    short: "EEG",
    long: "Electroencephalography",
  ),
  (
    key: "ERP",
    short: "ERP",
    long: "Event-Related Potentials",
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
  (
    key: "fMRI",
    short: "fMRI",
    long: "Functional magnetic resonance imaging",
  ),
)
    #print-glossary(
      abbreviations,
      user-print-title: custom-print-title,
      disable-back-references: true,
    )
  ]
}