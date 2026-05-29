#import "../utils/global.typ": *
#import "../utils/symbols.typ": *

The human brain generates complex electrical signals. When neurons fire in response to sensory stimuli and cognitive tasks, the patterns of these signals can be captured at the scalp using electroencephalography (EEG). Embedded within this high-dimensional, noisy signal is structured information about how the brain evolves through distinct states

The central challenge is one of visualization and representation of these brain states. Humans can interpret in at most three dimensions, yet EEG data lives in a space with as many dimensions as there are channels multiplied by temporal features. To make this data interpretable, dimensionality reduction techniques can help to project the data from its native high-dimensional space into a low-dimensional representation while preserving as much meaningful structure as possible.
