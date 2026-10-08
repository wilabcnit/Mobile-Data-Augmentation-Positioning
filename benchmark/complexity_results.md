Hardware: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs, Python 3.13.16, scikit-learn 1.9.1
Data: 7000 real training fingerprints, 35 cells; 50 timed runs per step.

| Step                                                     |   Median time (ms) | Mean ± std (ms)   |   Median per sample (µs) |
|:---------------------------------------------------------|-------------------:|:------------------|-------------------------:|
| Spatial augmentation: KDE sampling of 14000 positions    |               1.24 | 1.26 ± 0.08       |                     0.09 |
| Radio augmentation: 1NN fingerprints for 14000 positions |              46.53 | 47.81 ± 4.49      |                     3.32 |
| wKNN training on 21000 real + synthetic fingerprints     |               0.2  | 0.21 ± 0.03       |                     0.01 |
| Positioning inference: wKNN on 2000 test fingerprints    |            1203.58 | 1645.09 ± 709.86  |                   601.79 |
| End-to-end: augmentation + wKNN training + inference     |            1883.64 | 2094.77 ± 966.47  |                   941.82 |
