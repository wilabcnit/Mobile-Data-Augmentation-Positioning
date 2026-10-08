Hardware: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs, Python 3.13.16, scikit-learn 1.9.1
Data: 7000 real training fingerprints, 35 cells; 50 timed runs per step.

| Step                                                    |   Median time (ms) | Mean ± std (ms)   |   Median per sample (µs) |
|:--------------------------------------------------------|-------------------:|:------------------|-------------------------:|
| Spatial augmentation: KDE sampling of 2000 positions    |               0.33 | 0.35 ± 0.08       |                     0.16 |
| Radio augmentation: 1NN fingerprints for 2000 positions |               7.62 | 7.79 ± 0.92       |                     3.81 |
| wKNN training on 9000 real + synthetic fingerprints     |               0.21 | 0.26 ± 0.14       |                     0.02 |
| Positioning inference: wKNN on 2000 test fingerprints   |             478.07 | 682.81 ± 426.63   |                   239.04 |
| End-to-end: augmentation + wKNN training + inference    |             500.2  | 640.16 ± 303.71   |                   250.1  |
