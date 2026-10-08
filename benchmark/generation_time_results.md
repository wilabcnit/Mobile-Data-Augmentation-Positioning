Hardware: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs, 4 PyTorch threads, Python 3.13.16, PyTorch 2.14.1+cu130, scikit-learn 1.9.1
Real training samples: 7,000; median of 3 runs; time = KDE sampling + radio model inference (models preloaded, no file I/O).

| Model              | A = 10 (70,000 samples)   | A = 20 (140,000 samples)   | A = 30 (210,000 samples)   |
|:-------------------|:--------------------------|:---------------------------|:---------------------------|
| KDE-1NN (Proposed) | 0.3 s                     | 0.5 s                      | 0.7 s                      |
| KDE-RF             | 0.1 s                     | 0.3 s                      | 0.4 s                      |
| KDE-GPR (RQ)       | 20.6 s                    | 39.3 s                     | 61.4 s                     |
| KDE-MLP            | 0.1 s                     | 0.3 s                      | 0.3 s                      |
| KDE-CGAN           | 0.1 s                     | 0.2 s                      | 0.3 s                      |
| KDE-ReFlow         | 2.8 s                     | 5.9 s                      | 8.6 s                      |
| KDE-NF             | 0.1 s                     | 0.6 s                      | 0.8 s                      |
