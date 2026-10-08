Hardware: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs, 4 PyTorch threads, Python 3.13.16, PyTorch 2.14.1+cu130, scikit-learn 1.9.1
Real training samples: 7,000; median of 5 runs; time = KDE sampling + radio model inference (models preloaded, no file I/O).

| Model              | A = 10 (70,000 samples)   | A = 20 (140,000 samples)   | A = 30 (210,000 samples)   |
|:-------------------|:--------------------------|:---------------------------|:---------------------------|
| KDE-1NN (Proposed) | 0.120 s                   | 0.294 s                    | 0.346 s                    |
| KDE-RF             | 0.125 s                   | 0.357 s                    | 0.476 s                    |
| KDE-GPR (RQ)       | 22.424 s                  | 46.162 s                   | 68.371 s                   |
| KDE-MLP            | 0.109 s                   | 0.264 s                    | 0.398 s                    |
| KDE-CGAN           | 0.091 s                   | 0.222 s                    | 0.307 s                    |
| KDE-ReFlow         | 3.093 s                   | 5.921 s                    | 8.319 s                    |
| KDE-NF             | 0.146 s                   | 0.625 s                    | 0.862 s                    |
