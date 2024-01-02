# method-for-predicting-low-cycle-fatigue-life-at-various-temperatures-for-nickel-based-superalloys
## :pushpin: Introduction
Code and data accompanying the manuscript titled "A novel data-driven method for predicting low cycle fatigue life at various temperatures for nickel-based superalloys. The corresponding manuscript number is **COMMAT-D-2302774.**

## Content

This repository includes three parts:

+ Data_pre-processing:

  | File name                         | Description                                               |
  | --------------------------------- | --------------------------------------------------------- |
  | Spearman_correlation_analysis.m   | Spearman correlation analysis implemented in Matlab       |
  | Spearman_correlation_analysis.txt | Values of spearman correlation coefficients               |
  | Data_cleansing.m                  | Data cleansing implemented in Matlab                      |
  | Data_cleansing.txt                | Values of Manhattan distance separately and the threshold |

+ ML_algorithms_for_comparison:

  | File name          | Description                                                  |
  | ------------------ | ------------------------------------------------------------ |
  | RF_comparison.m    | Random forest algorithm implemented in Matlab                |
  | SVM_comparison.m   | Support vector machine algorithm implemented in Matlab       |
  | BPANN_comparison.m | Back-propagation artificial neural network algorithm implemented in Matlab |
  | GABP_comparison.m  | Genetic algorithm-optimized BP-ANN algorithm implemented in Matlab |
