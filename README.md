# A novel data-driven method for predicting low cycle fatigue life at various temperatures for nickel-based superalloys

## :pushpin: Introduction
Code and data accompanying the manuscript titled "A novel data-driven method for predicting low cycle fatigue life at various temperatures for nickel-based superalloys. The corresponding manuscript number is **COMMAT-D-2302774.**

## :pushpin: Content

This repository includes three parts:

+ Data_pre-processing:

  | File name                         | Description                                               |
  | --------------------------------- | --------------------------------------------------------- |
  | Spearman_correlation_analysis.m   | Spearman correlation analysis implemented in Matlab       |
  | Spearman_correlation_analysis.txt | Values of Spearman correlation coefficients               |
  | Data_cleansing.m                  | Data cleansing implemented in Matlab                      |
  | Data_cleansing.txt                | Values of Manhattan distance separately and the threshold |

+ ML_algorithms_for_comparison:

  | File name          | Description                                                  |
  | ------------------ | ------------------------------------------------------------ |
  | RF_comparison.m    | Random forest algorithm implemented in Matlab                |
  | SVM_comparison.m   | Support vector machine algorithm implemented in Matlab       |
  | BPANN_comparison.m | Back-propagation artificial neural network algorithm implemented in Matlab |
  | GABP_comparison.m  | Genetic algorithm-optimized BP-ANN algorithm implemented in Matlab |

+ ANN_algorithms_based_enlarged_dataset

  | File name               | Description                                                  |
  | ----------------------- | ------------------------------------------------------------ |
  | EnlargedDataset_raw.txt | The raw value of the enlarged dataset                        |
  | Dataset_Waspaloy.txt    | The raw value of the nickel-based superalloy, Waspaloy$^TM^$ |
  | GABP_algorithm.m        | The GA-BP model implemented in Matlab                        |
  | BPANN_algorithm.m       | The BP-ANN model implemented in Matlab                       |
  | Predicted_GABP.mat      | The GA-BP model that had been trained                        |
  | Predicted_BPANN.mat     | The BP-ANN model that had been trained                       |
  | Predicted_load.m        | Used to call the above trained models, which can be run directly. |

  ## :pushpin: Q&A

  **Q**: I want to reproduce the predicted results presented in Table 4 of the manuscript.
  
  **A**: Store the 'Predicted_BPANN.mat', 'Predicted_GABP.mat', and 'Predicted_load.m' in the same folder and then run the 'Predicted_load.m'.
  
  **Q**: I want to verify the potential applications of the public dataset.
  
  **A**: The dataset and code are intended to support academic research and analysis.
