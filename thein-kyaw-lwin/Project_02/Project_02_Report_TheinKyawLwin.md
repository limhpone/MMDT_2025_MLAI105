# Project 02 Report: Name Classification for Regional Identification
[By Thein Kyaw Lwin]

## Description
This project aims to classify names of locations within Myanmar into one of the 18 states and regions. The classification model will be built using deep learning techniques to accurately predict the state or region based on the given location name. The primary goal is to facilitate easy categorization and identification of locations for various applications, including geographic analysis, administrative management, and data organization.

## Data
The datasets used in this project are obtained from the Myanmar Information Management Unit (MIMU) Resource Centre, a reliable source providing extensive geographical data. The datasets include:

1.  Villages Dataset: Contains 14,047 village names along with their corresponding states and regions.
2.  Towns Dataset: Comprises 536 town names and their respective states and regions.
3.  Supplementary Dataset: Collected from various news articles, includes 58,789 records of town names, often with variations in spelling and formatting. This dataset also contains duplicated entries due to multiple mentions in articles.

## Experimentation with different hidden layers (deep and wide)

In this project, I did experimentation with deeper and wider neural networks to see their classification performance.
- For deeper neural networks, I used following architectures:
  - model_2layers: 2 hidden layers with 32 and 16 neurons
  - model_3layers: 3 hidden layers with 64, 32, and 16 neurons
  - model_4layers: 4 hidden layers with 128, 64, 32, and 16 neurons
- For wider neural networks, I used following architectures:
  - model_2layers_wide: 2 hidden layers with 256 and 128 neurons
  - model_3layers_wide: 3 hidden layers with 256, 128, and 64 neurons
  - model_4layers_wide: 4 hidden layers with 256, 128, 64, and 32 neurons

These architectures allows for comparing how increasing model depth (more layers) and width (more neurons per layer) affects model performance and training time.

Moreover, each architecture was tested under four settings:
1. **Baseline:** No regularization
2. **Dropout:** Dropout layers only
3. **BatchNorm:** Batch Normalization only
4. **Both:** Dropout and Batch Normalization combined

So, total 24 experimentations (6 architectures x 4 settings = 24) was tested for this project.

## Model Evaluation

|    | Model                        | Layers             |   Total Params |   Train Accuracy |   Test Accuracy | Dropout   | BatchNorm   |   Training Time (s) |
|----|------------------------------|--------------------|----------------|------------------|-----------------|-----------|-------------|---------------------|
|  0 | model_2layers_baseline       | [32, 16]           |         416962 |           0.8644 |          0.3041 | False     | False       |               21.28 |
|  1 | model_2layers_dropout        | [32, 16]           |         416962 |           0.8625 |          0.3032 | True      | False       |               22.85 |
|  2 | model_2layers_batchnorm      | [32, 16]           |         417154 |           0.8643 |          0.2827 | False     | True        |               23.53 |
|  3 | model_2layers_both           | [32, 16]           |         417154 |           0.8598 |          0.2726 | True      | True        |               25.71 |
|  4 | model_3layers_baseline       | [64, 32, 16]       |         835170 |           0.8641 |          0.2815 | False     | False       |               21.63 |
|  5 | model_3layers_dropout        | [64, 32, 16]       |         835170 |           0.8531 |          0.3047 | True      | False       |               24.86 |
|  6 | model_3layers_batchnorm      | [64, 32, 16]       |         835618 |           0.8626 |          0.2808 | False     | True        |               25.94 |
|  7 | model_3layers_both           | [64, 32, 16]       |         835618 |           0.8595 |          0.2793 | True      | True        |               29.31 |
|  8 | model_4layers_baseline       | [128, 64, 32, 16]  |        1675682 |           0.8571 |          0.2439 | False     | False       |               22.69 |
|  9 | model_4layers_dropout        | [128, 64, 32, 16]  |        1675682 |           0.8517 |          0.2865 | True      | False       |               27.78 |
| 10 | model_4layers_batchnorm      | [128, 64, 32, 16]  |        1676642 |           0.8613 |          0.2617 | False     | True        |               28.63 |
| 11 | model_4layers_both           | [128, 64, 32, 16]  |        1676642 |           0.8577 |          0.2776 | True      | True        |               34.04 |
| 12 | model_2layers_wide_baseline  | [256, 128]         |        3364242 |           0.8636 |          0.3013 | False     | False       |               22.36 |
| 13 | model_2layers_wide_dropout   | [256, 128]         |        3364242 |           0.8649 |          0.3049 | True      | False       |               24.83 |
| 14 | model_2layers_wide_batchnorm | [256, 128]         |        3365778 |           0.8635 |          0.2904 | False     | True        |               24.66 |
| 15 | model_2layers_wide_both      | [256, 128]         |        3365778 |           0.865  |          0.267  | True      | True        |               27.21 |
| 16 | model_3layers_wide_baseline  | [256, 128, 64]     |        3371346 |           0.861  |          0.2668 | False     | False       |               23.04 |
| 17 | model_3layers_wide_dropout   | [256, 128, 64]     |        3371346 |           0.8623 |          0.2993 | True      | False       |               26.67 |
| 18 | model_3layers_wide_batchnorm | [256, 128, 64]     |        3373138 |           0.8633 |          0.2754 | False     | True        |               27.46 |
| 19 | model_3layers_wide_both      | [256, 128, 64]     |        3373138 |           0.8632 |          0.2832 | True      | True        |               30.88 |
| 20 | model_4layers_wide_baseline  | [256, 128, 64, 32] |        3372850 |           0.8603 |          0.2772 | False     | False       |               23.9  |
| 21 | model_4layers_wide_dropout   | [256, 128, 64, 32] |        3372850 |           0.8569 |          0.3041 | True      | False       |               29.27 |
| 22 | model_4layers_wide_batchnorm | [256, 128, 64, 32] |        3374770 |           0.8621 |          0.2726 | False     | True        |               29.66 |
| 23 | model_4layers_wide_both      | [256, 128, 64, 32] |        3374770 |           0.8608 |          0.275  | True      | True        |               35.08 |

## Finding
- Models showed overfitting signs (high training accuracy > 85% and low testing accuracy 26% ~ 30%)
- **'model_2layers_wide_dropout'** with test accuracy (30.49%) slightly outperformed others.
- In this project, applying normalization and regularization didn't impact significantly on models' accuracy eventhough it increased training time.
- I think that overfitting occured due to 'class imbalance' in target classes (18 states and region). So, I check 'class distribution' as below.

![image](https://github.com/user-attachments/assets/48bcee1a-e2df-430a-9bd6-c0c74b8289f2)

|   ID | SR Name      |   Count |
|------|--------------|---------|
|    0 | Ayeyarwady   |    2178 |
|    1 | Bago (East)  |     921 |
|    2 | Bago (West)  |     814 |
|    3 | Chin         |     648 |
|    4 | Kachin       |     878 |
|    5 | Kayah        |     228 |
|    6 | Kayin        |     552 |
|    7 | Magway       |    2244 |
|    8 | Mandalay     |    1903 |
|    9 | Mon          |     600 |
|   10 | Nay Pyi Taw  |     234 |
|   11 | Rakhine      |    1331 |
|   12 | Sagaing      |    3278 |
|   13 | Shan (East)  |     257 |
|   14 | Shan (North) |    1476 |
|   15 | Shan (South) |     616 |
|   16 | Tanintharyi  |     524 |
|   17 | Yangon       |     831 |

## Challenges during this project
- For me, it is hard to understand data preprocessing steps for text input. As further study, I will try to understand Vectorizer, Tokenization and Lemmatization.
- At first, I used 'Google Colab' for model training and testing. Later, I noticed that training time was too long. So, I switched using Kaggle with GPU to speed up experimentation.

