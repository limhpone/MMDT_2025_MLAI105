# Project Report
[by Thein Kyaw Lwin]
## Title: Predicting Resale Prices for HDB Flats in Singapore
## Data sources
Firstly, I used 2 datasets for this project. 
1. [HDB Resale Prices](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view)
2. [HDB Resale Prices Index](https://data.gov.sg/datasets/d_14f63e595975691e7c24a27ae4c07c79/view)

Beyond that, I menually grab the latest index data for 2025 Q2, Q3, Q4 via this [link](https://www.hdb.gov.sg/residential/selling-a-flat/overview/resale-statistics). And, append to 'HDB Resale Prices Index' dataset. Reference RPI has been set to 2025 Q2 (current) for calculating 'adjusted resale price'.

![image](https://github.com/user-attachments/assets/3d74e4b1-e8a3-42c0-be46-45e921d8d678)

## Model improvement analysis
**1. Number of parameters and hidden layers in each model**
| Model Name | Total number of parameters | Number of hidden layers |
|---|---|---|
|Model_1|5,441|2|
|Model_2|17,025|3|
|Model_3|17,537|4|

**2. Model performance metric and training time**

| Performance Metric | Model_1_Train | Model_1_Test | Model_2_Train | Model_2_Test | Model_3_Train | Model_3_Test |
|---|---|---|---|---|---|---|
| Train Time (s) | 159.89 | N/A | 168.76 | N/A | 172.26 | N/A |
| RMSE | 72806.42 | 72840.68 | 61964.29 | 62378.17 | 59091.36 | 59051.36 |
| MAE | 53228.38 | 53332.39 | 44549.74 | 44795.44 | 42693.36 | 42691.35 |
| MAPE | 8.57 | 8.60 | 7.08 | 7.12 | 6.83 | 6.84 |
| R2 Score | 0.87 | 0.87 | 0.91 | 0.91 | 0.92 | 0.92 |

- Model 3 with 4 hidden layers achieved the best performance with the lowest RMSE(59051.36), MAE(42691.35) and R-square score(0.92). So, the deeper the model's network, the better the performance.
- As the number of hidden layers increased, training time also increased slightly.
- For further steps, I want to tweak the numbers of neurons in hidden layers whehter the performance are improved or not.
- Throughtout this project, I learned basic skills and understanding how to build neural network model from scratch to solve the problem on my own.

### Challenges during this project
- I put a lot of effort and time in 'Data Preprocessing' stage to understand the nature of the dataset. Fortunately, 'Reference Repo of Teacher Dr. Myo Thida' guide me to the right track.
- I realized that I need to re-polish my Python skills to get faster for the next projcts.
