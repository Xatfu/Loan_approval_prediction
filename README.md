# Loan Approval Prediction Project

## Introduction
This project aims to predict loan approval status (`loan_status`) based on various personal and loan-related features. The solution involves data loading, preprocessing, exploratory data analysis (implicitly done through `value_counts` and `info`), and building a K-Nearest Neighbors (KNN) classification model.

## Data Loading

You can find the dataset used for this project on Kaggle here: https://www.kaggle.com/competitions/loan-approval-predictions/data


## Dataset Description
The dataset consists of two main files: `train.csv` for training and `test.csv` for prediction. The training dataset `df_train` has 58,645 entries and 13 columns, while the test dataset `df_test` has 39,098 entries and 12 columns. The `loan_status` column is the target variable in the training set.

**Key Features:**
*   `id`: Unique identifier (dropped during preprocessing).
*   `person_age`: Age of the person.
*   `person_income`: Annual income of the person.
*   `person_home_ownership`: Type of home ownership (e.g., RENT, MORTGAGE, OWN, OTHER).
*   `person_emp_length`: Employment length in years.
*   `loan_intent`: Purpose of the loan (e.g., EDUCATION, MEDICAL, PERSONAL).
*   `loan_grade`: Loan grade assigned by the lender.
*   `loan_amnt`: Loan amount.
*   `loan_int_rate`: Interest rate of the loan.
*   `loan_percent_income`: Loan amount as a percentage of income.
*   `cb_person_default_on_file`: Credit history indicating default.
*   `cb_person_cred_hist_length`: Credit history length in years.
*   `loan_status`: Target variable (0 for approved, 1 for rejected).

## Data Preprocessing
The following steps were performed to prepare the data:
1.  **Dropped 'id' column**: The 'id' column was removed from both training and test datasets as it's not a predictive feature.
2.  **Handling Missing Values**: No missing values were found in either the training or test datasets.
3.  **Label Encoding Categorical Features**: The following categorical columns were converted into numerical representations using `LabelEncoder`:
    *   `person_home_ownership`
    *   `loan_intent`
    *   `loan_grade`
    *   `cb_person_default_on_file`

## Model Training and Evaluation
1.  **Data Splitting**: The training data (`df_train`) was split into training and validation sets (`X_train`, `X_val`, `y_train`, `y_val`) with a `test_size` of 0.2 and `random_state` of 42.
2.  **Model Selection**: A K-Nearest Neighbors (KNN) classifier was chosen for this task.
3.  **Hyperparameter Tuning (K-value)**: An iterative process was used to find the optimal `k` for the KNN model by testing `k` values from 1 to 29 and evaluating `accuracy_score` on the validation set. The plot generated suggests that a `k` value of 6 yielded a good accuracy.
4.  **Final Model Training**: The KNN model was trained with `n_neighbors = 6` on the full training data (`X_train`, `y_train`).
5.  **Prediction**: The trained model was used to predict `loan_status` on the `X_test` dataset.

## Submission
The predictions were saved to a CSV file named `sub.csv` with 'id' and 'loan_status' columns, suitable for submission.
