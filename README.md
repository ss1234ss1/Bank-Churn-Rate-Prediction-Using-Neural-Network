
#  Predicting Bank Customer Churn with Artificial Neural Networks (ANN)

## Overview
This project aims to predict **bank customer churn** using **Artificial Neural Networks (ANNs)**. The model is trained on **bank customer data**, incorporating **demographics, account features, and activity metrics**. By leveraging deep learning techniques, we seek to improve churn prediction accuracy and provide actionable insights for customer retention.

###  **Why Customer Churn Matters?**
- **Churn is Costly**: A **5% increase in retention** can lead to a **25-95% increase in profits**.
- **Customer Retention is Cheaper**: Acquiring new customers costs significantly more than retaining existing ones.
- **Actionable Insights**: Understanding churn patterns helps businesses make **data-driven retention strategies**.

---

##  Data and Preprocessing
### **Dataset**
- Source: [Kaggle - Churn Modelling Data](https://www.kaggle.com/datasets/shubh0799/churn-modelling/data)
- **10,000 samples** with **14 features**, including:
  - **Demographics**: Age, Gender, Geography
  - **Account Features**: Credit Score, Balance, Has Credit Card
  - **Customer Activity**: IsActiveMember, Number of Products

### **Data Preprocessing**
- **Encoding categorical variables** (`Geography`, `Gender`)
- **Feature scaling**: Applied **StandardScaler** to normalize `CreditScore`, `Balance`, etc.
- **Handling class imbalance**: Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.

---

##  Model Architecture
Our **Neural Network (ANN)** consists of:
1. **Input Layer**: 10 features
2. **Hidden Layers**:
   - 2 Dense layers with **ReLU activation**
   - 2 Dropout layers (to prevent overfitting)
   - 2 Batch Normalization layers
3. **Output Layer**: Sigmoid activation for binary classification

### **Training Parameters**
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Epochs**: **100** (Peak accuracy at **epoch 85**)
- **Best Validation Accuracy**: **0.8680**

---

##  Experiments and Model Optimization
### **1️⃣ Feature Selection**
- **Lasso Regularization**:
  - Selected: `Credit Score, Geography, Gender, Age, Balance, IsActiveMember`
  - **Validation Accuracy: 0.848**
- **Forward Selection**:
  - Selected: `Credit Score, Gender, Age, Balance, IsActiveMember`
  - **Validation Accuracy: 0.846**

### **2️⃣ Handling Class Imbalance**
- Applied **SMOTE** (Synthetic Minority Oversampling Technique).
- **Validation Accuracy dropped to 0.8008**, indicating imbalance might carry meaningful patterns.

### **3️⃣ Hyperparameter Tuning**
- Used **GridSearchCV** to find optimal hyperparameters.
- **Best Parameters**:
  - `batch_size`: 32
  - `activation`: ReLU
  - `dropout_rate`: 0.1
  - `neurons`: 32
  - `optimizer`: RMSprop
- **Best Validation Accuracy: 0.8756**

---

##  Results and Evaluation
| **Metric**        | **Score**  |
|------------------|------------|
| **Accuracy**      | 87.56% |
| **Precision**     | 76.35% |
| **Recall**        | 40.65% |
| **F1-Score**      | 53.26% |

- **Churn Prediction Insights**:
  - **Older customers** were more likely to churn.
  - Customers with **higher balances** exhibited **higher churn rates**.
  - **German customers** showed the highest churn probability.

- **Surprising Findings**:
  - Feature selection **did not** significantly improve accuracy.
  - Using a **balanced dataset (via SMOTE) performed worse than the imbalanced dataset**.

---




##  Future Improvements
- **Exploring more advanced architectures** (e.g., LSTMs for sequential customer behavior or Add More Dense Layers to improve on performance).
- **Feature engineering** to introduce interaction terms or external data.
- **Implementing ensemble models** (ANN + Random Forest for hybrid learning).
- **Deploying the model** as a **web service using Flask or FastAPI**.

---

## References
- **Bain & Company** - [Customer Retention Insights](https://www.bain.com/insights/retaining-customers-is-the-real-challenge/)
- **Kaggle Churn Data** - [Dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling/data)
- **AIMind** - [SMOTE for Handling Imbalanced Data](https://aimind.com/smote-technique/)

---

This project was developed as part of **DS340: Introduction to Machine Learning & AI** at Boston University.

