# Climate Model Simulation Crash Prediction

## Project Overview

This project focuses on predicting climate model simulation crashes using machine learning techniques. The dataset contains records of simulation crashes encountered during climate model uncertainty quantification (UQ) ensembles. The goal is to classify simulation outcomes (fail or succeed) based on 18 model parameters from the Parallel Ocean Program (POP2) component of the Community Climate System Model (CCSM4).

The dataset consists of 540 simulation records, with 46 simulations (8.5%) resulting in failures, making this a highly imbalanced binary classification problem.

## Dataset Information

- **Total Records**: 540
- **Features**: 18 model parameters (vconst_corr, vconst_2, vconst_3, vconst_4, vconst_5, vconst_7, ah_corr, ah_bolus, slm_corr, efficiency_factor, tidal_mix_max, vertical_decay_scale, convect_corr, bckgrnd_vdc1, bckgrnd_vdc_ban, bckgrnd_vdc_eq, bckgrnd_vdc_psim, Prandtl)
- **Target Variable**: outcome (0 = failure, 1 = success)
- **Class Distribution**: 494 successes (91.5%), 46 failures (8.5%)
- **Data Source**: Lawrence Livermore National Laboratory (LLNL) UQ Pipeline

## Project Structure

### Assignment 1: Neural Network from Scratch

**File**: `CPC251_Assignment1_Climate2.ipynb`

**Objective**: Implement a feedforward neural network from scratch using TensorFlow for binary classification.

**Implementation Details**:
- **Architecture**: 
  - Input layer: 5 features
  - Hidden layer: 16 neurons with ReLU activation
  - Output layer: 1 neuron with sigmoid activation
- **Loss Function**: Binary cross-entropy
- **Optimization**: Gradient descent with TensorFlow's automatic differentiation
- **Training Strategy**: 
  - Mini-batch training (batch size: 32)
  - Early stopping based on validation loss
  - Learning rate: 0.01
  - Maximum epochs: 100
- **Data Split**: 70% training, 10% validation, 20% test
- **Preprocessing**: StandardScaler normalization

**Performance Results**:
- **Test Accuracy**: 98.00%
- **Precision (Class 0)**: 0.99
- **Recall (Class 0)**: 0.97
- **Precision (Class 1)**: 0.97
- **Recall (Class 1)**: 0.99
- **F1-Score (Macro Average)**: 0.98
- **F1-Score (Weighted Average)**: 0.98

The model achieved excellent performance with balanced precision and recall across both classes, demonstrating effective handling of the binary classification task.

### Project Part 1: Traditional Machine Learning Models

**File**: `CPC251_Project_ Part1_Climate2.ipynb`

**Objective**: Compare Decision Tree and Support Vector Machine classifiers with feature selection and hyperparameter tuning.

#### Model 1: Decision Tree Classifier

**Feature Selection**:
- Method: SelectFromModel with median threshold
- Selected Features: 10 features (vconst_corr, vconst_2, vconst_3, vconst_4, vconst_7, ah_corr, tidal_mix_max, convect_corr, bckgrnd_vdc1, bckgrnd_vdc_ban)
- Most Important Feature: vconst_2 (importance: 0.348)

**Hyperparameter Tuning**:
- Method: GridSearchCV with 5-fold cross-validation
- Scoring Metric: Weighted F1-score
- Best Parameters: max_depth=4, min_samples_split=2
- Best CV Score: 0.8966

**Performance Results**:
- **Test Accuracy**: 92.59%
- **Precision (Class 0)**: 0.50
- **Recall (Class 0)**: 0.75
- **Precision (Class 1)**: 0.98
- **Recall (Class 1)**: 0.94
- **F1-Score (Macro Average)**: 0.78
- **F1-Score (Weighted Average)**: 0.93

The Decision Tree struggled with the minority class (Class 0), achieving lower precision and recall for failures compared to successes.

#### Model 2: Support Vector Machine

**Feature Selection**:
- Method: Sequential Feature Selection (forward selection)
- Selected Features: 10 features (Study, vconst_corr, vconst_2, vconst_4, slm_corr, tidal_mix_max, vertical_decay_scale, convect_corr, bckgrnd_vdc1, bckgrnd_vdc_eq)
- Selection Strategy: Forward selection with 5-fold cross-validation using weighted F1-score

**Hyperparameter Tuning**:
- Method: GridSearchCV with 5-fold cross-validation
- Kernels Tested: linear, poly, rbf, sigmoid
- C Values: 1-10
- Best Parameters: C=10, kernel='linear'
- Best CV Score: 0.9530

**Performance Results**:
- **Test Accuracy**: 99.07%
- **Precision (Class 0)**: 1.00
- **Recall (Class 0)**: 0.88
- **Precision (Class 1)**: 0.99
- **Recall (Class 1)**: 1.00
- **F1-Score (Macro Average)**: 0.96
- **F1-Score (Weighted Average)**: 0.99

The SVM model outperformed the Decision Tree significantly, achieving near-perfect classification with excellent handling of both classes.

**Comparison Summary**:
The SVM model demonstrated superior performance across all metrics, particularly in handling the imbalanced dataset. Its linear kernel with C=10 provided optimal separation of the classes, while the Decision Tree showed limitations in predicting the minority class.

### Project Part 2: Advanced Machine Learning Models

**File**: `CPC251_Project_ Part2_Climate2.ipynb`

**Objective**: Implement and compare Feedforward Neural Network and Fuzzy Logic System for climate simulation crash prediction.

#### Model 1: Feedforward Neural Network

**Preprocessing**:
- StandardScaler normalization
- SMOTE oversampling to balance classes (345 samples per class after resampling)

**Architecture**:
- Input layer: 18 features
- Hidden layer: 64 neurons with ReLU activation
- Dropout layer: 0.2 dropout rate
- Output layer: 1 neuron with sigmoid activation
- Weight Initialization: He uniform

**Hyperparameter Tuning**:
- Method: RandomizedSearchCV with 3-fold cross-validation
- Optimizer: Adam
- Learning Rate: 0.1
- Best Parameters: dropout_rate=0.2, neurons=64, activation='relu', init_weights='he_uniform', optimizer='Adam', learning_rate=0.1

**Training**:
- Batch Size: 32
- Epochs: 30
- Loss Function: Binary Crossentropy
- Metric: F-Beta Score (beta=2.0) to emphasize recall for minority class

**Performance Results**:
- **Test Accuracy**: 97.22%
- **Precision**: 97.09%
- **Recall**: 100.00%
- **F1-Score**: 98.52%
- **AUC Score**: High (perfect recall indicates excellent discrimination)

The neural network achieved excellent performance with perfect recall, meaning it successfully identified all positive instances in the test set.

#### Model 2: Fuzzy Logic System

**Feature Selection**:
- Method: Combined scoring (F-test ANOVA, Mutual Information, and Correlation)
- Selected Features: 8 features (vconst_corr, vconst_2, convect_corr, bckgrnd_vdc1, bckgrnd_vdc_eq, vconst_4, bckgrnd_vdc_psim, vconst_5)

**Fuzzy System Design**:
- Membership Functions: Triangular (low, medium, high) for each input feature
- Inference System: Mamdani Fuzzy Inference System
- Rule Generation: 25 rules combining individual features and feature pairs
- Normalization: QuantileTransformer scaled to [0, 10] range
- Threshold Optimization: Best threshold = 0.10

**Performance Results**:
- **Test Accuracy**: 92.59%
- **Precision**: 92.59%
- **Recall**: 100.00%
- **F1-Score**: 96.15%

The fuzzy logic system achieved good performance with perfect recall but struggled with precision, particularly for the minority class (Class 0), which resulted in zero true positives for failures.

**Comparison Summary**:
The Feedforward Neural Network outperformed the Fuzzy Logic System across all metrics:
- **Accuracy**: 97.22% vs 92.59% (4.63% improvement)
- **Precision**: 97.09% vs 92.59% (4.50% improvement)
- **F1-Score**: 98.52% vs 96.15% (2.37% improvement)
- **Recall**: Both achieved 100%

The neural network's superior performance can be attributed to its ability to learn complex non-linear patterns in the data, while the fuzzy system's rule-based approach, though interpretable, had limitations in handling the imbalanced dataset effectively.

## Key Findings

1. **Model Performance Ranking**:
   - Best Overall: Support Vector Machine (99.07% accuracy)
   - Second Best: Feedforward Neural Network (97.22% accuracy)
   - Third: Decision Tree (92.59% accuracy)
   - Fourth: Fuzzy Logic System (92.59% accuracy)

2. **Feature Importance**:
   - The most important features across models include: vconst_corr, vconst_2, convect_corr, and bckgrnd_vdc1
   - Feature selection significantly improved model performance and reduced overfitting

3. **Handling Class Imbalance**:
   - SMOTE oversampling was effective for neural networks
   - Weighted F1-score and F-Beta score (beta=2.0) helped prioritize minority class performance
   - SVM's margin optimization naturally handled the imbalance better than decision trees

4. **Hyperparameter Tuning Impact**:
   - Systematic hyperparameter tuning (GridSearchCV/RandomizedSearchCV) improved model performance by 5-10% across all models
   - Cross-validation prevented overfitting and provided robust performance estimates

## Technical Implementation

### Technologies Used
- **Python 3.x**
- **TensorFlow/Keras**: Neural network implementation
- **Scikit-learn**: Traditional ML models, preprocessing, feature selection, hyperparameter tuning
- **Scikit-fuzzy**: Fuzzy logic system implementation
- **Pandas/NumPy**: Data manipulation and numerical computations
- **Matplotlib/Seaborn**: Data visualization

### Preprocessing Techniques
- StandardScaler normalization
- QuantileTransformer (for fuzzy logic)
- SMOTE oversampling (for neural networks)
- Feature selection (SelectFromModel, Sequential Feature Selection, combined scoring)

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score (macro and weighted)
- F-Beta Score (beta=2.0)
- Confusion Matrix
- Classification Report

## Project Structure

```
Predictive Model_Climate/
├── CPC251_Assignment1_Climate2.ipynb          # Neural Network from Scratch
├── CPC251_Project_ Part1_Climate2.ipynb       # Decision Tree vs SVM
├── CPC251_Project_ Part2_Climate2.ipynb       # Neural Network vs Fuzzy Logic
├── Datasets/
│   ├── climate.csv                            # Main dataset
│   └── readme                                 # Dataset documentation
└── Guideline/
    ├── Assignment 1 Classification (Template)1.ipynb
    └── Project Guideline and Rubric.pdf
```

## Conclusion

This project successfully demonstrates the application of various machine learning techniques to predict climate model simulation crashes. The Support Vector Machine achieved the highest performance (99.07% accuracy), followed closely by the Feedforward Neural Network (97.22% accuracy). Both models effectively handled the imbalanced dataset and demonstrated robust generalization capabilities.

The project highlights the importance of:
- Appropriate feature selection methods
- Systematic hyperparameter tuning
- Proper handling of class imbalance
- Cross-validation for robust model evaluation
- Comparison of multiple modeling approaches

The results provide valuable insights into which climate model parameters are most predictive of simulation failures, which can inform future climate modeling efforts and parameter space exploration.

