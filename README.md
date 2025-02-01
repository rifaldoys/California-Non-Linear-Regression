# Non-Linear Regression with Advanced Techniques

## Project Overview
This project implements non-linear regression on the California Housing dataset, comparing four different methods:

1. **Decision Tree Regressor**
2. **Random Forest Regressor**
3. **XGBoost Regressor**
4. **Support Vector Machine (SVM) Regressor**

The project includes expert-level techniques such as hyperparameter tuning, model interpretation using SHAP, and 3D visualizations to analyze relationships between features and target variables.

---
## Steps & Implementation

### 1. Exploratory Data Analysis (EDA)
- **3D Surface Plot:** Visualizes the relationship between median income, house age, and price.
- **Non-linear Correlation Analysis:** Uses LOESS smoothing to detect complex feature interactions.
- **Feature Interaction Analysis:** Scatter plots with color-coded price values highlight interdependencies.
- **Feature Clustering:** K-Means clustering is applied to identify natural groupings in the data.

### 2. Advanced Preprocessing
- **Target Stratification:** Binning approach ensures balanced data distribution.
- **Adaptive Feature Scaling:** StandardScaler normalizes features before model training.
- **Feature Engineering:** Polynomial features are added to enhance non-linearity.

### 3. Model Development with Expert-Level Hyperparameter Tuning
- **Decision Tree:** Optimized max depth, min samples split, and feature selection.
- **Random Forest:** Fine-tuned number of trees, depth, and minimum leaf size.
- **XGBoost:** Bayesian optimization of learning rate, gamma, and tree depth.
- **SVM (RBF Kernel):** Optimized regularization, epsilon, and gamma values.

**Optimization Technique:**
- **Bayesian Search Optimization** is used instead of Grid Search for better efficiency and performance.

### 4. Model Evaluation & Interpretation
- **Metric Comparison:** RMSE and R² scores are visualized with error bars.
- **SHAP Analysis:** Identifies feature importance and interaction effects.
- **Learning Curve Analysis:** Examines model generalization using training & validation curves.
- **3D Partial Dependence Plot:** Displays relationships between selected features and predicted prices.
- **Residual Analysis:** Examines prediction errors to detect bias and variance.

### 5. Model Deployment Preparation
- **Pipeline Creation:** The best-performing model (XGBoost) is saved using `joblib`.
- **Model Card Generation:** Documents model performance, hyperparameters, and key insights for future reference.
- **Web Deployment (Future Work):** Model is prepared for deployment via Flask/FastAPI.

---
## Results Summary
- **Best Model:** XGBoost Regressor
- **Performance Metrics:**
  - **RMSE:** (Root Mean Squared Error) - Lower is better
  - **R² Score:** Higher is better
- **Feature Importance (Top 3):**
  1. Median Income
  2. Average Rooms
  3. Latitude

---
## Files & Structure
- `main.py`: Core implementation
- `best_nonlinear_model.pkl`: Saved model pipeline
- `model_card.md`: Summary of the best model
- `plots/`: Contains all generated visualizations

---
## Future Enhancements
- Implement deep learning-based regression models for further performance improvements.
- Extend SHAP analysis to compare multiple models.
- Deploy the model via a web API for real-time predictions.

---
## Acknowledgments
This project leverages powerful machine learning libraries such as Scikit-Learn, XGBoost, SHAP, and Plotly for comprehensive analysis and visualization.

---
## References
- Scikit-Learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- SHAP Library: https://shap.readthedocs.io/

