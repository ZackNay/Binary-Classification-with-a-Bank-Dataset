# Binary Classification with a Bank Dataset

LightGBM Solution with Exhaustive Hyperparameter Optimization for Kaggle Playground Series S5E

## Competition Overview

**Objective:** Binary classification on synthetically-generated banking data  
**Evaluation Metric:** ROC AUC Score  
**Timeline:** August 26-31, 2025

## Summary

This solution achieves competitive performance primarily through exhaustive hyperparameter optimization rather than feature engineering. Since I joined the competition only 5 days before the submission deadline, I prioritized tuning the LightGBM configuration, running over 10,000 Optuna trials with rigorous cross-validation. This process yielded a robust GBDT model capable of effectively handling the dataset’s class imbalance and categorical features without additional preprocessing. Although resource constraints forced me to stop training early, the approach still resulted in an okay final standing of /.

## Approach

### Two-Stage Strategy

**Stage 1: Exhaustive Hyperparameter Search**
- Utilized Optuna with up to 10,000 trials to find optimal LightGBM parameters
- Implemented 5-fold stratified cross-validation for robust evaluation
- Tested all three LightGBM boosting types (GBDT, DART, GOSS) with their specific parameters
- Applied early stopping after 1,000 minimum trials with 150-trial patience window
- Used TPE sampler with 500 startup trials and median pruning for efficient search

**Stage 2: Final Model Training**
- Trained model using optimized parameters from Stage 1
- Validated on 15% hold-out set to confirm performance
- Calculated optimal iteration count from CV results plus 10% safety margin
- Trained final model on full dataset for submission

### Key Technical Decisions

1. **No Feature Engineering:** Focused entirely on model optimization due to time constraint rather than feature creation, relying on LightGBM's native categorical handling
2. **Class Imbalance:** Used `is_unbalance=True` instead of synthetic sampling
3. **Comprehensive Search Space:** Included all LightGBM parameters affecting performance
4. **Memory Management:** Implemented garbage collection after each trial to handle long-running optimization

## Results

**Best Configuration:**
- Boosting Type: GBDT
- Estimators: 2,640
- Learning Rate: 0.0216
- Max Depth: 10 / Num Leaves: 148
- Regularization: L2 = 6.48, L1 = 2.4e-06

## Project Structure

```
├── train.csv                       # Training data
├── test.csv                        # Test data
├── train.ipynb                     # Main notebook (optimization + training)
├── submission_probabilities.csv    # Final predictions
├── feature_importance_final.csv    # Feature rankings
├── final_lgbm_model_optimized.txt  # Saved model
└── training_report.json            # Training metrics
```

## Requirements

- lightgbm
- optuna
- pandas, numpy, scikit-learn

## Usage

1. Run first cell for hyperparameter optimization
2. Run second cell for final training and submission generation

## Citation

Walter Reade and Elizabeth Park. Binary Classification with a Bank Dataset. https://kaggle.com/competitions/playground-series-s5e