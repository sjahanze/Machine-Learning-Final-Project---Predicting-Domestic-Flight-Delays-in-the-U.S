# Predicting Domestic Flight Delays in the U.S using XG Boost

## Overview
This project builds machine learning models to predict whether a U.S. domestic flight will be delayed upon arrival. A flight is classified as delayed if it arrives more than 15 minutes late, following the FAA standard definition. Flight delays cost the aviation industry approximately $33 billion annually in lost productivity, crew misalignment, and passenger disruption.

## Author
Shahzaib Jahanzeb

## Dataset
The dataset comes from the U.S. Department of Transportation's Bureau of Transportation Statistics (via Kaggle), containing detailed operational records for U.S. domestic flights from 2019 to 2023. A working sample of 150,000 flights was used for modelling.

### Features
- Scheduled departure and arrival times
- Actual elapsed time and air time
- Taxi-in and taxi-out durations
- Distance flown
- Day of week and month of travel
- Airline carrier indicators
- Delay cause indicators (weather, carrier, NAS, security, late aircraft)

## Methodology

### Data Preprocessing
- Removed cancelled and diverted flights
- Created binary target variable (delayed > 15 minutes)
- Applied cyclical encoding for departure hour
- One-hot encoded airline carriers
- Filtered for routes with at least 30 flights

### Models Implemented
Three supervised learning approaches were evaluated:

1. **Bagging Classifier** (Baseline)
   - 100 decision tree estimators with bootstrap sampling
   - Reduces variance through averaging

2. **XGBoost** (Default Parameters)
   - Sequential boosting to correct errors from previous iterations
   - Binary logistic objective with AUC evaluation

3. **Tuned XGBoost**
   - Hyperparameter tuning via 5-fold cross-validation
   - Optimised learning rate (0.05) and boosting rounds (503)

## Results

| Model | Accuracy | Test Error | AUC |
|-------|----------|------------|-----|
| Bagging Classifier | 84.31% | 0.1569 | 0.720 |
| XGBoost (Default) | 85.22% | 0.1478 | 0.746 |
| XGBoost (Tuned) | 85.38% | 0.1462 | 0.753 |

The tuned XGBoost model achieved the best performance across all metrics.

## Key Findings (SHAP Analysis)
The most influential features for predicting delays:
- **CRS_ELAPSED_TIME** - Scheduled flight duration
- **ELAPSED_TIME** - Actual flight duration
- **AIR_TIME** - Time spent in the air
- **TAXI_OUT** - Ground time before takeoff (indicates airport congestion)
- **DEP_HOUR** - Departure hour (delays accumulate throughout the day)
- **AIRLINE** - Carrier-specific operational patterns

## Tools and Libraries
- Python
- Pandas, NumPy
- Scikit-learn (BaggingClassifier, train_test_split, metrics)
- XGBoost
- SHAP (model interpretability)
- Matplotlib, Seaborn, Plotnine (visualisation)
- Google Colab

## Files
- `predicting_domestic_flight_delays_in_the_u_s.py` - Main analysis script
- `Final_Report.pdf` - Detailed project report with methodology and findings

## Limitations
- Class imbalance between on-time and delayed flights
- Missing real-world variables (weather conditions, runway closures, airport congestion levels)
- Many delay causes arise from factors not captured in the dataset

## Future Improvements
- Incorporate weather data and real-time airport congestion metrics
- Explore network-based features (upstream delays)
- Apply techniques to address class imbalance (SMOTE, class weights)

## References
1. Airlines for America. (2024). U.S. Passenger Carrier Delay Costs
2. Guo, S., et al. (2024). A hybrid machine learning-based model for predicting flight delay
3. Li, X., & Jing, Z. (2021). Generation and prediction of flight delays in air transport
4. U.S. Department of Transportation - Bureau of Transportation Statistics
