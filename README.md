# SPE DSEATS Africa 2025 Well Classification Pipeline

This repository implements a comprehensive machine learning pipeline for oil and gas well classification and production trend analysis as part of the SPE Africa DSEATS Datathon 2025. The pipeline processes well production data from the provided datasets (spe_africa_dseats_datathon_2025_wells_dataset.csv and reservoir_info.csv), performs data cleaning, feature engineering using TSFresh for time-series features, generates production profiles and visualizations, and trains multi-output classifiers (Random Forest and XGBoost) to predict well attributes such as reservoir name/type, well type, production stability, and trends in GOR (Gas-Oil Ratio), watercut, and productivity index (PI).
The model achieves multi-label classification across 7 targets, with hyperparameter tuning via RandomizedSearchCV, stratified cross-validation, and comprehensive evaluation metrics (accuracy, precision, recall, F1-score). Outputs include classified well data, production profile plots, confusion matrices, feature importances, and reservoir oil summaries.
Features

Data Preprocessing: Cleans numeric columns, handles missing values, assigns wells to reservoirs based on pressure thresholds, and engineers derived metrics (e.g., GOR, watercut, PI, daily production).
Time-Series Feature Extraction: Uses TSFresh with EfficientFCParameters for automated feature extraction from production time series.
Trend Labeling: Generates ground truth labels for GOR (above/below solution GOR), watercut, and PI trends (Increasing/Decreasing/Flat/Combo).
Well Classification: Multi-output Random Forest and XGBoost models predict 7 categorical targets: Reservoir Name/Type, Well Type (GL/NL), Production Type (Steady/Unsteady), and 3 trends.
Hyperparameter Tuning: RandomizedSearchCV with stratified K-Fold CV, optimized for F1-weighted score.
Visualizations: 
Individual well production profiles (oil/gas/water, GOR, watercut, PI).
Distribution histograms, correlation heatmaps, boxplots, pairplots, and KDE plots.
Confusion matrices and top-10 feature importance bar charts for each target.


Evaluation & Reporting: Detailed classification reports, metrics summary table, and overall performance bar chart.
Outputs: 
Classified CSV: TeamName_DSEATS_Africa_2025_Classification.csv
Production profiles: PNG plots in /production_profiles/
Metrics: CSVs and plots in /classification_metrics/
Reservoir oil totals summary.



Prerequisites

Python 3.8 or higher
Input datasets: spe_africa_dseats_datathon_2025_wells_dataset.csv and reservoir_info.csv (place in the repo root)
Required libraries: pandas, numpy, matplotlib, seaborn, tsfresh, scikit-learn, xgboost

Installation

Clone the repository:git clone https://github.com/yourusername/SPE-DSEATS-Africa-2025-Well-Analysis.git
cd SPE-DSEATS-Africa-2025-Well-Analysis


Create a virtual environment (recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Ensure datasets are in the root directory.

Usage

Run the Pipeline:
python well_analysis_pipeline.py

The script will:

Load and clean data, assign reservoirs, engineer features.
Extract TSFresh features and manual slopes.
Generate ground truth trends and train/test split.
Train and tune Random Forest (multi-output) and XGBoost models.
Produce predictions, evaluations, and visualizations.
Save outputs to directories and console summaries.


Expected Outputs:

Console: Model performance metrics, best hyperparameters, total oil per reservoir.
Files: Classified CSV, production profile PNGs (one per well), confusion matrices, feature importance plots, metrics CSV.
Example Console Snippet:Generated 50 production profile plots
Best hyperparameters: {'estimator__n_estimators': 200, ...}
Accuracy: 0.850 | Precision: 0.845 | Recall: 0.852 | F1-Score: 0.848
Total Oil per Reservoir:
Reservoir A: 1,234,567 STB
...
PROCESSING COMPLETE!




Customization:

Update Y_cols for different targets.
Adjust param_dist for hyperparameter grids.
Modify trend labeling thresholds in label_trend or label_gor_trend.
Add team name in output CSV filename.



Repository Structure
SPE-DSEATS-Africa-2025-Well-Analysis/
│
├── well_analysis_pipeline.py   # Main pipeline script
├── requirements.txt            # Dependencies
├── spe_africa_dseats_datathon_2025_wells_dataset.csv  # Input data
├── reservoir_info.csv          # Input data
├── production_profiles/        # Generated well profile plots
├── classification_metrics/     # Metrics CSVs and plots
├── TeamName_DSEATS_Africa_2025_Classification.csv  # Final output
└── README.md                  # This file

Requirements
See requirements.txt for pinned versions:
pandas==2.2.3
numpy==1.26.4
matplotlib==3.9.2
seaborn==0.13.2
tsfresh==0.20.2
scikit-learn==1.5.2
xgboost==2.1.1

Model Performance Highlights

Targets: 7 multi-class outputs with balanced class weights.
Evaluation: F1-weighted scoring; typical accuracies 80-90% across targets.
Key Features: TSFresh-extracted time-series stats (e.g., slopes, kurtosis) and manual derivations (e.g., GOR slope) dominate importances.
Reservoir Insights: Aggregates cumulative oil production by predicted reservoir for datathon submission.

Contributing
Contributions welcome! For improvements (e.g., advanced CV, ensemble models):

Fork the repository.
Create a branch: git checkout -b feature/enhanced-cv.
Commit changes: git commit -m 'Add stratified group CV'.
Push: git push origin feature/enhanced-cv.
Open a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

SPE Africa DSEATS Datathon 2025 organizers.
TSFresh for time-series feature extraction.
Scikit-learn and XGBoost for ML implementations.

Contact
For questions, open an issue or contact [your email or GitHub handle]. Good luck in the datathon!
