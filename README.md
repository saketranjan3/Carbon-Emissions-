# Carbon-Emissions-
This repository is the beginning of the Carbon Emission Intensity Modeling in Freight Operations using eXplainable AI in Real-World Conditions
## Prerequisites

- Python 3.10
- Git

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/saketranjan3/Carbon-Emissions-]
cd CO2_pred
```

2. Create and activate a Python virtual environment:
```bash
python -m venv venv
venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your .xlsx data files in the `datasets` folder
   - A sample dataset is provided for testing
   - Please make sure your data file is in the same format as sample data, including sheet and variable names

2. Run the evaluation script:
```bash
python eval.py
```

## Training the Model

### Dataset Structure
The training dataset should be organized in the following structure:
```
xgb_project/
├── data/
│   ├── train.csv          # training dataset
│   ├── valid.csv          # validation dataset
│   └── test.csv           # test dataset for prediction
│
├── models/
│   └── xgb_model.json     # saved trained XGBoost model
│
├── scripts/
│   ├── train_xgb.py       # script for training with early stopping
│   ├── predict.py         # script for making predictions on test data
│   ├── shap_analysis.py   # script for SHAP explainability
│   └── pdp_analysis.py    # script for Partial Dependence Plots
│
├── outputs/
│   ├── predictions.csv    # predictions on test set
│   ├── metrics.json       # evaluation metrics (RMSE, R², MAE)
│   ├── shap_summary.png   # SHAP summary plot
│   ├── shap_dependence.png# SHAP dependence plot for key features
│   ├── pdp_speed.png      # PDP for speed
│   ├── pdp_load.png       # PDP for load
│   └── pdp_accel.png      # PDP for acceleration/deceleration
│
└── data.yaml             # documentation of workflow
```

The `data.yaml` file should contain:
```yaml
names:
- co2 pred
nc: 1
path: [path to training_dataset folder]
train: images/train
val: images/val
```

### Training Process
1. Prepare your dataset following the structure above
2. Run the training script:
```bash
python train.py --models/xgb_co2.json--data ./training_dataset/data.yaml --epochs 100 --batch-size 2
```

Optional arguments:
- `--model`: Path to the model (default: xgb_co2.pt)
- `--data`: Path to the data yaml file (default: ./training_dataset/data.yaml)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 2)
- `--validate`: Run validation after training
- `--val-model`: Path to model for validation (defaults to best training weights)
- seed: Random seed (default: 42)
- cv-folds: Cross-validation folds (default: 5)
- metric: Eval metric (default: rmse)

⸻

MLR (Ordinary Least Squares)
- model mlr
- fit-intercept: Whether to fit intercept (default: true)
- normalize: Feature standardization before fit (default: false)

No regularization (plain OLS), per your table.

⸻

LASSO
- model lasso
- lasso-alpha: L1 penalty strength (range: 1e-5…1e-1; default: 0.0012)
- max-iter: Max iterations (default: 10000)
- tol: Convergence tolerance (default: 1e-4)

⸻

Bayesian Ridge
- model bayes_ridge
- alpha1: Gamma prior over weights shape (default: 1e-6)
- alpha2: Gamma prior over noise shape (default: 1e-4)
- lambda1: Prior over weight precision (default: 1e-4)
- lambda2: Prior over noise precision (default: 1e-6)
- tol: Convergence tolerance (default: 1e-4)
- n-iter: Max iterations (default: 300)

⸻

Random Forest (Regressor)
- model rf
- rf-n-estimators: Number of trees (range: 50…300; default: 200)
- rf-max-depth: Max depth per tree (range: 3…20; default: 19; None = unlimited)
- rf-min-samples-split: Min samples to split (int range: 2…20; default: 6)
- rf-min-samples-leaf: Min samples at leaf (int range: 1…10; default: 1)
- rf-max-features: Features per split (auto|sqrt|log2|float 0–1; default: sqrt)
- rf-bootstrap: Use bootstrap samples (default: true)
 
⸻

XGBoost (Regressor) 
- model xgb
- xgb-eta: Learning rate η (range: 0.01…0.30; default: 0.07)
- xgb-n-rounds: Boosting rounds/trees (range: 50…300; default: 250)
- xgb-gamma: Min loss reduction for split (range: 0…0.5; default: 0.048)
- xgb-max-depth: Max tree depth (range: 3…10; default: 7)
- xgb-subsample: Row subsample (range: 0.5…1.0; default: 0.6)
- xgb-colsample-bytree: Column subsample per tree (range: 0.5…1.0; default: 0.8)
- xgb-objective: (default: reg:squarederror)
- xgb-eval-metric: (default: rmse)
- xgb-early-stopping: Early stopping rounds (default: 100)
- xgb-tree-method: (hist|gpu_hist; default: hist)

The training results including newly trained model will be saved in the `training_result` folder.

## Results

The script will generate:
- CO2 event detection results
- Emission trends versus speed (e.g., high-speed clusters, non-linear variability)
- Impact of acceleration/deceleration on transient emission spikes
- Influence of vehicle weight/load on average CO₂ intensity
- Effect of air–fuel ratio (AFR) on combustion efficiency and emissions
- Role of altitude variations on instantaneous CO₂ output

## Output Files

After running the evaluation script, the following output files will be generated for each site in the `results` folder:

### Real-world operational dynamics
- Relationship between fuel usage rates and emission intensity
- Comparison of engine displacement with overall emission profiles
- Located in the `results_plot` subfolder
- If mode plotting is enabled, additional plots with mode lines will be in `results_plot_with_mode`

### Model outputs
- Trained XGBoost model (.json)
- Predictions on the test dataset (predictions.csv)
- Evaluation metrics (RMSE, MAE, R²)

### Explainability outputs
- Feature importance rankings
- SHAP summary and dependence plots
- Partial Dependence Plots (PDPs) for major predictors

Note: Empty cells in the acceleration columns indicate negative acceleration or deceleration.

## Author
- Saket Ranjan, Shiva Nagendra S. M.
## Correspondance
- Shiva Nagendra S. M.*
## Affiliation
- Environmental Engineering Division, Department of Civil Engineering, Indian Institute of Technology Madras, Chennai -600036, India.

For any queries, please contact:
- Email: saket.ranjan1331@gmail.com
- CC: ce20d095@smail.iitm.ac.in, snagendra@civil.iitm.ac.in
