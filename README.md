# Prediction of Final Steel Temperature at a Steel Production Plant

Build a model to predict **final molten steel temperature** during ladle treatment in order to **optimize energy consumption in steelmaking**.
As a result using the best model (tuned CatBoost Regressor with 6.0°C test MAE) for heating strategy could lead to **66% temperature error reduction** comparing with rule-based heats, causing the business effect of production cost reduction of 487.8K€/year.


## Problem Statement

### Business Context

The metallurgical plant processes ~100-ton steel batches in refractory-lined ladles before continuous casting. Electricity for graphite electrode arc heating accounts for 25-35% of secondary steelmaking costs. 
Operators must maintain steel within a narrow temperature window (typically ±5–10°C). Lower temperature causes recycles and delays. Higher temperature leads to wasted energy and faster ladle wear. Current heuristic rules ('heat until safe') result in systematic overheating and unpredictable cycle times. Thus, precise heating within set time could minimize energy waste and therefore reduce production costs while ensuring quality for continuous casting.

Example Heat Cost: 300 kWh electricity at €0.08/kWh = €24 per heat. At 300 heats/month/furnace, 10% savings = €7.2K/month = €86K/year per furnace.

### ML Objective

Predict the final steel temperature based on historical data from the steel melting process at a metallurgical plant in order to enable optimal heating trajectories and schedules, cutting cycle time and recycles.


## Project Structure

```
steel_temperature/
├── venv_steel/                                 # Virtual environment (not included in repo)
├── data/                                       # Data files (not included in repo)
├── results/                                    # Model outputs and results
├── main_steel_temperature_prediction.ipynb     # Main analysis notebook
├── requirements.txt                            # Python dependencies
├── .gitignore
└── README.md
```


## Libraries

- **Pandas** - data manipulation and analysis
- **NumPy** - numerical computations
- **Matplotlib & Seaborn** - data visualization
- **Scikit-learn** - machine learning utilities and Ridge Regression
- **PyOD** - outlier detection (KNN)
- **Optuna** - hyperparameter optimization
- **CatBoost** - gradient boosting model


## Data

The dataset contains public data from a steel production plant and is used for educational purposes only.

### Download Links

The data can be downloaded from the following sources and should be placed into `./data` directory:

- [data_arc_new.csv](https://drive.google.com/file/d/1Uc2WbhW9U5-TtLr8X82QQxk36J9CMNCu/view?usp=sharing)
- [data_bulk_new.csv](https://drive.google.com/file/d/1LtAejlRIp5xm736IJEcteEixNapV95pd/view?usp=sharing)
- [data_bulk_time_new.csv](https://drive.google.com/file/d/1hYBVDx2I5WtIDlDKkkmztu2ZvqKYPB9J/view?usp=sharing)
- [data_gas_new.csv](https://drive.google.com/file/d/1GSdUxiW0iKIm9r_0crC4_HjpJEi2Id1U/view?usp=sharing)
- [data_temp_new.csv](https://drive.google.com/file/d/1DE_OKJ9NvheG5x1PXzRngy4LCenpaZM-/view?usp=sharing)
- [data_wire_new.csv](https://drive.google.com/file/d/1Muzt0mFQNLYlJPoET8mm1fz2icTyOr6i/view?usp=sharing)
- [data_wire_time_new.csv](https://drive.google.com/file/d/1ibkWLU7GmAMenZkwDJ6_z-8gMbN-2kyF/view?usp=sharing)

### Dataset Descriptions

| File | Description |
|------|-------------|
| `data_arc_new.csv` | Electrode data (arc heating power) |
| `data_bulk_new.csv` | Bulk materials supply (volume) |
| `data_bulk_time_new.csv` | Bulk materials supply (time) |
| `data_gas_new.csv` | Gas purging data |
| `data_temp_new.csv` | Temperature measurements |
| `data_wire_new.csv` | Wire materials data (volume) |
| `data_wire_time_new.csv` | Wire materials data (time) |

**Note**: The `key` column in all datasets contains the batch number. Multiple rows may have the same `key` value, indicating that the corresponding batch underwent several processing iterations.


## Quick start

1. Clone the repository
```bash
git clone <your-repo-url>
cd promo_uplift
```

2. Prepare a virtual environment

Create environment:
```bash
python -m venv .venv_steel
```

Activate environment:
```bash
source .venv_steel/bin/activate
``` 
*(Linux/Mac)*

or

```bash
.\.venv_steel\Scripts\activate
```
*(Windows)*

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the data (see **Data**) and place it into `./data`

6. Run the Jupyter notebook
- Open `main_steel_temperature_prediction.ipynb`
- Select kernel **Python (.venv_steel)**
- Run all cells

7. Stopping services
```bash
# Stop virtual environment
deactivate

# Stop MLflow server
# (Ctrl+C in the terminal where MLflow is running)
```


## Approach

1. **Exploratory Data Analysis (EDA)**: 
   - Understanding data structure, distributions, and relationships.

2. **Data Preprocessing**:
   - Duplicate detection and removal;
   - Outlier detection and handling;
   - Missing value treatment;
   - Data type corrections.

3. **Feature Engineering**:
   - Aggregating data per batch (initial temperature, production stage data, final temperature);
   - Creating derived features (power factor, apparent power, work).

4. **Multicollinearity Analysis**: 
   - Identifying and removing highly correlated features.

5. **Model Training and Selection**:
   - Ridge Regression (linear model with regularization);
   - CatBoost Regressor (gradient boosting);
   - Hyperparameter optimization with Optuna.

6. **Model Evaluation**: 
   - Cross-validation and test set evaluation using MAE metric.


## Results

Model's test MAE is 6.0°C: predictions are off by 6.0°C on average.

From dataset analysis temperature standart error is 17.7°C* for rule-based heats
Thus, using model for heating strategy could lead to 66% temperature error reduction 
and business effect of production cost reduction of 487.8K€/year:

100-ton ladle × 0.68 kWh/ton/°C = 68 kWh/°C/batch
68 kWh × 0.07 €/kWh = 4.76 €/°C/batch
4.76 €/°C/batch × (17.7 - 6) °C = 55.69 €/batch
55.69 €/batch × 26 batches/day** × 365 days/year = 487.8K€/year

*Average positive temperature difference from target (1600°C): 17.69
**Average number of unique keys per day: 25.74


## Futher improvements
The project is done for educational purposes only. For production deployment, more complex model architectures should be used to implement dynamic optimization instead of static temperature prediction. For example:
- **Model Predictive Control (MPC)** using fitted Ridge/CatBoost model as base predictor, optimizing real-time electrode power trajectories every 30 seconds;
- **Model Ensembling** with Online Learning combining Ridge (fast baseline), CatBoost (feature interactions), and LSTM (temporal dynamics) with weekly retraining.


## Author
**Iuliia Kuznetsova**  
January 2024
