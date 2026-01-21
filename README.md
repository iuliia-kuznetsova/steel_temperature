# Prediction of Final Steel Temperature at a Steel Production Plant

A machine learning model for predicting molten steel temperature during ladle treatment to optimize energy consumption in steelmaking.

## Problem Statement

### Business Context

The metallurgical plant processes approximately 100-ton steel batches in refractory-lined ladles before continuous casting. Electricity for graphite electrode arc heating constitutes 25-35% of secondary steelmaking costs, with optimal temperature control directly impacting profitability.

### Current Challenges

- **Overheating**: Operators conservatively extend heating cycles → 8-12% excess energy consumption
- **Undershooting**: Insufficient heating → recycles (reheating + 15-20 min delay) → production bottlenecks
- **Variable heat loss**: Ladle brick wear, ambient conditions, idle times → unpredictable cooling rates
- **Energy cost volatility**: Electricity prices fluctuate → need precise energy budgeting per heat

### Financial Impact

1°C RMSE improvement = €15K-25K annual savings per furnace (based on 300 heats/month, €0.08/kWh).

### ML Value

Replace heuristic "heat until safe" rules with predictive temperature trajectories, enabling:

- Optimal heating schedules (reduce average cycle time 10-15%)
- Dynamic energy allocation across furnaces
- Reduced recycles (<2% vs current 5-7%)
- CO₂ footprint reduction

## Task

Predict the final steel temperature based on historical data from the steel melting process at a metallurgical plant.

## Results

The best model for predicting the final steel temperature is **CatBoost Regressor** with the following characteristics:
- **MAE on training dataset**: 5.61°C
- **MAE on test dataset**: 6.1°C
- **Model parameters**: iterations=981, learning_rate=0.044, l2_leaf_reg=0.67, depth=4

Key factors influencing final steel temperature:
1. Initial temperature of the alloy
2. Total heating duration
3. Production phase duration
4. Power factor coefficient
5. Wire material 1 volume
6. Bulk material 6 volume

## Quick Start

1. Clone the repository
2. Prepare virtual environment
    Create virtual environment
    ```bash 
    python3 -m venv venv_bak
    ```
    Activate virtual environment
    ```bash
    source venv_bak/bin/activate
    ```  # Linux/Mac

    or

    ```bash
    .\venv_bak\Scripts\activate
    ```   # Windows
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the data (see Data section) and place it in `./data` directory
4. Run the Jupyter notebook:
   ```bash
   jupyter notebook main_steel_temperature_prediction.ipynb
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

## Approach

1. **Exploratory Data Analysis (EDA)**: Understanding data structure, distributions, and relationships
2. **Data Preprocessing**:
   - Duplicate detection and removal
   - Outlier detection and handling
   - Missing value treatment
   - Data type corrections
3. **Feature Engineering**:
   - Aggregating data per batch (initial temperature, production stage data, final temperature)
   - Creating derived features (power factor, apparent power, work)
4. **Multicollinearity Analysis**: Identifying and removing highly correlated features
5. **Model Training and Selection**:
   - Ridge Regression (linear model with regularization)
   - CatBoost Regressor (gradient boosting)
   - Hyperparameter optimization with Optuna
6. **Model Evaluation**: Cross-validation and test set evaluation using MAE metric

## Project Structure

```
steel_temperature/
├── data/                    # Data files (not included in repo)
├── results/                 # Model outputs and results
├── main_steel_temperature_prediction.ipynb  # Main analysis notebook
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

## Environment Variables

The project uses environment variables for configuration:

- `DATA_DIR` - Path to data directory (default: `./data`)
- `RESULTS_DIR` - Path to results directory (default: `./results`)


## Author

**Iuliia Kuznetsova**  
January 2024
