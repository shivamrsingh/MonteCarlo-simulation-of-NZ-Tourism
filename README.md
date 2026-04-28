# NZ Tourism Economic Impact — Monte Carlo Simulation Model
 
**Version:** 2.0 (Production)  
**Author:** Shivam  
**Language:** Python 3.x
 
---
 
## Overview
 
A production-grade Monte Carlo simulation model analysing the impact of tourism fluctuations on the New Zealand economy. Simulates 10,000+ scenarios using Geometric Brownian Motion and links tourism outcomes to GDP via OLS regression.
 
---
 
## Data Sources (Official NZ Government)
 
| Dataset | Source | URL |
|---|---|---|
| Tourism Satellite Account (TSA) 2023 | Stats NZ | https://www.stats.govt.nz/information-releases/tourism-satellite-account-2023 |
| National Accounts (GDP) | Stats NZ | https://www.stats.govt.nz/topics/national-accounts |
| International Visitor Arrivals (IVA) | MBIE | https://www.mbie.govt.nz/immigration-and-tourism/tourism-research-and-data/ |
| Key Tourism Statistics April 2024 | MBIE | https://www.mbie.govt.nz/assets/tourism-key-statistics-april-2024.pdf |
 
---
 
## Installation
 
```bash
pip install pandas numpy scipy matplotlib seaborn statsmodels
```
 
---
 
## Usage
 
```bash
python nz_tourism_monte_carlo.py
```
 
Outputs are written to `/mnt/user-data/outputs/nz_tourism_mc/`
 
---
 
## Model Design
 
### Stochastic Process — Geometric Brownian Motion
```
T(t+1) = T(t) × exp((μ − 0.5σ²)Δt + σ√Δt × Z)
```
- **μ** = 5.85% p.a. (log drift, pre-COVID Stats NZ TSA)
- **σ** = 4.11% (log volatility)
- **Z** ~ N(0,1)
- **N** = 10,000 simulations, 10-year horizon
### Econometric Model — OLS Regression
```
GDP = 195.9 + 2.637 × Tourism_Expenditure
```
- R² = 0.127
- Elasticity (log-log) = 0.180
- Interpretation: 1% rise in tourism → 0.18% rise in GDP
---
 
## Output Files
 
| File | Description |
|---|---|
| `nz_tourism_monte_carlo.py` | Full Python source code |
| `fig1_fan_chart.png` | GBM paths fan chart (10,000 paths) |
| `fig2_distributions.png` | Tourism & GDP outcome distributions |
| `fig3_econometrics.png` | OLS regression panels |
| `fig4_scenario_comparison.png` | Negative shock / base / boom comparison |
| `fig5_sensitivity.png` | σ × μ sensitivity heatmap |
| `fig6_ci_bands.png` | Historical + forecast with CI bands |
| `mc_simulation_results.csv` | All 10,000 simulation outcomes |
| `scenario_summary.csv` | Scenario-level summary statistics |
| `sensitivity_analysis.csv` | Sensitivity grid results |
| `historical_data.csv` | Official NZ historical data (2008–2023) |
 
---
 
## Key Results
 
| Scenario       | Probability | Year-10 Tourism | GDP Impact   | Employment   |
|----------------|-------------|-----------------|--------------|--------------|
| Negative Shock | 15.7%       | NZD 68.7B       | −NZD 28B     | −147k FTEs   |
| Base Case      | 69.0%       | NZD 83.1B       | +NZD 10B     | +243k FTEs   |
| Boom           | 15.3%       | NZD 100.2B      | +NZD 55B     | +357k FTEs   |
 
**Value at Risk (P5 GDP):** −NZD 33.3B  
**95% CI for Year-10 Tourism:** [NZD 64B, NZD 108B]
 
---
 
## Baseline (2023 Official Data)
 
- Total Tourism Expenditure: **NZD 46.6B** (Stats NZ TSA 2023)
- GDP: **NZD 405B** (Stats NZ National Accounts)
- Visitor Arrivals: **3.15M** (MBIE IVA)
- Tourism/GDP Share: **5.9%** (Stats NZ TSA 2023)
- Tourism Employment: **310,000 FTEs** (Stats NZ TSA 2023)
 
