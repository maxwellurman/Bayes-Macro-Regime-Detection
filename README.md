# U.S. Macroeconomic Regime Detection

## Overview
What question are you investigating?

## Data
### Raw Data Summary:
| Variable                          | FRED Series   | Frequency    | Seasonally Adjusted             | Start Date   | End Date   |
|:----------------------------------|:--------------|:-------------|:--------------------------------|:-------------|:-----------|
| CPI                               | CPIAUCSL      | Monthly      | Seasonally Adjusted             | 1947-01-01   | 2026-07-01 |
| Unemployment Rate                 | UNRATE        | Monthly      | Seasonally Adjusted             | 1948-01-01   | 2026-07-01 |
| Federal Funds Rate                | FEDFUNDS      | Monthly      | Not Seasonally Adjusted         | 1954-07-01   | 2026-07-01 |
| Yield Curve Spread                | T10Y2Y        | Daily        | Not Seasonally Adjusted         | 1976-06-01   | 2026-08-13 |
| Industrial Production             | INDPRO        | Monthly      | Seasonally Adjusted             | 1919-01-01   | 2026-06-01 |
| Housing Starts                    | HOUST         | Monthly      | Seasonally Adjusted Annual Rate | 1959-01-01   | 2026-06-01 |
| Personal Consumption Expenditures | PCE           | Monthly      | Seasonally Adjusted Annual Rate | 1959-01-01   | 2026-06-01 |
| VIX                               | VIXCLS        | Daily, Close | Not Seasonally Adjusted         | 1990-01-02   | 2026-08-12 |
| Real GDP                          | GDPC1         | Quarterly    | Seasonally Adjusted Annual Rate | 1947-01-01   | 2026-04-01 |

### Cleaning Steps
1. Create monthly variables for VIX by taking the monthly average.
2. Create monthly variable for Yield Curve Spread by taking the month end value.
3. Move quarterly, real, seasonally adjusted GDP to its own table (Real GDP is used to evaluate the model, not as an observed variable).
4. Limit all variables to January 1990 to December 2025.

### Feature Engineering
1. Create roughly stationary time series variables:
    - Raw CPI -> year-over-year percent change
    - Raw Industrial Production -> year-over-year, quarter-over-quarter, and month-over-month percent change
    - Raw Housing Starts -> First create rolling, 3-month trailing average.  Housing Starts are noisy month over month.  Then, calculate year-over-year, quarter-over-quarter, and month-over-month percent change in the rolling average metric.
    - Unemployment Rate -> year-over-year, quarter-over-quarter, and month-over-month change (since already a percentage)
2. Log transform of VIX. VIX is strongly right-skewed.
3. **Discretize** the variables into 3 quantiles

### Final Analytical Dataset
Note: We include both the annual change and monthly change series for Industrial Production, Smoothed Housing Starts, and Unemployment Rate to account for short term and long term momentum.  For example, coming out of a recession, year-over-year change in Smoothed Housing Starts might be flat or negative, but month-over-month change might be positive.


### Example Time Series Plots


## Methodology No. 1: Discrete Emissions
What model/algorithm are you using?

## Results for Methodology No. 1: Discrete Emissions
What did you find?

## Methodology No. 2: Continuous Emissions
What model/algorithm are you using?

## Results for Methodology No. 2: Continuous Emissions
What did you find?

## Comparison of Approaches
[[__]]

## Repository Structure
What are the important files?

## Installation / Requirements
What packages are required?

## Running the Analysis
What should someone run and in what order?
