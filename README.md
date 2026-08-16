# U.S. Macroeconomic Regime Detection

## Overview
We investigate regime changes and latent states in the US economy using Hidden Markov Models.

## Data
### Data Source:
We are using data macroeconomic data available from the Federal Reserve Bank of St. Louis (“FRED”).  The data is publicly available through FRED’s API.

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

### Cleaning Steps:
1. Create monthly variables for VIX by taking the monthly average.
2. Create monthly variable for Yield Curve Spread by taking the month end value.
3. Move quarterly, real, seasonally adjusted GDP to its own table (Real GDP is used to evaluate the model, not as an observed variable).
4. Limit all variables to January 1990 to December 2025.

### Feature Engineering:
1. Create roughly stationary time series variables:
    - Raw CPI -> year-over-year percent change.
    - Raw Industrial Production -> year-over-year, quarter-over-quarter, and month-over-month percent change.
    - Raw Housing Starts -> First create rolling, 3-month trailing average.  Housing Starts are noisy month over month.  Then, calculate year-over-year, quarter-over-quarter, and month-over-month percent change in the rolling average metric.
    - Unemployment Rate -> year-over-year, quarter-over-quarter, and month-over-month change (since already a percentage).
2. Create Real PCE level by deflating Nominal PCE using the CPI. Then, create year-over-year percent change in Real PCE.
3. Log transform of VIX. VIX is strongly right-skewed.
4. **Discretize** the variables into 3 quantiles for the discrete emission analysis.

### Final Analytical Dataset
Note: We include both the annual change and monthly change series for Industrial Production, Smoothed Housing Starts, and Unemployment Rate to account for short term and long term momentum.  For example, coming out of a recession, year-over-year change in Smoothed Housing Starts might be flat or negative, but month-over-month change might be positive.
|   Variable |   Frequency |   Description |
|-------------------:|-----------------:|------:|
|                  Yield_Curve_Spread_MonthEnd |          Monthly | Month End Value |
|                  CPI_YOY |          Monthly | Percent Change YoY |
|                  Industrial_Production_YoY |          Monthly | Percent Change YoY |
|                  Industrial_Production_MoM |          Monthly | Percent Change MoM |
|                  Housing_Starts_YoY |          Monthly | Percent Change YoY in Smoothed Trailing 3-Mo Avg. |
|                  Housing_Starts_MoM |          Monthly | Percent Change MoM in Smoothed Trailing 3-Mo Avg. |
|                  Unemployment_Rate_YoY_Change |          Monthly | Change YoY |
|                  Unemployment_Rate_MoM_Change |          Monthly | Change MoM |
|                  Real_PCE_YoY |          Monthly | Percent Change YoY |
|                  Log_VIX_MonthAvg |          Monthly | Log of Monthly Average of Daily VIX |


### Example Time Series Plots:
<img width="9538" height="7080" alt="time_series_plots" src="https://github.com/user-attachments/assets/832e7c75-1e63-4cee-9c63-6e1a7e4eac4b" />

## Methodology No. 1: Discrete Emissions
### Hidden Markov Model Set Up
Hidden Markov Model with K hidden states representing macroeconomic regimes:

$$S_t \in \{1, \ldots, K\}$$

Each observed variable X_i is discretized into one of three categories:

$$X_{i,t} \in \{\text{Low}, \text{Middle}, \text{High}\}$$

Hidden Markov Model requires:

- An Initial State Distribution...

$$\boldsymbol{\pi}=\begin{bmatrix}
P(S_0 = 1) \\
P(S_0 = 2) \\
\vdots \\
P(S_0 = K)
\end{bmatrix}$$

- A Transition Matrix...

$$\mathbf{A}=\begin{bmatrix}
P(S_t=0 \mid S_{t-1}=0) & \cdots & P(S_t=K \mid S_{t-1}=0) \\
P(S_t=0 \mid S_{t-1}=1) & \cdots & P(S_t=K \mid S_{t-1}=1) \\
\vdots & \ddots & \vdots \\
P(S_t=0 \mid S_{t-1}=K) & \cdots & P(S_t=K \mid S_{t-1}=K)
\end{bmatrix}$$

- Emission Matrices for every variable...

$$\mathbf{B}_i=\begin{bmatrix}
P(X_{i,t}=\text{Low} \mid S_t=0) &
P(X_{i,t}=\text{Middle} \mid S_t=0) &
P(X_{i,t}=\text{High} \mid S_t=0) \\
P(X_{i,t}=\text{Low} \mid S_t=1) &
P(X_{i,t}=\text{Middle} \mid S_t=1) &
P(X_{i,t}=\text{High} \mid S_t=1) \\
\vdots & \vdots & \vdots \\
P(X_{i,t}=\text{Low} \mid S_t=K) &
P(X_{i,t}=\text{Middle} \mid S_t=K) &
P(X_{i,t}=\text{High} \mid S_t=K)
\end{bmatrix}$$

### The Baum-Welch Algorithm
- The Baum-Welch algorithm is an expectation maximization algorithm used to train HMM's by iteratively updating the transition and emission matrices and initial state distribution to maximize the likelihood of a set of observed sequences when hidden states are unobserved.

- Therefore, rather than set a fixed initial state distribution, transition matrix, and emission matrices, the model learns these from the data.
  
- The number of hidden states is set, not learned by the algorithm.

- The algorithm has several steps. Without getting into all the details, we instead highlight what is being maximized:

The Baum-Welch algorithm estimates the model parameters

$$\theta = \{\boldsymbol{\pi}, \mathbf{A}, \mathbf{B}\}$$

by iteratively maximizing the likelihood of the observed data:

$$\theta^*=\arg\max_{\theta}P(X_{1:T} \mid \theta)$$
  
- To start the algorithm, the initial state distribution, transition matrix, and emission matrices need to be initialized. The final results are influences by the initialization, so we run the algorithm 25 times and select the output with the highest likelihood. 

## Results for Methodology No. 1: Discrete Emissions
### How many hidden states are there?

There is no precise answer to this question.  AIC and BIC can be used to balance maximizing the log_likelihood and the number of parameters.  We reran the model with 2, 3, 4, and 5 states:

|   n_states |   log_likelihood |   n_parameters |     AIC |     BIC |
|-----------:|-----------------:|---------------:|--------:|--------:|
|          2 |         -4226.02 |             43 | 8538.04 | 8711.57 |
|          3 |         -3996.4  |             68 | 8128.8  | 8403.22 |
|          4 |         -3831.64 |             95 | 7853.28 | 8236.65 |
|          5 |         -3718.83 |            124 | 7685.66 | 8186.06 |

This suggest that there are at least 5 states.  

Another metric to consider is sparsity.  Even if BIC improves, the model may create sparsely populated hidden states.  **Note** the state labels are not comparable between models.  In other words, State_0 could be a "recession" in the N=2 scenario and an "expansion" in the N=5 scenario.

|   State_0 |   State_1 |   State_2 |   State_3 |   State_4 |
|----------:|----------:|----------:|----------:|----------:|
|       164 |       254 |         0 |         0 |         0 |
|       114 |       178 |       126 |         0 |         0 |
|       115 |        83 |       117 |       103 |         0 |
|       110 |        91 |       106 |        74 |        37 |

The fourth state is becoming sparsely populated, though there is an argument that enough timepoints are available.

Another factor is interpretability.  We use 4 states as a starting point.

### Visualizing / Interpreting 4 Latent States:

**Learned Transition Matrix:**
|   State_0 |   State_1 |   State_2 |   State_3 |
|----------:|----------:|----------:|----------:|
|   0.9478  |   0       |   0.03455 |   0.01765 |
|   0.04572 |   0.95428 |   0       |   0       |
|   0       |   0.02736 |   0.95103 |   0.02161 |
|   0.02793 |   0       |   0.04299 |   0.92908 |
- This means that the economy rarely changes states month to month.  This aligns with expectations-- the economy doesn't bounce between expansionary and recessionary cycles often.

**Learned Emission Matrices (Housing_Starts_YoY as an example):**
| |   Low |   Middle |   High |
|----------:|----------:|----------:|----------:|
|   State_0 |   0.096 |   0.240 |   0.663 |
|   State_1 |   0.490 |   0.315 |   0.195 |
|   State_2 |   0.489 |   0.332 |   0.179 |
|   State_3 |   0.220 |   0.490 |   0.290 |
- Example Interpretation: The probability of Housing_Starts_YoY being High given State = State_0 is 0.663.
- Therefore, State_0 is associated with "high" (i.e., a larger increase) in year over year housing starts.

Pulling the emission category with the maximum probability for each given state, can provide a more complete picture of each state.  However, categories may be very close in probability for a given state.
| index                        | State_0   | State_1   | State_2   | State_3   |
|:-----------------------------|:----------|:----------|:----------|:----------|
| Yield_Curve_Spread_MonthEnd  | High      | High      | Low       | Low       |
| CPI_YoY                      | Middle    | Low       | High      | Low       |
| Industrial_Production_YoY    | Middle    | Low       | High      | Low       |
| Industrial_Production_MoM    | Middle    | Low       | Middle    | Low       |
| Housing_Starts_YoY           | High      | Low       | Low       | Middle    |
| Housing_Starts_MoM           | High      | Low       | Middle    | Low       |
| Real_PCE_YoY                 | High      | Low       | Middle    | High      |
| Unemployment_Rate_YoY_Change | Low       | High      | Middle    | Middle    |
| Unemployment_Rate_MoM_Change | Low       | High      | Low       | Low       |
| Log_VIX_MonthAvg             | Low       | High      | Low       | High      |

**Most Likely State Plotted Against Real GDP**
- The red bars are recessionary months according to NBER.  The NBER does not use a formulaic methodology for establishing a recession, but instead looks at a variety of variables.  However, recessions tend to correspond to multiple quarters of declines in seasonally adjusted real GDP.

- Below, you can see that State_1 strongly corresponds to recessions.  During the provided time period, in 1990, 2001, 2008, and 2020, when the latent state changes to State_1, the economy enters into a decline in real GDP and a recession one or two months after the state transition. Notably, the state transitioned to State_1 in November 2025, yet we are not in a recession.

- You can also see that State_0 follows State_1, indicating a recovery period after a downturn.  State_0 has best alignment with what you would expect in a strong expansionary phase. Middle/high inflation, middle/high increases in economic activity, and unemployment rate falling, low volatility in the stock market.
  
- State_2 and State_3 have more of a mix of "good" and "bad" economic indicators.

<img width="7997" height="3271" alt="r_GDP_and_states" src="https://github.com/user-attachments/assets/2607268b-ef7c-43e3-88e6-e25c15df0c41" />

- This is the same plot but with the maximum state probability plotted when it is less than 99%.  You can see that around regime changes there is more ambiguity between states.
<img width="12647" height="3271" alt="r_GDP_and_states_w _max_prob" src="https://github.com/user-attachments/assets/692a0afe-64b8-470b-962e-d16a4aceaada" />


### Visualizing / Interpreting 5 Latent States:
**Most Likely State Plotted Against Real GDP**
- The BIC results suggested that the data could support a fifth regime.  Note that it is harder to interpret this version, even if it is "better" according to BIC.  An interesting result does appear here.  State_1 is associate with all the previous recessions, and States_0, 2, 3, and 4 with expansionary periods.  However, the last several years have been State_1.  This tells a story that this economic expansion is "different" than historic expansions.
<img width="7997" height="3271" alt="r_GDP_and_states_5_states" src="https://github.com/user-attachments/assets/1adb0f17-e049-4fdf-bdef-422ab183d476" />

## Methodology No. 2: Continuous Emissions

### Continuous Emissions — Feature Engineering

#### Short- and Long-Horizon Signals

Additional signals are constructed to capture both **recent economic momentum** and **longer-term economic conditions**:

- **Growth variables** — CPI, Industrial Production, Housing Starts, and Real PCE use **1-month percent change (MoM)** for short-term momentum and **12-month percent change (YoY)** for long-term momentum.

- **Rate variables** — Unemployment Rate and Federal Funds Rate use **1-month and 12-month changes in percentage points**.

- **Level variables** — Yield Curve Spread and VIX are retained as levels because their absolute values provide meaningful signals of **economic expectations and market stress**, respectively. Monthly-average values are used to reduce daily noise, with VIX additionally log-transformed.

> **Housing Starts:** Raw 12-month YoY growth is retained because the YoY horizon already reduces short-term noise, while the full-covariance Gaussian HMM can absorb remaining variability within each state's covariance structure.

**Model-specific choices:** The continuous model retains Federal Funds Rate momentum and uses the monthly-average yield-curve spread.

> **Preprocessing note:** The final continuous features are robustly scaled using the median and IQR before Gaussian HMM estimation; details are provided in the methodology section.


### Methodology Walkthrough

#### 1. Gaussian Hidden Markov Model Setup

We use a **Gaussian Hidden Markov Model (Gaussian HMM)** with K hidden states representing unobserved macroeconomic regimes:

$$
S_t \in \{1, 2, \ldots, K\}
$$

The state $S_t$ is **latent**, meaning that we do not directly observe whether a month is in an expansion, slowdown, or contraction regime. Instead, the HMM infers the hidden state from the observed macroeconomic data.

Unlike the discrete-emission model, the continuous model does **not** convert observations into Low / Middle / High categories. It retains the actual continuous values of the macroeconomic features.

For each month $t$, the observed variables form a feature vector:

$$
X_t = (X_{1,t}, X_{2,t}, \ldots, X_{p,t})
$$

where $p$ is the number of macroeconomic features included in the model.


#### 2. HMM Components

The Gaussian HMM contains three main components:

**Initial State Distribution**

The initial state distribution describes the probability that the economy begins in each hidden state:

$$
\pi_k = P(S_0 = k)
$$

**Transition Matrix**

The transition probabilities describe how the hidden economic regime evolves over time:

$$
A_{ij} = P(S_t = j \mid S_{t-1} = i)
$$

A high probability of remaining in the same state indicates a **persistent regime**.

**Continuous Emission Distribution**

The main difference from the discrete HMM is the emission model.

For each hidden state $k$, the observed macroeconomic feature vector follows a multivariate Gaussian distribution:

$$
X_t \mid S_t = k \sim N(\mu_k, \Sigma_k)
$$

where:

- $\mu_k$ represents the typical macroeconomic conditions of state $k$.
- $\Sigma_k$ represents the covariance structure among the macroeconomic features within state $k$.

Therefore, each hidden state represents a different **joint macroeconomic environment**.


#### 3. Feature Processing

The model uses economically meaningful features designed to capture both the **current condition** and **momentum** of the economy.

- Short-term changes capture recent economic momentum.
- Long-term changes capture broader economic trends.
- Level variables capture economically meaningful conditions such as the yield curve and market stress.

Because the continuous features have different units and scales, they are standardized using **robust scaling**.

The robust-scaled observation is:

$$
Z_{i,t} = \frac{X_{i,t} - \text{median}(X_i)}{\text{IQR}(X_i)}
$$

where the median centers each feature and the IQR scales it by its typical spread.

Robust scaling puts the variables on comparable scales while being less sensitive to extreme historical observations than conventional mean/standard-deviation scaling.

A very small number of extreme observations are also lightly winsorized before scaling. This helps prevent isolated shocks from dominating the Gaussian estimation and creating artificial one-month regimes.


#### 4. Gaussian HMM Estimation

For each hidden state, the model estimates:

- Initial state probabilities: $\pi$
- Transition probabilities: $A$
- State-specific mean vectors: $\mu_k$
- State-specific covariance matrices: $\Sigma_k$

Collectively, these model parameters are represented by $\theta$.

The parameters are estimated using the **Expectation-Maximization (EM) / Baum-Welch algorithm**.

The objective is to select the parameter values that maximize the likelihood of the observed macroeconomic sequence:

$$
\theta^* = \arg\max_{\theta} P(X_{1:T} \mid \theta)
$$

Intuitively, the algorithm repeatedly performs two steps:

1. **E-step:** Estimate the probability that each month belongs to each hidden economic state.
2. **M-step:** Update the initial-state probabilities, transition probabilities, state means, and covariance matrices so that the model better explains the observed data.

These steps repeat until the likelihood converges.

Because the final solution can depend on initialization, the model is fitted multiple times using different random starting values, and the solution with the highest likelihood is retained.


#### 5. How the Model Infers Economic Regimes

For each month, the model essentially asks:

> Given this month's inflation, economic activity, labor-market conditions, monetary policy, yield curve, and market stress, which hidden state's economic profile is most likely to have generated these observations?

The HMM combines two pieces of information:

1. **Emission information:** How closely the observed macroeconomic conditions match each state's Gaussian distribution.
2. **Transition information:** How likely the economy is to move from the previous month's state into each possible current state.

Conceptually:

**Observed macroeconomic conditions → State likelihoods → Transition probabilities → Inferred hidden regime**

This allows the HMM to identify **persistent economic regimes through time**, rather than simply clustering each month independently.


#### 6. Interpretation of Hidden States

The economic meaning of each hidden state is **not specified before fitting the model**.

The HMM first identifies numerical states such as State 0, State 1, State 2, and so on. Each state is then interpreted using:

- Its macroeconomic feature profile
- Its transition probabilities and persistence
- The historical periods assigned to the state
- External economic references such as NBER recession periods

For example, a state characterized by weak economic activity, deteriorating labor conditions, and elevated VIX may be interpreted as a **stress / contraction regime**.

Importantly, NBER recession dates are **not used to train the HMM**. They are used afterward to help interpret and externally validate the inferred regimes.


#### 7. Core Difference from the Discrete-Emission HMM

Both the discrete and continuous approaches use the same underlying HMM structure:

$$
S_{t-1} \rightarrow S_t \rightarrow X_t
$$

Both models use observed macroeconomic information to infer hidden regimes and estimate how those regimes persist and transition through time.

The key difference is the **emission assumption**.

The discrete model converts each observation into a category such as Low / Middle / High and estimates:

$$
P(X_{i,t} = c \mid S_t = k)
$$

The continuous model retains the actual numerical values and assumes:

$$
X_t \mid S_t = k \sim N(\mu_k, \Sigma_k)
$$

Therefore:

- **Discrete emissions** simplify the observations into categories, making the model easier to interpret and naturally more robust to extreme values.
- **Continuous emissions** preserve the magnitude and joint relationships among the macroeconomic variables, allowing a richer description of each regime.

For example, VIX values of 35 and 80 might both be classified as **High** in the discrete model. The continuous model preserves the difference in magnitude between these two observations.

> **Note:** The current Gaussian HMM uses maximum-likelihood EM/Baum-Welch estimation. It does not perform full Bayesian inference over the HMM parameters.

## Results for Methodology No. 2: Continuous Emissions
What did you find?

## Comparison of Approaches
[[__]]  [[Cathy to fill]]

## Repository Structure
What are the important files?  [[Cathy to fill]]

## Installation / Requirements
What packages are required?  [[Cathy to fill]]

## Running the Analysis
What should someone run and in what order?  [[Cathy to fill]]
