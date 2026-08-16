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
    - Raw Housing Starts -> year-over-year, quarter-over-quarter, and month-over-month percent change.
    - Unemployment Rate -> year-over-year, quarter-over-quarter, and month-over-month change (since already a percentage).
2. Create Real PCE level by deflating Nominal PCE using the CPI. Then, create year-over-year percent change in Real PCE.
3. Log transform of VIX. VIX is strongly right-skewed.
4. **Discretize** the variables into 3 quantiles for the discrete emission analysis.

### Final Analytical Dataset
Note: We include both the annual change and monthly change series for Industrial Production, Housing Starts, and Unemployment Rate to account for short term and long term momentum.  For example, coming out of a recession, year-over-year change in Housing Starts might be flat or negative, but month-over-month change might be positive.
|   Variable |   Frequency |   Description |
|-------------------:|-----------------:|------:|
|                  Yield_Curve_Spread_MonthEnd |          Monthly | Month End Value |
|                  CPI_YOY |          Monthly | Percent Change YoY |
|                  Industrial_Production_YoY |          Monthly | Percent Change YoY |
|                  Industrial_Production_MoM |          Monthly | Percent Change MoM |
|                  Housing_Starts_YoY |          Monthly | Percent Change YoY |
|                  Housing_Starts_MoM |          Monthly | Percent Change MoM |
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

#### 1. Feature and Momentum Summary

The continuous model uses a combination of **short-term momentum, long-term trends, and economically meaningful level variables** to represent different aspects of macroeconomic conditions:

- **Growth variables** — CPI, Industrial Production, Housing Starts, and Real PCE use **1-month percent change (MoM)** for short-term momentum and **12-month percent change (YoY)** for long-term momentum.

- **Rate variables** — Unemployment Rate and Federal Funds Rate use **1-month and 12-month changes in percentage points**.

- **Level variables** — Yield Curve Spread and VIX are retained as levels because their absolute values provide meaningful signals of **economic expectations and market stress**, respectively. Monthly-average values are used to reduce daily noise, with VIX additionally log-transformed.

> **Housing Starts:** Raw 12-month YoY growth is retained because the YoY horizon already reduces short-term noise, while the full-covariance Gaussian HMM can absorb remaining variability within each state's covariance structure.

**Model-specific choice:** The continuous specification retains **Federal Funds Rate momentum**, using the 12-month change to capture broader monetary-policy cycles rather than infrequent month-to-month moves. 

#### 2. Feature Diagnostics and Selection Decisions

Candidate features are evaluated using **distribution, correlation, and outlier diagnostics** to support the final selection.

| Diagnostic | Key Result | Decision |
|---|---|---|
| **Skewness / ST Noise** | Unemployment ST has the highest skew (**15.13**); Industrial Production ST is also highly skewed (**−5.29**). ST measures are generally noisier than LT measures. | Favor **LT measures** for Industrial Production, Real PCE, and Housing Starts. Keep **Unemployment ST** as an important turning-point signal. |
| **Real-Activity Redundancy** | Real PCE and Industrial Production momentum are correlated at both horizons (**ST r = 0.67; LT r = 0.67**). | **Drop Real PCE momentum** and retain **Industrial Production LT** to reduce overlapping real-activity signals. |
| **Activity vs. Labor** | Industrial Production / Real PCE and Unemployment are negatively correlated (**r ≈ −0.69 to −0.74**). | **Keep Unemployment.** Although correlated with growth measures, it captures a different economic dimension: **labor-market conditions rather than economic activity**. |
| **Extreme Outlier** | Unemployment ST reaches **52.0 IQR units** in April 2020 due to the COVID unemployment shock. | **Keep Unemployment ST**, but winsorize the extreme tail before robust scaling to prevent one shock from dominating the Gaussian HMM. |

**Selection takeaway:** The results generally favor **long-term measures over short-term measures**. PCE momentum is removed due to overlap with Industrial Production, while short-term unemployment is retained as an important turning-point signal.


#### 3. Final Continuous-HMM Feature Set

Based on the diagnostics above, **8 economically distinct features** are selected, balancing short- and long-term information while limiting unnecessary redundancy.

| Feature | Economic Signal | Selection Rationale |
|---|---|---|
| `CPI_ST_Momentum` | Inflation — recent | Captures short-term inflation changes and turning points; only moderately correlated with CPI LT (**r = 0.41**). |
| `CPI_LT_Momentum` | Inflation — trend | Captures the longer-term YoY inflation trend. |
| `Industrial_Production_LT_Momentum` | Real activity | Captures the longer-term business-cycle growth trend. |
| `Housing_Starts_LT_Momentum` | Housing activity | Captures rate-sensitive housing-cycle conditions distinct from Industrial Production. |
| `Unemployment_Rate_ST_Momentum` | Labor market | Captures short-term labor-market changes and economic turning points. |
| `Federal_Funds_Rate_LT_Momentum` | Monetary policy | Captures the broader hiking/cutting cycle; LT is retained over ST for parsimony (**ST/LT r = 0.59**). |
| `Yield_Curve_Spread_MonthAvg` | Yield curve | Provides a forward-looking signal of economic conditions. |
| `Log_VIX_MonthAvg` | Market stress | Captures financial-market stress while the log transformation reduces VIX skew. |

**Final analytical input:** 8 features covering **inflation, real activity, housing, labor conditions, monetary policy, the yield curve, and market stress**.

> **Modeling note:** Feature selection reduces unnecessary redundancy; it does not require the inputs to be independent. The full-covariance Gaussian HMM directly learns how the selected macroeconomic variables move together within each hidden regime.

### Continuous Emissions — Methodology Walkthrough

#### 1. Gaussian HMM Framework

We use a **Gaussian Hidden Markov Model (Gaussian HMM)** to infer unobserved macroeconomic regimes over time:

$$
S_t \in \{1, 2, \ldots, K\}
$$

The hidden state $S_t$ represents the underlying economic regime in month $t$. The regime is not directly observed; instead, it is inferred from the macroeconomic feature vector:

$$
X_t = (X_{1,t}, X_{2,t}, \ldots, X_{p,t})
$$

Unlike the discrete HMM, the continuous model retains the **actual numerical values** of the macroeconomic features.

The model has three core components:

- **Initial state probabilities:** $\pi_k = P(S_0 = k)$ describe the probability of starting in each regime.
- **Transition probabilities:** $A_{ij} = P(S_t = j \mid S_{t-1} = i)$ describe how likely the economy is to remain in or move between regimes.
- **Gaussian emissions:** describe the macroeconomic conditions associated with each hidden regime.

For each state $k$:

$$
X_t \mid S_t = k \sim N(\mu_k, \Sigma_k)
$$

where $\mu_k$ represents the state's typical macroeconomic conditions and $\Sigma_k$ captures the variability and relationships among the features within that regime.

Together, the mean and covariance define the **economic profile of each hidden state**.


#### 2. Continuous Feature Processing

Before fitting the Gaussian HMM, the selected features are lightly winsorized and **robustly scaled**:

$$
Z_{i,t} = \frac{X_{i,t} - \text{median}(X_i)}{\text{IQR}(X_i)}
$$

Robust scaling puts variables with different units on comparable scales while reducing sensitivity to extreme historical observations.

Light winsorization is applied only to extreme tails so that isolated shocks do not dominate the Gaussian estimation or create artificial one-month regimes.


#### 3. Model Estimation

The Gaussian HMM learns:

- the initial state probabilities $\pi$
- the transition matrix $A$
- a mean vector $\mu_k$ for each state
- a covariance matrix $\Sigma_k$ for each state

These parameters are estimated using the **Expectation-Maximization (EM) / Baum-Welch algorithm** to maximize the likelihood of the observed macroeconomic sequence:

$$
\theta^* = \arg\max_{\theta} P(X_{1:T} \mid \theta)
$$

The algorithm iterates between two steps:

1. **E-step:** Estimate the probability that each month belongs to each hidden state.
2. **M-step:** Update the transition probabilities and Gaussian emission parameters to better explain the observed data.

The process repeats until the likelihood converges. Because the solution can depend on initialization, the model is fitted from multiple random starting points and the highest-likelihood solution is retained.


#### 4. Inferring Economic Regimes

For each month, the HMM combines two sources of information:

1. **Emission likelihood:** How well the observed macroeconomic conditions match each state's Gaussian profile.
2. **Transition probability:** How likely each state is given the previous economic regime.

Conceptually:

**Previous regime + Current macroeconomic conditions → State probabilities → Inferred hidden regime**

This allows the model to identify **persistent macroeconomic regimes over time**, rather than treating each month as an independent observation.

> **Methodology note:** The Gaussian HMM uses EM/Baum-Welch maximum-likelihood estimation. The hidden states are inferred probabilistically, but the current specification does not perform full Bayesian inference over the model parameters.

## Results for Methodology No. 2: Continuous Emissions

### 1. HMM Model Selection

The number of hidden states ($k$) is evaluated based on **model fit, stability, state occupancy, and complexity**.

| **k** | **Params** | **Log-L** | **BIC** | **Run Stability** | **Smallest State** |
|------:|-----------:|----------:|--------:|------------------:|-------------------:|
| 2 | 91  | -3544.6 | 7639.0 | 98.2% | 31.0% |
| 3 | 140 | -3264.9 | 7375.4 | **99.0%** | 18.1% |
| 4 | 191 | -2994.4 | 7142.5 | 75.4% | 4.1% |
| 5 | 244 | -2752.0 | **6977.9** | 68.9% | 1.4% |

**Key findings:**

- **Fit favors more states:** BIC improves through $k=5$.
- **Stability favors fewer states:** Stability measures how consistently repeated model fits identify the same hidden regimes. Run stability declines from **99.0% at $k=3$** to **68.9% at $k=5$.
- **Complexity increases:** Higher-state models introduce more parameters and smaller, more specialized regimes.

**Takeaway:** The diagnostics show a tradeoff between **fit and stability/parsimony**. Since no single metric provides a clear overall winner, the final state count also considers whether the additional states provide meaningful economic structure.

### 2. Final State Selection

The final choice focuses on **$k=4$ vs. $k=5$**:

- **$k=4$** is simpler and more stable (**75.4%**).
- **$k=5$** provides better statistical fit (**BIC = 6977.9**) but lower stability (**68.9%**).
- The additional complexity of $k=5$ is evaluated next by examining whether the extra state produces **distinct and economically meaningful regimes**.

**Selection approach:** The **5-state HMM** is carried forward as the primary candidate based on its stronger fit, while the **4-state model** serves as the more stable and parsimonious benchmark. Final support for $k=5$ depends on the state interpretation that follows.

### 3. Economic Interpretation of Hidden States

State-specific distributions show the typical macroeconomic profile of each regime. Historical assignments provide an additional check on whether these profiles correspond to recognizable economic periods.

| State | % of Months | Key Economic Profile | Representative Periods | Preliminary Interpretation |
|---|---:|---|---|---|
| **State 0** | 34.5% | Strong IP growth (**+3.93%**), rising Fed Funds (**+1.01 pp**), relatively flat yield curve | 1994–2001; 2004–06; 2021–23 | **Hiking-Cycle Expansion** |
| **State 1** | 1.4% | Weak trailing IP (**−8.46%**) but rapidly falling unemployment (**−1.18 pp**), housing rebound (**+5.93%**), deep Fed cuts (**−2.11 pp**) | May–Oct 2020 | **Post-Crisis Snapback** |
| **State 2** | 27.9% | Strongest housing growth (**+12.12%**), steep yield curve (**1.72**), lowest VIX | 2003–04; 2011–18 | **Steady / Housing-Led Expansion** |
| **State 3** | 26.9% | Flat IP (**−0.09%**), weakening housing (**−3.28%**), moderate Fed easing (**−1.26 pp**) | 2006–08; 2019–early 2020; 2023–25 | **Slowdown / Policy Easing** |
| **State 4** | 9.3% | Weak IP (**−1.77%**), sharp housing weakness (**−9.96%**), highest VIX, Fed easing (**−1.20 pp**) | 2008–09; Mar–Apr 2020 | **Elevated Stress / Contraction** |

**Key findings:**

- **States 0 and 2** represent two distinct expansion environments: a **policy-tightening expansion** versus a **calmer, housing-led expansion**.
- **State 3** captures broader slowing/easing conditions, while **State 4** represents shorter periods of substantially greater economic and market stress.
- **State 1** is highly specialized but economically interpretable: all six months occur during the **May–October 2020 post-crisis rebound**.

**State-selection implication:** The 5-state specification produces economically distinct and historically recognizable regimes rather than simply splitting the data into additional statistical clusters. In particular, it separates different expansion environments and distinguishes **acute stress from the unusual post-crisis rebound**.


## Comparison of Approaches
Both models were able to identify persistent and economically interpretable macroeconomic regimes. Rather than comparing the numerical state labels directly, we compared the timing of regime transitions and how the inferred states aligned with real GDP and known recession periods.
The discrete model identified a recession-like state that aligned closely with the 1990–91, 2001, 2008–09, and 2020 recessions. It also showed higher uncertainty around several regime transitions, suggesting that economic turning points are not always clearly classified.
The continuous model identified [______]. Compared with the discrete model, it [_______identified similar/different turning points, produced more/fewer states, showed greater/less sensitivity around specific periods, something - TBD results].
Overall, the two approaches [_______showed similar / different broad regime patterns / in ___ periods]. The discrete approach produced simpler and more interpretable state profiles, while the continuous approach preserved more information about the magnitude and joint behavior of the economic variables. The results suggest that the two approaches provide complementary views of macroeconomic regimes rather than one clearly outperforming the other. Depending on the use case and degree of inter-pretability required, the continuous model could be more practical for deeper or continued analysis, and the discrete model’s results could be more easily communicated to a broader audience.


## Repository Structure
What are the important files?  
1. Create_Dataset_v2.ipynb - cleans data and creates dataset for analysis
2. Continuous_HMM_Regime_Model_v3.ipynb - runs continuous-emission HMM
3. Discrete Emissions.ipynb - runs discrete-emission HMM
4. master_monthly_dataset.csv - data source
5. gdp_quarterly_growth.csv - data source
6. USREC.csv - data source


## Installation / Requirements
What packages are required? 
1. pandas
2. numbpy
3. matplotlib
4. scikit-learn
5. scipy
6. hmmlearn
7. requests

## Running the Analysis
What should someone run and in what order? 
1. Run Create_Dataset_v2.ipynb to create the cleaned datasets
2. Run Discrete Emissions.ipynb to run the discrete-emission HMM
3. Run Continuous_HMM_Regime_Model.ipynb to run the continuous-emission HMM
4. Compare the inferred regimes with recession periods, each other, GDP, and other indicators as needed
Footer
