
# Smart Load Forecasting using Hybrid Deep Learning with Fuzzy Smoothing

This project implements a **deep learning-based time series forecasting model** to predict future energy load in a smart grid system. The architecture combines **Conv1D**, **Bidirectional LSTM**, and **Multi-Head Attention**, enhanced with **fuzzy logic smoothing** for better error control. It includes advanced feature engineering techniques such as lag features, rolling statistics, time encoding, and fuzzy membership functions.

---

## File Structure

| File             | Description                                                                                     |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| `main.ipynb`     | Core notebook containing preprocessing, model building, training, evaluation, and visualization |
| `power_data.csv` | Time series data of power consumption and related grid metrics *(uploaded during execution)*    |

---

## Installation

Install required packages (if using a local environment):

```
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```
---

## Dataset Overview

The dataset includes real-time smart grid features such as:

* Power Consumption (kW)
* Solar & Wind Power (kW)
* Temperature (°C), Humidity (%)
* Electricity Price (USD/kWh)
* Grid Supply, Voltage Fluctuation
* Overload & Transformer Fault flags
* Time components (hour, day, month)

---

## Model Architecture

| Layer                  | Description                                    |
| ---------------------- | ---------------------------------------------- |
| Conv1D                 | Extracts temporal patterns                     |
| BiLSTM x2              | Captures bidirectional sequential dependencies |
| Multi-Head Attention   | Learns long-term dependencies                  |
| GlobalAveragePooling1D | Reduces dimensionality                         |
| Dense Layers           | Learns final regression output                 |
| Output                 | Single continuous value (load forecast)        |

---

## Feature Engineering

* **Lag Features**: Past values of load and power
* **Rolling Statistics**: Mean and std over 6 and 12-hour windows
* **Time Encoding**: Hour/day/month sin/cos
* **Fuzzy Logic Features**:

  * Temperature: Low / Medium / High (trapezoidal)
  * Price: Low / High

---

## Results

| Metric | Raw Prediction | Smoothed Prediction (Fuzzy) |
| ------ | -------------- | --------------------------- |
| MAE    | 0.660          |  0.542                      |
| RMSE   | 0.767          |  0.652                      |
| MAPE   | 11.23%         |  9.42%                      |

 Fuzzy smoothing significantly reduces error and improves prediction stability.

---

## Visuals Included

*  Actual vs Raw and Smoothed Predictions
*  Residual Distribution Comparison (KDE)
*  Predicted vs Actual Scatter Plot
*  Absolute Prediction Error Over Time

---

## Highlights

* Uses hybrid deep learning (Conv1D + BiLSTM + Attention)
* Incorporates **fuzzy logic** smoothing for stability
* Feature-rich model with **temporal**, **statistical**, and **fuzzy** inputs
* Performs well with **low RMSE and MAPE**


