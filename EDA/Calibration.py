from sklearn.isotonic import IsotonicRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calibrate_with_isotonic(df, model_output_col, default_label_col, k=20):
    # Sort output probabilities
    df = df.sort_values(by=model_output_col, ascending=False).reset_index(drop=True)
    
    # Number of firms in each bucket
    N = len(df)
    bucket_size = N // k
    
    default_rates = []
    quantiles = []
    
    # Calculate the observed default rate for each bucket
    for i in range(k):
        bucket = df.iloc[i * bucket_size: (i + 1) * bucket_size]
        
        # Calculate observed default rate
        default_rate = bucket[default_label_col].mean()
        default_rates.append(default_rate)
        
        # Take the minimum output in this bucket as the quantile
        quantiles.append(bucket[model_output_col].min())
    
    # Fit isotonic regression to map quantiles to default rates
    # Other fit curve (quadratic) can be considered
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(quantiles, default_rates)
    
    # Map all probabilities
    calibrated_probs = iso_reg.transform(df[model_output_col].values)
    
    # Plot the isotonic calibration curve
    plt.plot(quantiles, default_rates, 'o', label="Observed default rates")
    plt.plot(np.sort(df[model_output_col].values), np.sort(calibrated_probs), '-', label="Isotonic calibration curve")
    plt.xlabel("Quantiles of Model Output")
    plt.ylabel("Observed Default Rate")
    plt.legend()
    plt.title("Isotonic Regression Calibration Curve")
    plt.show()
    
    return calibrated_probs, iso_reg
