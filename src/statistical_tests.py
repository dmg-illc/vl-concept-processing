from scipy.stats import norm
import numpy as np

def corr_significance_test(sample_size, corr1, corr2):
    
    # Fisher transformation
    fisher1 = 0.5 * np.log((1 + corr1) / (1 - corr1))
    fisher2 = 0.5 * np.log((1 + corr2) / (1 - corr2))
    

    expected_sd = np.sqrt(1.060 / (sample_size - 3))
    
    # Compute p-value
    z_score = abs(fisher1 - fisher2) / expected_sd
    p_value = 2 * (1 - norm.cdf(z_score))
    
    return p_value
