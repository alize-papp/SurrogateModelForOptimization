import numpy as np

class GaussianProcess2D():
    def __init__(self, mean_function, cov_function):
        self.mean_function = mean_function
        self.cov_function = cov_function
        
    def parametrize_for_a_query(self, x1, x2):
        mean = self.mean_function(x1, x2) 
        cov = self.cov_function(x1, x2)
        return mean, cov
    
    def generate_sample(self, x1, x2, n_samples=1000):
        mean, cov = self.parametrize_for_a_query(x1, x2)
        sample_y1, sample_y2 = np.random.default_rng(seed=42).multivariate_normal(
            mean, 
            cov, 
            n_samples
        ).T
        return sample_y1, sample_y2
    
    def compute_expected_sum(self, x1, x2, add_sharpe=False):
        mean, cov = self.parametrize_for_a_query(x1, x2)
        expected_sum = mean[0] + mean[1]
        if add_sharpe:
            sd_y1 = cov[0][0]
            sd_y2 = cov[1][1]
            sd_sum = np.sqrt(sd_y1 + sd_y2 + 2 * cov[0][1])
            sharpe_ratio = (expected_sum) / sd_sum
            return expected_sum, sharpe_ratio
        
        return expected_sum