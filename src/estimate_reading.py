import math

import numpy as np

def reading_function(array_allocations, estimators, type_estimator, estimator_sd=None, sign=1):
    """Compute estimated reading for any given estimator.

    Args:
        array_allocations (np.array): the spendings for fiction and self-help.
        estimators: the estimator to use
        type_estimator (str): keyword to describe the methods to run a prediction with the estimator
        estimator_sd (optional): the estimator for standard deviation, to use for Sharpe ratio. 
            If provided, the function returns the Sharpe ratio. Defaults to None.
        sign (int, optional): Whether to return -1 * reading (useful for optimization). Defaults to 1.

    Returns:
        Estimated reading
    """
    assert type_estimator in ['smt_kriging', 'smt_kriging_reading', 'sklearn']
    # Predict reading
    if type_estimator == "sklearn":
        reading = estimators.predict(array_allocations)

    if type_estimator == 'smt_kriging_reading': 
        reading = estimators.predict_values(array_allocations)
    
    if type_estimator == "smt_kriging":
        # Run estimator
        reading_fiction = estimators[0].predict_values(array_allocations)
        reading_help = estimators[1].predict_values(array_allocations)
        reading = reading_fiction + reading_help 

    # Divide by sd if asked
    if estimator_sd is not None:
        if type_estimator == "smt_kriging":
            var_fiction = estimator_sd[0].predict_variances(array_allocations)
            var_help = estimator_sd[1].predict_variances(array_allocations)
            var_reading = var_fiction + var_help # for simplicity, we ignore the covariance
            sd_reading = np.sqrt(var_reading)

        elif type_estimator == "sklearn":
            sd_reading = np.array(
                [
                    estimator_sd(allocation[0], allocation[1])[0] 
                    for allocation 
                    in zip(array_allocations[:, 0], array_allocations[:, 1])
            ])
        else:
            sd_reading = np.array(
                [
                    estimator_sd(allocation[0], allocation[1]) 
                    for allocation 
                    in zip(array_allocations[:, 0], array_allocations[:, 1])
            ])
        reading = reading / sd_reading
    
    if reading.shape == (1, 1):
        reading = reading[0, 0]

    if sign == -1:
        reading = -1 * reading
    
    return reading

def allocate_by_proportions(p, total_budget):
    """Generate free time allocations for fiction and self-help. 

    Args:
        p (float): the proportion of the time that goes for fiction
        total_budget (int): total free time

    Returns:
        np.darray: free time allocation for fiction and self-help
    """
    if isinstance(p, np.ndarray):
        p = p[0]
        
    time_fiction = p * total_budget
    time_help = (1 - p) * total_budget
    allocation = np.array([[time_fiction, time_help]])
    return allocation

def compute_reading_for_proportion(p, total_budget, estimators, type_estimator, estimator_sd=None, sign=1):
    """Compute the reading associated with spendings of a certain proportion of the time for fiction

    Args:
        p (float): the proportion of the free time that goes for fiction
        total_budget (int): total free time
        estimators: the estimator to use
        type_estimator (str): keyword to describe the methods to run a prediction with the estimator
        estimator_sd (optional): the estimator for standard deviation, to use for Sharpe ratio. 
            If provided, the function returns the Sharpe ratio. Defaults to None.
        sign (int, optional): Whether to return -1 * reading (useful for optimization). Defaults to 1.

    Returns:
        Estimated reading
    """
    allocation = allocate_by_proportions(p, total_budget)
    reading = reading_function(
        allocation, 
        estimators, 
        type_estimator=type_estimator, 
        estimator_sd=estimator_sd,
        sign=sign    
    )
    return reading
