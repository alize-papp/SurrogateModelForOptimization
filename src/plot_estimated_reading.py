import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize_scalar

from estimate_reading import reading_function

def compute_reading_for_all_combinations(max_time, estimators, type_estimator, sampling_step=10, estimator_sd=None):
    """Compute a 2d matrix with reading times for all allocations between 0 and max_time.

    Args:
        max_time (int): maximal time for each product.
        estimators: the estimators to use for predicting reading time. Either one estimator or a tupple.
        type_estimator (str): name of the estimator, to find predict method.
        sampling_step (int, optional): the sampling step in the square. Defaults to 10.
        estimator_sd (optional): the estimator for standard deviation, to use for Sharpe ratio. 
            If provided, the function returns the Sharpe ratio. Defaults to None.

    Returns:
        (np.array, np.array, np.array): budgets for Fiction, budgets for Self-Help, associated reading.
    """
    # Generate allocations of free reading time for Fiction and Self-help below max_time
    x = np.arange(0, max_time + sampling_step, step=sampling_step)
    y = np.arange(0, max_time + sampling_step, step=sampling_step)
    # Format for the estimator generating all pairs of x, y
    x_extended = np.tile(x, x.shape[0])
    y_extended = np.repeat(y, x.shape[0])
    xy_long = np.vstack((x_extended, y_extended)).T
    reading = reading_function(
        array_allocations=xy_long, 
        estimators=estimators, 
        type_estimator=type_estimator, 
        estimator_sd=estimator_sd
    )  
    reading_square = reading.reshape(x.shape[0], x.shape[0])
    return x, y, reading_square

def plot_reading_for_2d_allocation(max_time, estimators, type_estimator, sampling_step=10, estimator_sd=None):
    """Plot contour of reading function for all combinations of free-time budget between 0 and max_time.

    Args:
        max_time (int): maximal budget for each product.
        estimators: the estimators to use for predicting reading. Either one estimator or a tupple.
        type_estimator (str): name of the estimator, to find predict method.
        sampling_step (int, optional): the sampling step in the square. Defaults to 10.
        estimator_sd (optional): the estimator for standard deviation, to use for Sharpe ratio. 
            If provided, the plot will represent the Sharpe ratio. Defaults to None.

    """
    # Compute reading
    x, y, z = compute_reading_for_all_combinations(
        max_time=max_time, 
        estimators=estimators, 
        type_estimator=type_estimator, 
        sampling_step=sampling_step,
        estimator_sd=estimator_sd
    )
    # Plot contour
    fig = go.Figure(
        data=go.Contour(
            x = x,
            y = y,
            z = z,
            colorscale = 'Viridis'
        )
    )
    # Darken out-of-budget area
    fig.add_trace(go.Scatter(
        x=[1, max_time, max_time, 1], 
        y=[max_time, max_time, 1, max_time],
        fill='toself', 
        fillcolor='rgba(192, 192, 192, 0.5)',
        line_color='black',
        hoveron = 'points+fills',
        text="Not available",
        hoverinfo = 'text'
    ))    
    # Format graph
    title = "Expected reading time"
    if estimator_sd is not None:
        title = "Sharpe ratio of reading time"

    fig.update_layout(
        autosize=False,
        width=700,
        height=700,
        xaxis={'title': 'Free time for Fiction (min)'},
        yaxis={'title': 'Free time for Self-Help (min)'},
        title=title,
        title_x=0.5,
    )
    fig.show()

def track_optimization_improvement(fun_to_optimize):
    """Compute the errors over iterations of the optimization algorithm.

    Args:
        fun_to_optimize (function): the function to optimize

    Returns:
        np.darray: errors over all iterations
    """
    #all_x = list()
    all_y = list()

    for iter in range(30):
        result = minimize_scalar(fun_to_optimize, bounds=(0, 1), method='bounded', options={"maxiter": iter})
        if result.success:
            break

        #all_x.append(result.x)
        all_y.append(fun_to_optimize(result.x))

    all_errors = np.abs(all_y - all_y[-1])
    return all_errors

def plot_optimization_improvement(fun_to_optimize):
    """Plot error over iterations of the optimization algorithm

    Args:
        fun_to_optimize (function): the function to optimize
    """
    all_errors = track_optimization_improvement(fun_to_optimize)
    fig = go.Figure(data=go.Scatter(x=list(range(1, 1 + len(all_errors))), y=all_errors))
    fig.update_layout(
        title="Reduction of the error accross iterations of the optimization process",
        title_x=0.5,
        xaxis={'title': 'Iterations'},
        yaxis={'title': 'Error on max f(x)'},
    )
    fig.show()