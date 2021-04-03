import os
import sys

import numpy as np
import pandas as pd

def prepare_data_for_one_strategy(df_sales, strategy_id, dict_budget):
    """Clean the data for one strategy

    Args:
        df_sales (pd.DataFrame): the data with all strategies
        strategy_id (int): the id of the strategy
        dict_budget (dict): the budget for each strategy

    Returns:
        pd.DataFrame: DataFrame about one strategy, for analysis
    """
    # Keep id and sales
    col_fiction = 'Fiction.' + str(strategy_id)
    col_help = 'Self-Help.' + str(strategy_id)
    df_strategy = df_sales[[col_fiction, col_help]]
    df_strategy = df_strategy.rename(
        columns={
            'Fiction.' + str(strategy_id): 'reading_fiction', 
            'Self-Help.' + str(strategy_id): 'reading_help'
        }
    )
    # Add marketing budgets
    df_strategy['free_fiction'] = dict_budget[col_fiction]
    df_strategy['free_help'] = dict_budget[col_help]
    # Compute profit
    df_strategy['total_reading'] = df_strategy['reading_fiction'] + df_strategy['reading_help']
    return df_strategy

def prepare_data(df_raw, format_to_long=True, return_dict_budget=False):
    """Prepare data from its raw format to one that can be used for modeling or analysis

    Args:
        df_raw (pd.DataFrame): the raw data
        format_to_long (bool, optional): whether to format to long for modeling
        or wide for analysis. Defaults to True.
        return_dict_budget (bool, optional): whether to also return a dict with the budget
        for each strategy. Defaults to False.

    Returns:
        pd.DataFrame: cleaned data
    """
    df_wide = df_raw.rename(
        columns={
            'Fiction': 'Fiction.0', 
            'Self-Help': 'Self-Help.0',
        }
    )
    columns_wide = df_wide.columns
    columns_to_drop = [
        column
        for column in columns_wide
        if 'Unnamed' in column
    ]
    df_wide = df_wide.drop(
        columns_to_drop,
        axis=1
    )
    dict_budget = df_wide.loc[0]
    df_wide = df_wide.drop([0, 1], axis=0)
    if not format_to_long:
        if return_dict_budget:
            return df_wide, dict_budget
        return df_wide
    
    else:
        df_long = pd.concat(
            [prepare_data_for_one_strategy(df_wide, i, dict_budget) for i in range(12)], 
            ignore_index=True
        )
        if return_dict_budget:
            return df_long, dict_budget
        return df_long