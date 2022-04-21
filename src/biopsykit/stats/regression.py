"""Functions for performing regression analysis."""
from typing import Union

import pandas as pd
import pingouin as pg

__all__ = ["stepwise_backwards_linear_regression"]


def stepwise_backwards_linear_regression(predictors: pd.DataFrame, dv: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Perform a stepwise backwards linear regression.

    The stepwise backwards linear regression is performed iteratively by running a linear regression on the predictors
    and removing the predictor with the highest p-value. The best-fitting model is obtained by choosing the regression
    model with the highest adjusted R-squared value.

    Parameters
    ----------
    predictors : :class:`pandas.DataFrame`
        Dataframe with predictors for the regression.
    dv : :class:`pandas.DataFrame` or :class:`pandas.Series`
        Dependent variable for the regression.

    Returns
    -------
    :class:`pandas.DataFrame`
        Best-fitted regression model.

    """
    # dv must only be a single column
    if dv.ndim > 1:
        if dv.shape[-1] != 1:
            raise ValueError("dv must be a single column")
        dv = dv.iloc[:, 0]

    list_adj_r2 = []
    list_reg_models = []
    while len(predictors.columns) > 0:
        reg_results = _lin_reg(predictors, dv)
        # drop the most correlated predictor
        drop_col = reg_results["pval"].idxmax()
        predictors = predictors.drop(columns=drop_col)
        adj_r2 = reg_results["adj_r2"].iloc[0]
        list_adj_r2.append(adj_r2)
        list_reg_models.append(reg_results)

    # get the best regression model, i.e. the model with the highest adjusted R2
    return list_reg_models[list_adj_r2.index(max(list_adj_r2))]


def _lin_reg(predictors: pd.DataFrame, dv: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    return pg.linear_regression(predictors, dv).set_index("names").drop(index="Intercept")
