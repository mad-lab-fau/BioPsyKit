import pandas as pd


def bmi(data: pd.DataFrame) -> pd.DataFrame:
    """ Computes the **Body Mass Index**.

    This function assumes the data passed in the following way:

    * 1st column: weight
    * 2nd column: height

    Parameters
    ----------
    data : pd.DataFrame
        pandas dataframe containing weight and height information

    Returns
    -------
    pd.DataFrame
        dataframe with body mass index

    """
    return pd.DataFrame(data.iloc[:, 0] / (data.iloc[:, 1] / 100.0) ** 2, columns=["bmi"])


def whr(score: str, df: pd.DataFrame) -> pd.DataFrame:
    """"Waist to Hip Ratio"""
    return pd.DataFrame(df.iloc[:, 0] / df.iloc[:, 1], columns=[score])
