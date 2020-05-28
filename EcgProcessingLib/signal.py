import pandas as pd
import numpy as np


def interpolate_sec(df: pd.DataFrame) -> pd.DataFrame:
    from scipy import interpolate
    x_old = np.array((df.index - df.index[0]).total_seconds())
    x_new = np.arange(1, np.ceil(x_old[-1]) + 1)
    interpol_f = interpolate.interp1d(x=x_old, y=df['ECG_Rate'], fill_value="extrapolate")
    return pd.DataFrame(interpol_f(x_new), index=x_new, columns=df.columns)
