from typing import Dict, Any, Optional

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.pipeline import Pipeline
from tqdm.notebook import tqdm

from biopsykit.classification.utils import split_train_test


__all__ = ["nested_cv_grid_search"]


def nested_cv_grid_search(
    x_data: np.ndarray,
    y_data: np.ndarray,
    params: Dict[str, Any],
    pipeline: Pipeline,
    outer_cv: BaseCrossValidator,
    inner_cv: BaseCrossValidator,
    groups: Optional[np.ndarray] = None,
    **kwargs,
):
    results_dict = {key: [] for key in ["grid_search", "cv_results", "test_score", "best_estimator", "conf_matrix"]}

    for train, test in tqdm(list(outer_cv.split(x_data, y_data, groups)), desc="Outer CV"):
        if groups is None:
            x_train, x_test, y_train, y_test = split_train_test(x_data, y_data, train, test)
            grid = GridSearchCV(pipeline, param_grid=params, cv=inner_cv, **kwargs)
            grid.fit(x_train, y_train)
        else:
            x_train, x_test, y_train, y_test, groups_train, groups_test = split_train_test(
                x_data, y_data, train, test, groups
            )
            grid = GridSearchCV(pipeline, param_grid=params, cv=inner_cv, **kwargs)
            grid.fit(x_train, y_train, groups_train)

        results_dict["grid_search"].append(grid)
        results_dict["test_score"].append(grid.score(x_test, y_test))
        results_dict["cv_results"].append(grid.cv_results_)
        results_dict["best_estimator"].append(grid.best_estimator_)
        results_dict["conf_matrix"].append(confusion_matrix(y_test, grid.predict(x_test), normalize=None))

    return results_dict
