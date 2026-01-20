"""
Classical Ensemble Methods Module

This module provides implementations and documentation for classical
ensemble learning methods that combine multiple models for improved
predictive performance.

Ensemble Methods:
- Random Forest (Breiman, 2001)
- AdaBoost (Freund & Schapire, 1997)
- Gradient Boosting (Friedman, 2001)
- XGBoost (Chen & Guestrin, 2016)
"""

from .random_forest import (
    RandomForestMethod,
    create_random_forest_entry,
)

from .boosting import (
    AdaBoostMethod,
    GradientBoostingMethod,
    create_adaboost_entry,
    create_gradient_boosting_entry,
)

from .xgboost import (
    XGBoostMethod,
    create_xgboost_entry,
)

__all__ = [
    # Random Forest
    "RandomForestMethod",
    "create_random_forest_entry",
    # AdaBoost
    "AdaBoostMethod",
    "create_adaboost_entry",
    # Gradient Boosting
    "GradientBoostingMethod",
    "create_gradient_boosting_entry",
    # XGBoost
    "XGBoostMethod",
    "create_xgboost_entry",
]
