import numpy as np
from scipy.stats import norm

CS = 1.0
CH = 0.2
Q_STAR = CS / (CS + CH)
Z_Q = norm.ppf(Q_STAR)


def project_inventory(on_hand, transit_1, transit_2, fc_h1, fc_h2):
    after_h1 = max(on_hand + transit_1 - fc_h1, 0.0)
    after_h2 = max(after_h1 + transit_2 - fc_h2, 0.0)
    return after_h2


def compute_order(forecast_h1, forecast_h2, forecast_h3,
                  inventory_on_hand, in_transit_1, in_transit_2,
                  safety_margin=None):
    projected = project_inventory(
        inventory_on_hand, in_transit_1, in_transit_2, forecast_h1, forecast_h2
    )

    if safety_margin is not None:
        target = max(forecast_h3 + safety_margin, 0.0)
    else:
        sigma = np.sqrt(max(forecast_h3, 0.0))
        target = max(forecast_h3 + Z_Q * sigma, 0.0)

    return int(np.round(max(target - projected, 0.0)))
