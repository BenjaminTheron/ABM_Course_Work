PARAMETERS = {
    "MC": 1,                   # Monte Carlo replications
    "T": 1200,                  # Number of trading sessions
    "NL": 10000,                # Number of low-frequency traders
    "NH": 100,                  # Number of high-frequency traders
    "theta": 20,                # LF traders' trading frequency mean
    "theta_min": 10,            # Min trading frequency
    "theta_max": 40,            # Max trading frequency
    "alpha_c": 0.04,            # Chartists' order size parameter
    "sigma_c": 0.05,            # Chartists' shock standard deviation
    "alpha_f": 0.04,            # Fundamentalists' order size parameter
    "sigma_f": 0.01,            # Fundamentalists' shock standard deviation
    "sigma_y": 0.01,            # Fundamental value shock standard deviation
    "delta": 0.0001,            # Price drift parameter
    "sigma_z": 0.01,            # LF traders' price tick standard deviation
    "zeta": 1,                  # LF traders' intensity of switching
    "gamma_L": 20,              # LF traders' resting order periods
    "gamma_H": 1,               # HF traders' resting order periods
    "eta_min": 0,               # HF traders' activation threshold min
    "eta_max": 0.2,             # HF traders' activation threshold max
    "lambda": 0.625,            # Market volumes weight in HF traders' order size distribution
    "kappa_min": 0,             # HF traders' order price distribution support min
    "kappa_max": 0.01,          # HF traders' order price distribution support max
}