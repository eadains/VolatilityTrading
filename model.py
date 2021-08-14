import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import arviz as az
from model_testing import create_lags
import diagnostics
from scipy.optimize import minimize

from datamodel import SPX, StockData

plt.rcParams["figure.figsize"] = (15, 10)


class ShortButteryfly:
    def __init__(self, chain):
        self.chain = chain
        self.call, self.put = self.get_atm_options()

        self.underlying_price = self.call["underprice"]
        # Because of exception handling, put and call strikes are always the same
        self.strike = self.call["strike"]
        self.premium = self.calc_premium()

    def get_atm_options(self):
        calls = self.chain[self.chain["right"] == "C"]
        puts = self.chain[self.chain["right"] == "P"]

        atm_call = calls.loc[np.abs(calls["delta"] - 0.50).idxmin()]
        atm_put = puts.loc[np.abs(puts["delta"] + 0.50).idxmin()]

        # Make sure selected options have the same strike. This should always be the case.
        if atm_call["strike"] != atm_put["strike"]:
            raise ValueError("Strike of computed ATM call and put is not identical")

        return atm_call, atm_put

    def calc_premium(self):
        call_mid = (self.call["bid"] + self.call["ask"]) / 2
        put_mid = (self.put["bid"] + self.put["ask"]) / 2
        return call_mid + put_mid

    def calc_pnl(self, underlying_price):
        if underlying_price > self.strike:
            pnl = self.premium - (underlying_price - self.strike)
        elif underlying_price < self.strike:
            pnl = self.premium - (self.strike - underlying_price)
        return pnl


def get_returns_forecast():
    spx = SPX()
    vix_data = StockData(["^VIX"])

    spx_wk_prices = spx.prices.resample("W-FRI").last()
    spx_wk_returns = (np.log(spx_wk_prices) - np.log(spx_wk_prices.shift(1))).dropna()
    vix_wk_prices = vix_data.prices.VIX["close"].resample("W-FRI").last()
    spx_wk_vol = spx.vol.resample("W-FRI").sum()

    wk_returns_lags = create_lags(spx_wk_returns, 4, "wk_returns")
    wk_vix_lags = create_lags(vix_wk_prices, 4, "wk_vix")
    wk_vol_lags = create_lags(spx_wk_vol, 4, "wk_vol")

    x = pd.concat(
        [np.log(wk_vix_lags), np.log(wk_vol_lags), wk_returns_lags], axis=1
    ).dropna()
    y = spx_wk_returns.shift(-1).dropna()

    oos_x = x.iloc[-1]

    common_index = x.index.intersection(spx_wk_returns.index)
    x = x.loc[common_index]
    y = y.loc[common_index]

    model_spec = """
        data {
            int N;                              // Length of data
            int M;                              // Exogenous regressors dimensions
            vector[N] r;                        // SPX returns
            matrix[N, M] x;                     // Exogenous regressors data

            vector[M] x_tilde;                  // Most recent data for out-of-sample forecast
        }
        parameters {
            real mu_h;                          // Volatility mean term
            real mu_r;                          // Returns mean term
            vector[M] beta;                     // Exogenous regressors coefficients
            real<lower=0> sigma;                // Volatility noise
            vector[N] h_std;                    // Log volatility
            real<lower=0> rho;                  // Regularization term
            real alpha;                         // Skew normal shape parameter
        }
        transformed parameters {
            vector[N] h = h_std * sigma;        // h ~ normal(0, sigma);
            for (t in 1:N) {
                h[t] += mu_h + x[t] * beta;     // h ~ normal(mu_h + x * beta, sigma)
            }
        }
        model {
            rho ~ normal(0, 5);
            beta ~ normal(0, rho);
            sigma ~ normal(0, 10);
            mu_h ~ normal(0, 10);
            mu_r ~ normal(0, 10);
            alpha ~ normal(0, 10);
            
            h_std ~ std_normal();
            r ~ skew_normal(mu_r, exp(h / 2), alpha);
        }
        generated quantities {
            real h_tilde = normal_rng(mu_h + x_tilde * beta, sigma);
            real r_tilde = skew_normal_rng(mu_r, exp(h_tilde / 2), alpha);  // Forecasted return distribution for next week
        }
    """
    with open("./stan_model/model.stan", "w") as file:
        file.write(model_spec)

    model = CmdStanModel(stan_file="./stan_model/model.stan")
    data = {
        "N": len(y),
        "M": len(x.columns),
        "r": y.values,
        "x": x.values,
        "x_tilde": oos_x.values,
    }
    sample = model.sample(
        data=data,
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=2500,
    )
    model_data = az.from_cmdstanpy(posterior=sample, posterior_predictive="r_tilde")
    return diagnostics.get_post_pred(model_data)


def log_wealth_optim(f, pnl):
    """
    Returns the negative of log wealth for optimization
    """
    return -np.mean(np.log(1 + f * pnl))


def calc_kelly(position: ShortButteryfly, returns):
    prices = position.underlying_price * (1 + returns)
    pnl_func = np.vectorize(position.calc_pnl)
    pnls = pnl_func(prices)
    # Kelly function uses log of wealth, so it cannot be negative
    # This scales the pnl values so the lowest they ever go is 0
    # Because it's a monotonic transformation, the optimization
    # still finds the best point
    scaled_pnls = pnls / np.max(pnls)

    initial = 0.50
    result = minimize(log_wealth_optim, initial, (scaled_pnls))
    return result
