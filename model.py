import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
import arviz as az
from model_testing import create_lags
import diagnostics
from scipy.optimize import minimize
from scipy.stats import norm

from datamodel import SPX, StockData


def bs_price(right, S, K, T, sigma, r):
    """
    Return's option price via Black-Scholes

    right: "P" or "C"
    S: Underlying price
    K: Strike price
    T: time to expiration (in fractions of a year)
    sigma: volatility of the underlying
    r: interest rate (in annual terms)
    """
    d1 = (1 / (sigma * np.sqrt(T))) * (np.log(S / K) + (r + sigma ** 2 / 2) * T)
    d2 = d1 - sigma * np.sqrt(T)

    if right == "C":
        price = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)
        return price

    if right == "P":
        price = norm.cdf(-d2) * K * np.exp(-r * T) - norm.cdf(-d1) * S
        return price


class ShortIronCondor:
    def __init__(self, chain, dte, risk_free_rate):
        """
        Class for all information required to determine what option position to enter into
        and the kelly sizing percentage

        chain: pandas dataframe containing options chain
        vols: sample of future WEEKLY volatilities
        dte: days to expiration of options contracts given in chain
        risk_free_rate: current market risk free rate of return given in annualized terms
        """
        self.chain = chain
        self.underlying_price = self.chain.iloc[0]["underprice"]
        # For checking
        print(f"SPX Price: {self.underlying_price}")
        self.dte = dte
        self.risk_free_rate = risk_free_rate

        self.calc_values()
        (
            self.short_put,
            self.short_call,
            self.long_put,
            self.long_call,
        ) = self.find_contracts()
        self.premium = (self.short_call["mid_price"] + self.short_put["mid_price"]) - (
            self.long_put["mid_price"] + self.long_call["mid_price"]
        )
        # Deflate the premium by 10% to be conservative and account for slippage
        self.premium = 0.90 * self.premium
        self.max_loss = (
            self.short_put["strike"] - self.long_put["strike"] - self.premium
        )

    def calc_values(self):
        """
        Calculates Mid price and skew premium for each option in chain
        """
        atm_contract_index = (
            np.abs(self.chain["strike"] - self.underlying_price)
        ).idxmin()
        atm_impliedvol = self.chain.iloc[atm_contract_index]["impvol"]

        # Calculate option value for all options using ATM volatility
        self.chain["model_value"] = self.chain.apply(
            lambda x: bs_price(
                x["right"],
                x["underprice"],
                x["strike"],
                self.dte / 252,
                atm_impliedvol,
                self.risk_free_rate,
            ),
            axis=1,
        )
        self.chain["mid_price"] = (self.chain["bid"] + self.chain["ask"]) / 2
        self.chain["skew_premium"] = self.chain["mid_price"] - self.chain["model_value"]

    def find_contracts(self):
        """
        Finds put contract with highest skew premium, then call contract with closest delta.
        Then picks hedging contracts on either side so that required margin equals $1000
        Essentially, picks contracts for short Iron Condor position.
        """
        # Select a put to short that is OTM
        short_put = self.chain[
            (self.chain["right"] == "P")
            & (self.chain["strike"] < self.underlying_price)
        ]["skew_premium"].idxmax()
        short_put = self.chain.iloc[short_put]
        # Buy put option so our margin required is $1000
        long_put = self.chain[
            (self.chain["strike"] == (short_put["strike"] - 10))
            & (self.chain["right"] == "P")
        ].squeeze()

        # Find the corresponding call option to make the position delta neutral
        put_contract_delta = short_put["delta"]
        short_call = np.abs(
            self.chain[self.chain["right"] == "C"]["delta"] + put_contract_delta
        ).idxmin()
        short_call = self.chain[self.chain["right"] == "C"].iloc[short_call]
        # Find respective call hedge option
        long_call = self.chain[
            (self.chain["strike"] == (short_call["strike"] + 10))
            & (self.chain["right"] == "C")
        ].squeeze()

        return short_put, short_call, long_put, long_call

    def pnl(self, underlying_price):
        """
        Calculates profit and loss of position at expiration given the underlying price.
        """
        # If underlying is between short strikes
        if self.short_call["strike"] >= underlying_price >= self.short_put["strike"]:
            pnl = self.premium
        # If underlying is under the short put
        elif underlying_price < self.short_put["strike"]:
            pnl = max(
                (underlying_price - self.short_put["strike"]) + self.premium,
                -self.max_loss,
            )
        # If underlying is above short call
        elif underlying_price > self.short_call["strike"]:
            pnl = max(
                (self.short_call["strike"] - underlying_price) + self.premium,
                -self.max_loss,
            )

        return pnl


def get_returns_forecast():
    spx = SPX()
    vix_data = StockData(["^VIX"])
    # For checking
    print(f"Data date: {spx.prices.index[-1]}")

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

    common_index = x.index.intersection(y.index)
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
            real h_tilde = normal_rng(mu_h + dot_product(x_tilde, beta), sigma);
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
        output_dir="./stan_model",
    )
    model_data = az.from_cmdstanpy(posterior=sample, posterior_predictive="r_tilde")
    return diagnostics.get_post_pred(model_data)


def log_wealth_optim(f, pnl):
    """
    Returns the negative of log wealth for optimization
    """
    return -np.mean(np.log(1 + f * pnl))


def calc_kelly(position: ShortIronCondor, returns):
    prices = position.underlying_price * (1 + returns)
    pnl_func = np.vectorize(position.pnl)
    pnls = pnl_func(prices) * 100
    # Max loss is $1000 dollars on condor, so compute return relative to that
    rets = pnls / 1000

    initial = 0.50
    result = minimize(
        log_wealth_optim,
        x0=initial,
        args=(rets),
        constraints=(
            {"type": "ineq", "fun": lambda x: x},
            {"type": "ineq", "fun": lambda x: 1 - x},
        ),
        tol=1e-10,
    )
    return result.x


def run_model():
    chain = pd.read_csv("option_chain.csv")
    position = ShortIronCondor(chain, 5, 0.01)
    print("Position computed")
    returns = get_returns_forecast()
    print("Returns forecast computed")
    kelly = calc_kelly(position, returns)
    print(
        f"""Short Call strike: {position.short_call["strike"]}\nShort Put strike: {position.short_put["strike"]}\nExpected Premium: {position.premium}\nKelly Percent: {round(float(kelly), 4) * 100}%\nReturns 95% interval: {np.percentile(returns, 5) * 100}% | {np.percentile(returns, 95) * 100}%"""
    )


if __name__ == "__main__":
    run_model()
