import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm


def computed_lpd(log_probs):
    """
    Calculates pointwise log predictive density from posterior samples.
    See equation 3 at https://arxiv.org/abs/1507.04544

    Parameters:
        log_probs: Log probabilities from model
    Returns:
        Log predictive density for each data point
    """
    pointwise_probs = np.mean(np.exp(log_probs), axis=1)
    return np.log(pointwise_probs)


def waic(arviz_data):
    """
    Calculates Watanabe–Akaike information criterion
    See equation 11 at https://arxiv.org/abs/1507.04544

    Parameters:
        arviz_data: Arviz InferenceData object.
                    Must have a defined log_likelihood variable
    Returns:
        WAIC
        Pointwise log predictive density.
        Pointwise effective number of parameters.
            Values above 0.4 makes validity questionable, see paper above
    """
    log_probs_name = list(arviz_data.log_likelihood.keys())[0]
    log_probs = arviz_data.log_likelihood.stack(sample=("draw", "chain"))[
        log_probs_name
    ].values
    pointwise_lpd = computed_lpd(log_probs)
    pointwise_eff_params = np.var(log_probs, axis=1)
    lpd = np.sum(pointwise_lpd)
    eff_params = np.sum(pointwise_eff_params)
    return lpd - eff_params, pointwise_lpd, pointwise_eff_params


def waic_diff(model1, model2):
    model1_waic, model1_lpd, model1_eff_params = waic(model1)
    model2_waic, model2_lpd, model2_eff_params = waic(model2)

    diff = round(model1_waic - model2_waic, 2)
    diff_se = round(np.sqrt(len(model1_lpd) * np.var(model1_lpd - model2_lpd)), 2)
    diff_interval = [round(diff - 2.6 * diff_se, 2), round(diff + 2.6 * diff_se, 2)]

    print(f"Model 1 WAIC: {model1_waic}\nModel 2 WAIC: {model2_waic}")
    print(
        f"M1 - M2 Difference: {diff}\nStandard Error: {diff_se}\nDifference 99% Interval: {diff_interval[0]} | {diff_interval[1]}"
    )

    eff_param1_max = round(np.max(model1_eff_params), 2)
    eff_param2_max = round(np.max(model2_eff_params), 2)

    eff_param1_gt = round(
        sum(model1_eff_params > 0.4) / len(model1_eff_params) * 100, 2
    )
    eff_param2_gt = round(
        sum(model2_eff_params > 0.4) / len(model2_eff_params) * 100, 2
    )

    print(
        f"Max Variance for Model 1: {eff_param1_max}\nMax Variance for Model 2: {eff_param2_max}"
    )
    print(
        f"Percentage of values greater than 0.4 for Model 1: {eff_param1_gt}%\nPercentage of values greater than 0.4 for Model 2: {eff_param2_gt}%"
    )


def get_observed_data(arviz_data):
    """
    Finds name of observed data variable and returns it.

    Parameters:
        arviz_data: Arviz InferenceData object.
                    Must have defined observed data variable.

    Returns:
        NxM array where N is observed data length and M is number of posterior samples
    """
    data_keys = list(arviz_data.observed_data.keys())
    # If there is more than one variable name, raise exception
    if len(data_keys) == 1:
        obs_data_name = data_keys[0]
    else:
        raise ValueError("Observed data has more than 1 variable")

    obs_data_name = list(arviz_data.observed_data.keys())[0]
    obs_data = arviz_data.observed_data[obs_data_name].values
    return obs_data


def get_post_pred(arviz_data):
    """
    Finds name of posterior predictive data variable and returns it.

    Parameters:
        arviz_data: Arviz InferenceData object.
                    Must have defined posterior predictive variable.

    Returns:
        NxM array where N is observed data length and M is number of posterior samples
    """
    data_keys = list(arviz_data.posterior_predictive.keys())
    # If there is more than one variable name, raise exception
    if len(data_keys) == 1:
        post_pred_name = data_keys[0]
    else:
        raise ValueError("Posterior predictive has more than 1 variable")

    post_pred = arviz_data.posterior_predictive.stack(sample=("draw", "chain"))[
        post_pred_name
    ].values
    return post_pred


def plot_moment_checks(arviz_data):
    """
    Plots posterior predictive checks for the first 4 moments:
        mean, standard deviation, skew, and kurtosis.
    Black line is observed data, and histogram comes from the posterior predictive distribution.

    Parameters:
        arviz_data: Arviz InferenceData object.
                    Must have defined posterior predictive and observed data variables.

    Returns:
        plot
    """
    post_pred = get_post_pred(arviz_data)
    mean = np.mean(post_pred, axis=0)
    std = np.std(post_pred, axis=0)
    skew = stats.skew(post_pred, axis=0)
    kurt = stats.kurtosis(post_pred, axis=0)

    obs_data = get_observed_data(arviz_data)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(mean, bins=50)
    axs[0, 0].axvline(np.mean(obs_data), color="black")
    axs[0, 0].set_title("Mean")
    axs[0, 1].hist(std, bins=50)
    axs[0, 1].axvline(np.std(obs_data), color="black")
    axs[0, 1].set_title("Standard Deviation")
    axs[1, 0].hist(skew, bins=50)
    axs[1, 0].axvline(stats.skew(obs_data), color="black")
    axs[1, 0].set_title("Skew")
    axs[1, 1].hist(kurt, bins=50)
    axs[1, 1].axvline(stats.kurtosis(obs_data), color="black")
    axs[1, 1].set_title("Kurtosis")

    return fig


def exceedances(arviz_data):
    """
    Calculates exceedances given a range of percentiles.
    Ideally 5% of data should be below the 5th percentile, 25% below 25th percentile, etcetera.
    Checks observed data against posterior predictive.

    Parameters:
        arviz_data: Arviz InferenceData object.
                    Must have defined posterior predictive and observed data variables.

    Returns:
        dictionary with percentiles as keys and actual percentages as values
    """
    post_pred = get_post_pred(arviz_data)
    obs_data = get_observed_data(arviz_data)

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    actual = {}
    for p in percentiles:
        percent = np.sum(obs_data < np.percentile(post_pred, p, axis=1)) / len(obs_data)
        actual[p] = round(percent, 2) * 100
    return actual


def plot_qq(arviz_data):
    """
    Uses posterior predictive to transform observed data using probability integral transform.
    Returns QQ plot comparing this to a uniform. If estimated distribution matches the data,
    blue dots should lie on red line.

    Parameters:
        arviz_data: Arviz InferenceData object.
                    Must have defined posterior predictive and observed data variables.

    Returns:
        plot
    """
    obs_data = get_observed_data(arviz_data)
    post_pred = get_post_pred(arviz_data)
    values = []

    for t in range(len(obs_data)):
        ecdf = sm.distributions.empirical_distribution.ECDF(post_pred[t, :])
        values.append(ecdf(obs_data[t]))

    return sm.graphics.qqplot(np.array(values), dist=stats.uniform, line="45")
