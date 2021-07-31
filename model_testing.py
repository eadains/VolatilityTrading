from abc import abstractmethod
import pandas as pd


def create_lags(series, lags, name="x"):
    """
    Creates a dataframe with lagged values of the given series.
    Generates columns named x_t-n which means the value of each row is the value of the original
    series lagged n times

    series: Pandas series
    lags: number of lagged values to include
    name: String to put as prefix for each column name

    Returns: Pandas dataframe
    """
    result = pd.DataFrame(index=series.index)
    result[f"{name}_t"] = series

    for n in range(lags):
        result[f"{name}_t-{n+1}"] = series.shift(n + 1)

    return result


class Model:
    """
    All model objects used for TimeSeriesCrossVal must have these methods
    """

    @abstractmethod
    def fit(self, train_x, train_y):
        """
        Fit model using training data.
        """
        pass

    @abstractmethod
    def loss(self, test_x, test_y):
        """
        Return object containing loss metrics using testing data.
        """
        pass


class BayesModel:
    """
    All model objects used for BayesTimeSeriesCrossVal must have these methods
    """

    @abstractmethod
    def fit(self, train_x, train_y, test_x, test_y):
        """
        Fit model using training data.
        """
        pass


class TimeSeriesCrossVal:
    """
    Conducts walk forward testing for time series models.
    Splits dataset into 'splits' number of parts.
    Train data is an expanding window.
    splits: number of splits to use for cross validation
    min_samples: minimum number of samples to include for training in the first split
    """

    def __init__(self, x, y, model, splits=10, min_samples=52):
        if len(x) != len(y):
            raise ValueError(
                "Independent variable and dependent variable must have same length"
            )
        self.x = x
        self.y = y
        self.model = model
        self.splits = splits
        self.min_samples = min_samples

    def index_gen(self):
        """
        Generates indexing numbers for training and testing data
        """
        split_length = (len(self.x) - self.min_samples) // self.splits
        for t in range(self.splits - 1):
            if t == 0:
                train_index = self.min_samples
            else:
                train_index = self.min_samples + (split_length * t)
            # When we're on the last test split, use entire rest of dataset.
            # This fixes rounding issues between number of splits and length of dataset.
            if t == (self.splits - 2):
                test_index = len(self.x)
            else:
                test_index = train_index + split_length
            yield t, train_index, test_index

    def walk_forward_test(self):
        """
        Conducts walk forward test.
        Returns: list of objects returned from model's 'loss' function
        """
        results = []
        for t, train_index, test_index in self.index_gen():
            train_x = self.x.iloc[:train_index]
            train_y = self.y.iloc[:train_index]
            self.model.fit(train_x, train_y)

            test_x = self.x.iloc[train_index:test_index]
            test_y = self.y.iloc[train_index:test_index]
            results.append(self.model.loss(test_x, test_y))
            print(f"Split {t} complete.")
        return results


class BayesTimeSeriesCrossVal:
    """
    Conducts walk forward testing for time series models in Stan.
    Splits dataset into 'splits' number of parts.
    Train data is an expanding window.
    splits: number of splits to use for cross validation
    min_samples: minimum number of samples to include for training in the first split
    """

    def __init__(self, x, y, model, splits=10, min_samples=52):
        if len(x) != len(y):
            raise ValueError(
                "Independent variable and dependent variable must have same length"
            )
        self.x = x
        self.y = y
        self.model = model
        self.splits = splits
        self.min_samples = min_samples

    def index_gen(self):
        """
        Generates indexing numbers for training and testing data
        """
        split_length = (len(self.x) - self.min_samples) // self.splits
        for t in range(self.splits - 1):
            if t == 0:
                train_index = self.min_samples
            else:
                train_index = self.min_samples + (split_length * t)
            # When we're on the last test split, use entire rest of dataset.
            # This fixes rounding issues between number of splits and length of dataset.
            if t == (self.splits - 2):
                test_index = len(self.x)
            else:
                test_index = train_index + split_length
            yield t, train_index, test_index

    def walk_forward_test(self):
        """
        Conducts walk forward test.
        Returns: list of objects returned from model's 'loss' function
        """
        results = []
        for t, train_index, test_index in self.index_gen():
            train_x = self.x.iloc[:train_index]
            train_y = self.y.iloc[:train_index]
            test_x = self.x.iloc[train_index:test_index]
            test_y = self.y.iloc[train_index:test_index]
            model_results = self.model.fit(train_x, train_y, test_x, test_y)
            results.append(model_results)
            print(f"Split {t} complete.")
        return results
