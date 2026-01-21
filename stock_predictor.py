import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


class StockPredictor:
    """
    Stock price direction predictor using Random Forest and
    rolling technical features.
    """

    def __init__(
        self,
        n_estimators=200,
        min_samples_split=50,
        probability_threshold=0.6,
        random_state=1
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        self.threshold = probability_threshold

    def load_data(self, ticker, start_date="2000-01-01"):
        """
        Fetch historical stock data for a given ticker.
        """
        stock = yf.Ticker(ticker)
        df = stock.history(period="max")
        df = df.loc[start_date:].copy()
        return df

    def prepare_target(self, df):
        """
        Create target variable:
        1 -> price goes up tomorrow
        0 -> price goes down or stays the same
        """
        df["Tomorrow"] = df["Close"].shift(-1)
        df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
        return df

    def add_features(self, df):
        """
        Generate rolling technical indicators.
        """
        horizons = [2, 5, 60, 250, 1000]
        predictors = ["Close", "Volume", "Open", "High", "Low"]

        for horizon in horizons:
            rolling_avg = df["Close"].rolling(horizon).mean()
            df[f"Close_Ratio_{horizon}"] = df["Close"] / rolling_avg
            df[f"Trend_{horizon}"] = df["Target"].shift(1).rolling(horizon).sum()
            predictors += [f"Close_Ratio_{horizon}", f"Trend_{horizon}"]

        df.dropna(inplace=True)
        return df, predictors

    def backtest(self, df, predictors, start=2500, step=250):
        """
        Walk-forward backtesting for time-series data.
        """
        all_predictions = []

        for i in range(start, df.shape[0], step):
            train = df.iloc[:i]
            test = df.iloc[i:i + step]

            self.model.fit(train[predictors], train["Target"])
            probs = self.model.predict_proba(test[predictors])[:, 1]
            preds = (probs >= self.threshold).astype(int)

            combined = pd.DataFrame({
                "Target": test["Target"],
                "Prediction": preds
            }, index=test.index)

            all_predictions.append(combined)

        return pd.concat(all_predictions)

    def run(self, ticker):
        """
        End-to-end execution pipeline.
        """
        df = self.load_data(ticker)
        df = self.prepare_target(df)
        df, predictors = self.add_features(df)
        predictions = self.backtest(df, predictors)

        precision = precision_score(
            predictions["Target"],
            predictions["Prediction"]
        )

        latest_signal = predictions.iloc[-1]["Prediction"]

        return {
            "ticker": ticker.upper(),
            "precision": round(precision, 4),
            "latest_signal": "UP" if latest_signal == 1 else "DOWN"
        }
