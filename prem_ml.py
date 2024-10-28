import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
from typing import List, Tuple, Dict
from datetime import datetime

class PremierLeaguePredictor:
    """
    A class to predict Premier League match outcomes using historical data and machine learning.
    """
    
    def __init__(self, csv_path: str, split_date: str = '2022-01-01'):
        """
        Initialize the predictor with data and configuration.
        
        Args:
            csv_path: Path to the matches CSV file
            split_date: Date to split training and test data
        """
        self.split_date = split_date
        self.rf_model = RandomForestClassifier(
            n_estimators=50,
            min_samples_split=10,
            random_state=1
        )
        self.team_mapping = {
            "Brighton and Hove Albion": "Brighton",
            "Manchester United": "Manchester Utd",
            "Newcastle United": "Newcastle Utd",
            "Tottenham Hotspur": "Tottenham",
            "West Ham United": "West Ham",
            "Wolverhampton Wanderers": "Wolves"
        }
        self.load_and_prepare_data(csv_path)

    def load_and_prepare_data(self, csv_path: str) -> None:
        """
        Load and prepare the match data for analysis.
        """
        try:
            self.matches = pd.read_csv(csv_path, index_col=0)
            self._preprocess_data()
            self._create_rolling_averages()
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find CSV file at {csv_path}")
        except Exception as e:
            raise Exception(f"Error processing data: {str(e)}")

    def _preprocess_data(self) -> None:
        """
        Preprocess the raw match data.
        """
        # Validate required columns
        required_columns = ['date', 'venue', 'opponent', 'time', 'result']
        missing_columns = [col for col in required_columns if col not in self.matches.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.matches["date"] = pd.to_datetime(self.matches["date"])
        self.matches["venue_code"] = self.matches["venue"].astype("category").cat.codes
        self.matches["opp_code"] = self.matches["opponent"].astype("category").cat.codes
        
        # More robust time parsing
        self.matches["hour"] = pd.to_numeric(
            self.matches["time"].str.extract(r'(\d+):', expand=False),
            errors='coerce'
        )
        self.matches["day_code"] = self.matches["date"].dt.dayofweek
        self.matches["target"] = (self.matches["result"] == "W").astype("int")

    def _rolling_averages(self, group: pd.DataFrame, cols: List[str], new_cols: List[str]) -> pd.DataFrame:
        """
        Calculate rolling averages for specified columns.
        
        Args:
            group: DataFrame for a single team
            cols: Columns to calculate rolling averages for
            new_cols: Names for the new rolling average columns
        """
        group = group.sort_values("date")
        rolling_stats = group[cols].rolling(3, closed='left').mean()
        group[new_cols] = rolling_stats
        return group.dropna(subset=new_cols)

    def _create_rolling_averages(self) -> None:
        """
        Create rolling averages for all teams.
        """
        cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
        new_cols = [f"{c}_rolling" for c in cols]
        
        # Fix for deprecation warning by explicitly handling the grouping
        grouped = self.matches.groupby("team", group_keys=False)
        self.matches_rolling = pd.concat([
            self._rolling_averages(group, cols, new_cols)
            for name, group in grouped
        ])
        
        # Reset index properly
        self.matches_rolling = self.matches_rolling.reset_index(drop=True)
        self.rolling_cols = new_cols

    def make_predictions(self, predictors: List[str]) -> Tuple[pd.DataFrame, float]:
        """
        Make predictions using the random forest model.
        
        Args:
            predictors: List of predictor column names
            
        Returns:
            Tuple containing predictions DataFrame and error score
        """
        # Validate predictors
        missing_predictors = [pred for pred in predictors if pred not in self.matches_rolling.columns]
        if missing_predictors:
            raise ValueError(f"Missing predictor columns: {missing_predictors}")

        train = self.matches_rolling[self.matches_rolling["date"] < self.split_date]
        test = self.matches_rolling[self.matches_rolling["date"] > self.split_date]
        
        # Check for sufficient data
        if len(train) == 0 or len(test) == 0:
            raise ValueError("Insufficient data for train/test split")

        # Fit model and make predictions
        self.rf_model.fit(train[predictors], train["target"])
        preds = self.rf_model.predict(test[predictors])
        
        # Create detailed prediction DataFrame
        combined = pd.DataFrame({
            "date": test["date"],
            "team": test["team"],
            "opponent": test["opponent"],
            "actual": test["target"],
            "predicted": preds,
            "prediction_probability": self.rf_model.predict_proba(test[predictors])[:, 1]
        }, index=test.index)
        
        combined["new_team"] = combined["team"].map(
            lambda x: self.team_mapping.get(x, x)
        )
        
        error = precision_score(test["target"], preds)
        return combined, error

    def analyze_head_to_head(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze head-to-head predictions.
        
        Args:
            predictions: DataFrame containing predictions
            
        Returns:
            DataFrame with merged head-to-head predictions
        """
        merged = predictions.merge(
            predictions,
            left_on=["date", "new_team"],
            right_on=["date", "opponent"],
            suffixes=('_team1', '_team2')
        )
        return merged

    def get_model_metrics(self, predictions: pd.DataFrame) -> Dict:
        """
        Calculate and return comprehensive model metrics.
        
        Args:
            predictions: DataFrame containing predictions
            
        Returns:
            Dictionary containing various model metrics
        """
        return {
            'precision': precision_score(predictions['actual'], predictions['predicted']),
            'accuracy': accuracy_score(predictions['actual'], predictions['predicted']),
            'confusion_matrix': confusion_matrix(predictions['actual'], predictions['predicted']).tolist(),
            'classification_report': classification_report(predictions['actual'], predictions['predicted'])
        }

def main():
    """
    Main execution function.
    """
    try:
        # Initialize predictor
        predictor = PremierLeaguePredictor(
            r"C:\Git\Premier League Predictions\matches.csv"
        )
        
        # Define predictors
        base_predictors = ["venue_code", "opp_code", "hour", "day_code"]
        all_predictors = base_predictors + predictor.rolling_cols
        
        # Make predictions
        predictions, error = predictor.make_predictions(all_predictors)
        print(f"Prediction precision score: {error:.3f}")
        
        # Get comprehensive metrics
        metrics = predictor.get_model_metrics(predictions)
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        # Analyze head-to-head predictions
        h2h_analysis = predictor.analyze_head_to_head(predictions)
        
        # Show win probability validation
        win_validation = h2h_analysis[
            (h2h_analysis["predicted_team1"] == 1) & 
            (h2h_analysis["predicted_team2"] == 0)
        ]["actual_team1"].value_counts()
        
        print("\nWin prediction validation:")
        print(win_validation)
        
    except Exception as e:
        print(f"Error in prediction pipeline: {str(e)}")

if __name__ == "__main__":
    main()