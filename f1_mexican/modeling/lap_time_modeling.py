import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import pickle
import warnings

warnings.filterwarnings('ignore')

class LapTimeModelTrainingPipeline:
    def __init__(self, df):
        self.df = df
        self.embedding_columns = [
            'GridPositionEmbedding', 'WeightedCumulativeMeanCompoundEmbeddings', 
            'TeamEmbeddings', 'DriverEmbeddings', 'EventNameEmbeddings', 'CompoundEmbedding'
        ]
        self.target = 'LapTimeSeconds'
        self.model = None
        self.scaler = None

    def preprocess_data(self):
        """
        Preprocess the data: flatten embeddings, scale numerical features.
        """
        data = self.df.copy()

        # Flatten embedding columns
        for col in self.embedding_columns:
            embeddings = pd.DataFrame(data[col].tolist(), index=data.index)
            embeddings.columns = [f'{col}_{i}' for i in range(embeddings.shape[1])]
            data = pd.concat([data, embeddings], axis=1)

        # Drop original embedding columns
        data.drop(columns=self.embedding_columns, inplace=True)

        # Identify features and target
        X = data.drop(columns=[self.target])
        y = data[self.target]

        # Normalize numerical columns
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.scaler = StandardScaler()
        X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
        print('Numerical Columns', numerical_columns)
        return X, y

    def split_data(self, X, y, test_size=0.2):
        """
        Split the data into training and test sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def optimize_model(self, X_train, y_train):
        """
        Optimize the XGBoost model using GridSearchCV.
        """
        # Define the parameter grid for XGBoost
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.8, 1]
        }
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42
        )
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        print(f"Best parameters found: {grid_search.best_params_}")
        self.model = grid_search.best_estimator_
        return grid_search.best_estimator_

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on the test set.
        """
        predictions = self.model.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        print(f"Test Mean Squared Error: {mse:.4f}")
        return mse

    def feature_importance(self, X_train):
        """
        Display feature importances.
        """
        importances = self.model.feature_importances_
        feature_names = X_train.columns
        important_features = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        print("\nFeature Importances:")
        for feature, importance in important_features:
            print(f"{feature}: {importance:.4f}")
        return important_features

    def explain_model(self, X_train):
        """
        Use SHAP values to explain the model.
        """
        explainer = shap.Explainer(self.model, X_train)
        shap_values = explainer(X_train)
        # Plot summary of feature importance
        shap.summary_plot(shap_values, X_train, max_display=20)

    def save_model(self, filename):
        """
        Save the trained model and scaler as a pickle file.
        """
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
        print(f"Model saved to {filename}")

    def retrain_on_full_data(self, X, y):
        """
        Retrain the model on the full dataset (train + test) after optimization.
        """
        print("Retraining the model on the full dataset...")
        self.model.fit(X, y)

    def run(self):
        """
        Execute the training pipeline.
        """
        # Preprocess data
        X, y = self.preprocess_data()

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        print(X_train)

        # Optimize and train model
        self.optimize_model(X_train, y_train)

        # Evaluate model
        self.evaluate_model(X_test, y_test)

        # Display feature importances
        self.feature_importance(X_train)

        # Explain model predictions
        # self.explain_model(X_train)

        # Retrain on the full dataset
        self.retrain_on_full_data(X, y)

        # Save the final model
        self.save_model('models/final_lap_time_xgboost_model.pkl')
