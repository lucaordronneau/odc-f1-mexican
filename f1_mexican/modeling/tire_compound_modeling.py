import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import shap
import pickle
import warnings

warnings.filterwarnings('ignore')

class TireCompoundModelTrainingPipeline:
    def __init__(self, df):
        self.df = df
        self.embedding_columns = [
            'GridPositionEmbedding', 'WeightedCumulativeMeanCompoundEmbeddings',
            'TeamEmbeddings', 'DriverEmbeddings', 'EventNameEmbeddings', 'ShiftedCompoundEmbedding'
        ]
        self.target = 'Compound'
        self.model = None
        self.scaler = None
        self.label_encoder = None

    def preprocess_data(self):
        """
        Preprocess the data: flatten embeddings, encode target variable, scale numerical features.
        """
        data = self.df.copy()

        # Flatten embedding columns
        for col in self.embedding_columns:
            embeddings = pd.DataFrame(data[col].tolist(), index=data.index)
            embeddings.columns = [f'{col}_{i}' for i in range(embeddings.shape[1])]
            data = pd.concat([data, embeddings], axis=1)

        # Drop original embedding columns
        data.drop(columns=self.embedding_columns, inplace=True)

        # Encode the target variable
        self.label_encoder = LabelEncoder()
        data[self.target] = self.label_encoder.fit_transform(data[self.target])

        # Identify features and target
        X = data.drop(columns=[self.target])
        y = data[self.target]

        # Normalize numerical columns
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.scaler = StandardScaler()
        X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
        
        print(numerical_columns)

        return X, y

    def split_data(self, X, y, test_size=0.2):
        """
        Split the data into training and test sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def optimize_model(self, X_train, y_train):
        """
        Optimize the XGBoost classifier using GridSearchCV.
        """
        # Define the parameter grid for XGBoost
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.8, 1]
        }
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            random_state=42,
            num_class=len(self.label_encoder.classes_)
        )
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='accuracy',
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
        accuracy = np.mean(predictions == y_test)
        print(f"Test Accuracy: {accuracy:.4f}")

        # Optionally, compute more detailed classification metrics
        from sklearn.metrics import classification_report, confusion_matrix
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=self.label_encoder.classes_))

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        print("\nConfusion Matrix:")
        print(cm)
        return accuracy

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
        for feature, importance in important_features[:20]:  # Display top 20 features
            print(f"{feature}: {importance:.4f}")
        return important_features

    def explain_model(self, X_train):
        """
        Use SHAP values to explain the model.
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_train)
        # Plot summary of feature importance
        shap.summary_plot(shap_values, X_train, class_names=self.label_encoder.classes_, max_display=20)

    def save_model(self, filename):
        """
        Save the trained model, scaler, and label encoder as a pickle file.
        """
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder
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

        # Optimize and train model
        self.optimize_model(X_train, y_train)

        # Evaluate model
        self.evaluate_model(X_test, y_test)

        # Display feature importances
        self.feature_importance(X_train)

        # Explain model predictions (optional)
        # self.explain_model(X_train)

        # Retrain on the full dataset
        self.retrain_on_full_data(X, y)

        # Save the final model
        self.save_model('models/final_tire_compound_xgboost_model.pkl')
