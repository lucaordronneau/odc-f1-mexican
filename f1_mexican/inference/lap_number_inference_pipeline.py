import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder

# Assuming these constants are defined elsewhere and imported
from constants import (
    grid_pos_embeddings, drivers_teams, f1_drivers_embeddings, f1_teams_embeddings,
    grand_prix_embeddings, tyre_compounds_embeddings
)

class LapNumberInferencePipeline:
    def __init__(self, df, model_path, event):
        # Load the saved model, scaler, and label encoder from the specified path
        with open(model_path, 'rb') as f:
            saved_objects = pickle.load(f)
            self.loaded_model = saved_objects['model']
            self.loaded_scaler = saved_objects['scaler']
        
        # Assign the dataframe
        self.df = df
        
        # Hard-coded variables
        self.year = 2024
        self.event = event  # You can change this if needed

        # Load embeddings and other necessary data
        self.drivers_teams = drivers_teams
        self.f1_teams_embeddings = f1_teams_embeddings
        self.f1_drivers_embeddings = f1_drivers_embeddings
        self.grand_prix_embeddings = grand_prix_embeddings
        self.grid_pos_embeddings = grid_pos_embeddings

    def classify_position(self, position):
        if 1 <= position <= 3:
            return "FRONT"
        elif 4 <= position <= 10:
            return "MID"
        else:
            return "BACK"

    def compute_driver_statistics(self, driver, grid_position, stint_number, predicted_stint_number, compound, previous_lapnumber, previous_lapnumberbystint, previous_lapnumbertogo, MaxNumber):
        # Set team and embeddings
        team = self.drivers_teams[driver]
        team_embedding = self.f1_teams_embeddings[team]
        driver_embedding = self.f1_drivers_embeddings[driver]
        event_embedding = self.grand_prix_embeddings[self.event]

        # Classify grid position and set embeddings
        text_grid_position = self.classify_position(grid_position)
        grid_position_embedding = self.grid_pos_embeddings[text_grid_position]

        # Extract historical data for the driver
        df_driver = self.df[self.df['Driver'] == driver]

        # Compute cumulative points up to the current year
        df_points_inference = df_driver[["Year", "EventName", "Driver", "TeamOfficialName", "Points"]].drop_duplicates(
            subset=["Year", "EventName", "Driver", "TeamOfficialName"]
        )
        driver_points_to_date = df_points_inference[df_points_inference["Year"] == self.year]['Points'].sum()

        # Function to compute weighted cumulative mean for a given feature
        def compute_weighted_cumulative_mean(df, feature):
            mean_per_year = df.groupby('Year')[feature].mean().reset_index()
            mean_per_year[f'expanding_mean_{feature}'] = mean_per_year[feature].expanding().mean()
            return mean_per_year[f'expanding_mean_{feature}'].iloc[-1]
        
        # Compute weighted cumulative mean for grid position with an added year
        def compute_weighted_cumulative_mean_grid_position(df, feature, year, grid_position):
            mean_per_year = df.groupby('Year')[feature].mean().reset_index()
            # Add the new year and grid position
            new_row = pd.DataFrame({'Year': [year], feature: [grid_position]})
            mean_per_year = pd.concat([mean_per_year, new_row], ignore_index=True)
            mean_per_year = mean_per_year.sort_values(by='Year').reset_index(drop=True)
            mean_per_year[f'expanding_mean_{feature}'] = mean_per_year[feature].expanding().mean()
            return mean_per_year[f'expanding_mean_{feature}'].iloc[-1]

        # Compute different cumulative means using historical data up to the current year
        df_driver_event = self.df[(self.df['Driver'] == driver) & (self.df['EventName'] == self.event) & (self.df['Stint'] == stint_number)]
        if not len(df_driver_event):
            df_driver_event = self.df[(self.df['Driver'] == driver) & (self.df['EventName'] == self.event)]
        
        WeightedCumulativeMeanLapTimeSeconds = compute_weighted_cumulative_mean(df_driver_event, 'LapTimeSeconds')
        WeightedCumulativeMeanLapDeltaSeconds = compute_weighted_cumulative_mean(df_driver_event, 'LapDelta')
        WeightedCumulativeMeanFinalPosition = compute_weighted_cumulative_mean(df_driver_event, 'Position_result')
        WeightedCumulativeMeanLapPosition = compute_weighted_cumulative_mean(df_driver_event, 'Position_lap')
        WeightedCumulativeMeanGridPosition = compute_weighted_cumulative_mean(df_driver_event, 'GridPosition')
        WeightedCumulativeMeanSpeedI1 = compute_weighted_cumulative_mean(df_driver_event, 'SpeedI1')
        WeightedCumulativeMeanSpeedI2 = compute_weighted_cumulative_mean(df_driver_event, 'SpeedI2')
        WeightedCumulativeMeanSpeedFL = compute_weighted_cumulative_mean(df_driver_event, 'SpeedFL')
        WeightedCumulativeMeanSpeedST = compute_weighted_cumulative_mean(df_driver_event, 'SpeedST')
        WeightedCumulativeMeanGridPosition = compute_weighted_cumulative_mean_grid_position(
            df_driver_event, 'GridPosition', self.year, grid_position)
        
        NumberOfRacesOnStint = df_driver_event.drop_duplicates(["Year", "Driver", "TeamOfficialName", "EventName", "Stint"]).groupby(['Driver', 'Stint']).cumcount().to_frame(name='RaceNumber')
        df_display = df_driver_event[['Year', 'EventName', 'Driver', 'Stint']].drop_duplicates(['Year', 'EventName', 'Driver', 'Stint'])
        df_display = pd.concat([df_display, NumberOfRacesOnStint], axis=1)
        df_driver_event = pd.merge(df_driver_event, df_display, on=['Year', 'EventName', 'Driver', 'Stint'])
        
        try:
            NumberOfRacesOnStint = df_display[(df_display["Stint"] == stint_number)]["RaceNumber"].iloc[-1]
        except:
            NumberOfRacesOnStint = 0

        # Count number of races
        df_driver_event_tmp = self.df[(self.df['Driver'] == driver) & (self.df['EventName'] == self.event)]
        RaceNumber = len(df_driver_event_tmp.groupby(["Year", "EventName", "Driver"]))

        GridPositionEmbedding_0 = grid_position_embedding[0]
        GridPositionEmbedding_1 = grid_position_embedding[1]

        # Compute weighted cumulative mean compound embeddings
        def compute_weighted_cumulative_compound_embeddings(df):
            result = df.groupby('Year').apply(self.generalized_weighted_mean_compound).reset_index()
            expanded_embeddings = pd.DataFrame(
                result['WeightedCompoundEmbeddings'].tolist(),
                columns=['WeightedCompoundEmbeddings_0', 'WeightedCompoundEmbeddings_1']
            )
            expanded_embeddings['expanding_mean_0'] = expanded_embeddings['WeightedCompoundEmbeddings_0'].expanding().mean()
            expanded_embeddings['expanding_mean_1'] = expanded_embeddings['WeightedCompoundEmbeddings_1'].expanding().mean()
            return expanded_embeddings['expanding_mean_0'].iloc[-1], expanded_embeddings['expanding_mean_1'].iloc[-1]

        new_row = {col: np.nan for col in df_driver.columns}
        tyre_compounds_embeddings_inversed = {tuple(v): k for k, v in tyre_compounds_embeddings.items()}
        new_row["CompoundEmbeddings"] = compound
        new_row["Compound"] = tyre_compounds_embeddings_inversed[tuple(compound)]
        new_row["Year"] = 2024
        new_row_df = pd.DataFrame([new_row])
        df_driver = pd.concat([df_driver, new_row_df], ignore_index=True)

        WeightedCumulativeMeanCompoundEmbeddings_0, WeightedCumulativeMeanCompoundEmbeddings_1 = compute_weighted_cumulative_compound_embeddings(df_driver)

        return [
            self.year,
            driver_points_to_date,
            WeightedCumulativeMeanLapTimeSeconds,
            WeightedCumulativeMeanLapDeltaSeconds,
            WeightedCumulativeMeanFinalPosition,
            WeightedCumulativeMeanLapPosition,
            WeightedCumulativeMeanGridPosition,
            grid_position,
            WeightedCumulativeMeanSpeedI1,
            WeightedCumulativeMeanSpeedI2,
            WeightedCumulativeMeanSpeedFL,
            WeightedCumulativeMeanSpeedST,
            NumberOfRacesOnStint,
            RaceNumber,
            stint_number,
            predicted_stint_number,
            previous_lapnumber,
            previous_lapnumberbystint,
            previous_lapnumbertogo,
            MaxNumber,
            GridPositionEmbedding_0,
            GridPositionEmbedding_1,
            WeightedCumulativeMeanCompoundEmbeddings_0,
            WeightedCumulativeMeanCompoundEmbeddings_1,
            team_embedding[0],    # TeamEmbeddings_0
            team_embedding[1],    # TeamEmbeddings_1
            driver_embedding[0],  # DriverEmbeddings_0
            driver_embedding[1],  # DriverEmbeddings_1
            event_embedding[0],   # EventNameEmbeddings_0
            event_embedding[1],   # EventNameEmbeddings_1
            compound[0],
            compound[1], 
        ]

    def generalized_weighted_mean_compound(self, group):
        # Convert embeddings to numpy array
        embeddings = np.array(group['CompoundEmbeddings'].tolist())
        
        # Get the unique compounds and their counts
        compound_counts = group['Compound'].value_counts()
        
        # Initialize weighted sum and total weight
        weighted_sum = np.zeros(embeddings[0].shape)
        total_weight = 0
        
        # Loop through each compound and calculate its contribution
        for compound, count in compound_counts.items():
            compound_embeddings = embeddings[group['Compound'] == compound]
            weighted_sum += compound_embeddings.sum(axis=0)  # Sum of embeddings for the compound
            total_weight += count  # Add the count to the total weight
        
        # Calculate the weighted average of embeddings
        weighted_avg = weighted_sum / total_weight
        
        return pd.Series([list(weighted_avg)], index=['WeightedCompoundEmbeddings'])

    def predict_lap_number(self, driver, grid_position, predicted_stint_number, compounds, lapnumber_max):
        lap_number_by_stint_result = []
        previous_lapnumber = 0
        previous_lapnumberbystint = 0
        max_tmp = lapnumber_max

        for stint_number in range(1, predicted_stint_number + 1):
            compound = compounds[stint_number-1]

            tyre_compounds_embeddings_tmp = {
                "HARD": [0.9592574834823608, 0.2825334072113037],
                "MEDIUM": [0.9999638795852661, 0.008497972041368484],
                "SOFT": [0.982682466506958, 0.1852976679801941],
                "SUPERSOFT": [0.9986254572868347, -0.05241385102272034],
                "HYPERSOFT": [-0.8185507655143738, 0.5744342803955078],
                "ULTRASOFT": [0.9489226341247559, 0.31550878286361694],
                "INTERMEDIATE": [0.8133333325386047, 0.5817979574203491],
                "WET": [0.5332446694374084, 0.8459610342979431],
                "nan": [0, 0],
            }
            compound = tyre_compounds_embeddings_tmp[compound]

            # Compute features for the current stint
            features = self.compute_driver_statistics(
                driver, grid_position, stint_number, predicted_stint_number, compound, previous_lapnumber, previous_lapnumberbystint, lapnumber_max, max_tmp
            )

            # Convert features to DataFrame
            feature_columns = [
                'Year',
                'CumulativePoints',
                'WeightedCumulativeMeanLapTimeSeconds',
                'WeightedCumulativeMeanLapDeltaSeconds',
                'WeightedCumulativeMeanFinalPosition',
                'WeightedCumulativeMeanLapPosition',
                'WeightedCumulativeMeanGridPosition',
                'GridPosition',
                'WeightedCumulativeMeanSpeedI1',
                'WeightedCumulativeMeanSpeedI2',
                'WeightedCumulativeMeanSpeedFL',
                'WeightedCumulativeMeanSpeedST',
                'NumberOfRacesOnStint',
                'RaceNumber',
                'Stint',
                'StintNumber',
                'ShiftLapNumber',
                'ShiftLapNumberByStint',
                'ShiftLapNumberToGo',
                'MaxNumber',
                'GridPositionEmbedding_0',
                'GridPositionEmbedding_1',
                'WeightedCumulativeMeanCompoundEmbeddings_0',
                'WeightedCumulativeMeanCompoundEmbeddings_1',
                'TeamEmbeddings_0',
                'TeamEmbeddings_1',
                'DriverEmbeddings_0',
                'DriverEmbeddings_1',
                'EventNameEmbeddings_0',
                'EventNameEmbeddings_1',
                'CompoundEmbedding_0',
                'CompoundEmbedding_1'
            ]
            
            dictionary = dict(zip(feature_columns, features))

            feature_df = pd.DataFrame([features], columns=feature_columns)

            # Ensure the order of features matches the training data
            # Apply scaling to numerical features
            numerical_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_df_scaled = feature_df.copy()
            feature_df_scaled[numerical_columns] = self.loaded_scaler.transform(feature_df[numerical_columns])

            # Make prediction
            lap_number_on_stint = self.loaded_model.predict(feature_df_scaled)

            if stint_number == (predicted_stint_number):
                previous_lapnumberbystint = max_tmp - previous_lapnumber
            else:
                previous_lapnumber+=round(lap_number_on_stint[0])
                previous_lapnumberbystint=round(lap_number_on_stint[0])
                lapnumber_max-=previous_lapnumber
            lap_number_by_stint_result.append(previous_lapnumberbystint)
        return lap_number_by_stint_result