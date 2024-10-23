import numpy as np
import pandas as pd
import pickle

from f1_mexican.constants import grid_pos_embeddings, drivers_teams, f1_drivers_embeddings, f1_teams_embeddings, grand_prix_embeddings

class StintNumberInferencePipeline:
    def __init__(self, df, model_path, event):
        # Load the saved model and scaler from the specified path
        with open(model_path, 'rb') as f:
            saved_objects = pickle.load(f)
            self.loaded_model = saved_objects['model']
            self.loaded_scaler = saved_objects['scaler']
        
        # Hard-coded variables
        self.year = 2024
        self.event = event  # You can change this if needed

        # Assign the dataframe
        self.df = df

        # Load embeddings and other necessary data
        # These should be loaded or defined appropriately
        self.drivers_teams = drivers_teams           # Should be loaded or defined
        self.f1_teams_embeddings = f1_teams_embeddings     # Should be loaded or defined
        self.f1_drivers_embeddings = f1_drivers_embeddings   # Should be loaded or defined
        self.grand_prix_embeddings = grand_prix_embeddings   # Should be loaded or defined
        self.grid_pos_embeddings = grid_pos_embeddings     # Should be loaded or defined
    
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

    def compute_driver_statistics(self, driver, grid_position, MaxNumber):
        # Set team and embeddings
        team = self.drivers_teams[driver]
        team_embedding = self.f1_teams_embeddings[team]
        driver_embedding = self.f1_drivers_embeddings[driver]
        event_embedding = self.grand_prix_embeddings[self.event]

        # Helper function to classify grid position
        def classify_position(position):
            if 1 <= position <= 3:
                return "FRONT"
            elif 4 <= position <= 10:
                return "MID"
            else:
                return "BACK"

        # Classify grid position and set embeddings
        text_grid_position = classify_position(grid_position)
        grid_position_embedding = self.grid_pos_embeddings[text_grid_position]

        # Extract historical data for the driver
        df_driver = self.df[self.df['Driver'] == driver]

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

        df_driver_event = self.df[(self.df['Driver'] == driver) & (self.df['EventName'] == self.event)]
        
        # Compute different cumulative means
        WeightedCumulativeMeanLapTimeSeconds = compute_weighted_cumulative_mean(df_driver_event, 'LapTimeSeconds')
        WeightedCumulativeMeanPitTimeSeconds = compute_weighted_cumulative_mean(df_driver_event, 'PitTime')
        WeightedCumulativeMeanLapDeltaSeconds = compute_weighted_cumulative_mean(df_driver_event, 'LapDelta')
        WeightedCumulativeMeanFinalPosition = compute_weighted_cumulative_mean(df_driver_event, 'Position_result')
        WeightedCumulativeMeanLapPosition = compute_weighted_cumulative_mean(df_driver_event, 'Position_lap')
        WeightedCumulativeMeanSpeedI1 = compute_weighted_cumulative_mean(df_driver_event, 'SpeedI1')
        WeightedCumulativeMeanSpeedI2 = compute_weighted_cumulative_mean(df_driver_event, 'SpeedI2')
        WeightedCumulativeMeanSpeedFL = compute_weighted_cumulative_mean(df_driver_event, 'SpeedFL')
        WeightedCumulativeMeanSpeedST = compute_weighted_cumulative_mean(df_driver_event, 'SpeedST')
        WeightedCumulativeMeanGridPosition = compute_weighted_cumulative_mean_grid_position(
            df_driver_event, 'GridPosition', self.year, grid_position)
        
        # Count number of races
        NumberOfRaces = len(df_driver_event.groupby(["Year", "EventName", "Driver"]))

        # Compute weighted cumulative mean for max stint
        def compute_weighted_cumulative_max(df, feature):
            max_per_year = df.groupby('Year')[feature].max().reset_index()
            max_per_year[f'expanding_mean_{feature}'] = max_per_year[feature].expanding().mean()
            return max_per_year[f'expanding_mean_{feature}'].iloc[-1]

        WeightedCumulativeMeanMaxStint = compute_weighted_cumulative_max(df_driver_event, 'Stint')

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

        WeightedCumulativeMeanCompoundEmbeddings_0, WeightedCumulativeMeanCompoundEmbeddings_1 = compute_weighted_cumulative_compound_embeddings(df_driver)

        return [
            self.year,
            driver_points_to_date,
            WeightedCumulativeMeanLapTimeSeconds,
            WeightedCumulativeMeanPitTimeSeconds,
            WeightedCumulativeMeanLapDeltaSeconds,
            WeightedCumulativeMeanFinalPosition,
            WeightedCumulativeMeanLapPosition,
            WeightedCumulativeMeanGridPosition,
            grid_position,
            WeightedCumulativeMeanSpeedI1,
            WeightedCumulativeMeanSpeedI2,
            WeightedCumulativeMeanSpeedFL,
            WeightedCumulativeMeanSpeedST,
            WeightedCumulativeMeanMaxStint,
            NumberOfRaces,
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
        ]

    def predict_stint(self, driver, grid_position, MaxNumber):
        features = self.compute_driver_statistics(driver, grid_position, MaxNumber)
        new_data = np.array([features])
        
        new_data_scaled = self.loaded_scaler.transform(new_data)

        stint_prediction = self.loaded_model.predict(new_data_scaled)
        return stint_prediction