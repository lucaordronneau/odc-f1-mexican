import numpy as np
import pandas as pd

from f1_mexican.utils.helpers import remove_outliers_iqr
from f1_mexican.constants import grid_pos_embeddings, team_mapping, tyre_compounds_embeddings, f1_drivers_embeddings, f1_teams_embeddings

class PreprocessPipeline:
    def __init__(self, lap_data, result_data, gp_name_map):
        self.lap_data = lap_data
        self.result_data = result_data
        self.gp_name_map = gp_name_map

    def run(self):
        self._map_grand_prix()
        self._merge_datasets()
        self._feature_engineering()
        self._compute_cumulative_points()
        self._filter_outliers()
        return self.df
    
    def _map_grand_prix(self):
        # Map circuit names to Grand Prix
        self.lap_data['EventName'] = self.lap_data['GP'].map(self.gp_name_map)
        self.result_data['EventName'] = self.result_data['GP'].map(self.gp_name_map)

    def _merge_datasets(self):
        # Merging lap and result data
        self.result_data_clean = self.result_data[["Year", 'EventName', "Abbreviation", "FullName", "TeamName", "Position", 
                                                   "ClassifiedPosition", 'GridPosition', 'Status', 'Points']]
        self.lap_data_clean = self.lap_data[['Year', 'EventName', 'Driver', 'Team', 'LapTime', 'LapNumber', 'Stint', 
                                             'PitOutTime', 'PitInTime', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 
                                             'Compound', 'TyreLife', 'FreshTyre', 'LapStartDate', 'Position', 'IsAccurate']]
        
        self.df = pd.merge(self.lap_data_clean, self.result_data_clean, 
                           left_on=["Year", 'EventName', "Driver"], 
                           right_on=["Year", 'EventName', "Abbreviation"], 
                           suffixes=["_lap", "_result"])

    def _feature_engineering(self):
        self.df['GridPositionCategory'] = self.df['GridPosition'].apply(self.classify_position)
        self.df['GridPositionEmbedding'] = self.df['GridPositionCategory'].map(grid_pos_embeddings)

        self.df['LapTime'] = pd.to_timedelta(self.df['LapTime'])
        self.df['LapTimeSeconds'] = self.df['LapTime'].dt.total_seconds()

        self.df['PitOutTime'] = pd.to_timedelta(self.df['PitOutTime']).shift(-1)
        self.df['PitInTime'] = pd.to_timedelta(self.df['PitInTime'])
        self.df['PitTime'] = (self.df['PitOutTime'] - self.df['PitInTime']).dt.total_seconds()

        teams_map = {team: t for t, teams in team_mapping.items() for team in teams}
        self.df['TeamOfficialName'] = self.df['TeamName'].map(teams_map)
        self.df['TeamEmbeddings'] = self.df['TeamOfficialName'].map(f1_teams_embeddings)
        self.df['DriverEmbeddings'] = self.df['Driver'].map(f1_drivers_embeddings)
        
        self.df["Compound"] = self.df["Compound"].astype(str)
        self.df['CompoundEmbeddings'] = self.df['Compound'].map(tyre_compounds_embeddings)
        
    def _compute_cumulative_points(self):
        # Handle cumulative points computation
        df_points = self.df[["Year", "EventName", "Driver", "TeamOfficialName", "Points"]].drop_duplicates(subset=["Year", "EventName", "Driver", "TeamOfficialName"])
        df_points['CumulativePoints'] = df_points.groupby(['Year', 'Driver', 'TeamOfficialName'])['Points'].cumsum()
        df_points['CumulativePoints'] = df_points.groupby(['Year', 'Driver', 'TeamOfficialName'])['CumulativePoints'].shift(1).fillna(0)

        # Merge cumulative points back into the main dataframe
        self.df = pd.merge(self.df, df_points[["Year", "EventName", "Driver", "TeamOfficialName", "CumulativePoints"]], 
                           on=["Year", "EventName", "Driver", "TeamOfficialName"])

    def classify_position(self, position):
        if 1 <= position <= 3:
            return "FRONT"
        elif 4 <= position <= 10:
            return "MID"
        else:
            return "BACK"

    def _filter_outliers(self):
        # Remove outliers from PitTime
        self.df = remove_outliers_iqr(self.df, 'PitTime')

        # Calculate lap delta and filter out outliers
        self.df['LapDelta'] = self.df.groupby(['Year', 'EventName', 'TeamOfficialName', 'Driver'])['LapTime'].diff()
        self.df['LapDelta'] = self.df['LapDelta'].dt.total_seconds()

        Q1 = self.df['LapDelta'].quantile(0.1)
        Q3 = self.df['LapDelta'].quantile(0.9)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        condition = (self.df['LapDelta'] > upper_bound) | (self.df['LapDelta'] < lower_bound)
        self.df.loc[condition, 'LapDelta'] = np.nan

        condition_laptime = (self.df['LapTimeSeconds'] > 1000)
        self.df.loc[condition_laptime, 'LapTimeSeconds'] = np.nan

        self.df.loc[~self.df['IsAccurate'], 'LapTimeSeconds'] = np.nan
        self.df.loc[~self.df['IsAccurate'], 'LapDelta'] = np.nan
