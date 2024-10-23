import numpy as np
import pandas as pd

from f1_mexican.constants import f1_drivers_embeddings, f1_teams_embeddings, grand_prix_embeddings


class StintNumberFeatureEngineeringPipeline:
    def __init__(self, df):
        """
        Initialize the pipeline with the input dataframe.
        Args:
        df (pd.DataFrame): The input dataframe containing F1 data.
        """
        self.df = df
        self.result_drivers_df = []

    @staticmethod
    def calculate_weighted_feature(df, feature, group_cols, method='mean'):
        """
        Generalized function to calculate the weighted cumulative mean or max for a given feature.
        
        Args:
        df (pd.DataFrame): The input dataframe.
        feature (str): The feature column to process.
        group_cols (list of str): Columns to group by.
        method (str): 'mean' or 'max' to apply to the feature.
        
        Returns:
        pd.DataFrame: A DataFrame with weighted cumulative mean/max results.
        """
        if method == 'mean':
            mean_per_year = df.groupby(['Year'] + group_cols, as_index=False)[feature].mean().reset_index()
            if feature == "GridPosition":
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.expanding().mean())
            else:
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.shift(1).expanding().mean())

        elif method == 'max':
            mean_per_year = df.groupby(['Year'] + group_cols, as_index=False)[feature].max().reset_index()
            if feature == "GridPosition":
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.expanding().mean())
            else:
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.shift(1).expanding().mean())

        return weighted.reset_index(drop=True)

    @staticmethod
    def generalized_weighted_mean_compound(group):
        """
        Function to compute the weighted average of compound embeddings.
        
        Args:
        group (pd.DataFrame): Grouped dataframe containing compound embeddings.
        
        Returns:
        pd.Series: A pandas Series containing the weighted average embeddings.
        """
        embeddings = np.array(group['CompoundEmbeddings'].tolist())
        compound_counts = group['Compound'].value_counts()

        weighted_sum = np.zeros(embeddings[0].shape)
        total_weight = 0

        for compound, count in compound_counts.items():
            compound_embeddings = embeddings[group['Compound'] == compound]
            weighted_sum += compound_embeddings.sum(axis=0)
            total_weight += count

        weighted_avg = weighted_sum / total_weight
        return pd.Series([list(weighted_avg)], index=['WeightedCompoundEmbeddings'])

    def process_event_and_driver(self, event, driver):
        """
        Process feature engineering for a single event and driver.
        
        Args:
        event (str): The event name.
        driver (str): The driver's name.
        """
        df_driver = self.df[(self.df["EventName"] == event) & (self.df["Driver"] == driver)]
        max_lap_event = self.df[self.df["EventName"] == event]["LapNumber"].max()
        if len(df_driver):
            # print(f"Processing driver: {driver} at event: {event}")

            result_df = df_driver[['Year', 'TeamOfficialName', 'EventName', 'Driver', 'CumulativePoints']].drop_duplicates(
                subset=['Year', 'EventName', 'TeamOfficialName', 'Driver']).reset_index(drop=True)

            result_df['WeightedCumulativeMeanLapTimeSeconds'] = self.calculate_weighted_feature(df_driver, 'LapTimeSeconds', ['TeamOfficialName', 'Driver'], method='mean')
            result_df['WeightedCumulativeMeanPitTimeSeconds'] = self.calculate_weighted_feature(df_driver, 'PitTime', ['TeamOfficialName', 'Driver'], method='mean')
            result_df['WeightedCumulativeMeanLapDeltaSeconds'] = self.calculate_weighted_feature(df_driver, 'LapDelta', ['TeamOfficialName', 'Driver'], method='mean')
            result_df['WeightedCumulativeMeanFinalPosition'] = self.calculate_weighted_feature(df_driver, 'Position_result', ['TeamOfficialName', 'Driver'], method='mean')
            result_df['WeightedCumulativeMeanLapPosition'] = self.calculate_weighted_feature(df_driver, 'Position_lap', ['TeamOfficialName', 'Driver'], method='mean')
            result_df['WeightedCumulativeMeanGridPosition'] = self.calculate_weighted_feature(df_driver, 'GridPosition', ['TeamOfficialName', 'Driver'], method='mean')
            result_df['GridPosition'] = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver'])['GridPosition'].max().values
            result_df['GridPositionEmbedding'] = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver'])['GridPositionEmbedding'].last().values
            result_df['WeightedCumulativeMeanSpeedI1'] = self.calculate_weighted_feature(df_driver, 'SpeedI1', ['TeamOfficialName', 'Driver'], method='mean')
            result_df['WeightedCumulativeMeanSpeedI2'] = self.calculate_weighted_feature(df_driver, 'SpeedI2', ['TeamOfficialName', 'Driver'], method='mean')
            result_df['WeightedCumulativeMeanSpeedFL'] = self.calculate_weighted_feature(df_driver, 'SpeedFL', ['TeamOfficialName', 'Driver'], method='mean')
            result_df['WeightedCumulativeMeanSpeedST'] = self.calculate_weighted_feature(df_driver, 'SpeedST', ['TeamOfficialName', 'Driver'], method='mean')
            result_df['WeightedCumulativeMeanMaxStint'] = self.calculate_weighted_feature(df_driver, 'Stint', ['TeamOfficialName', 'Driver'], method='max')

            result = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver'], as_index=False).apply(self.generalized_weighted_mean_compound)

            expanded_embeddings = pd.DataFrame(result['WeightedCompoundEmbeddings'].tolist(), columns=['WeightedCompoundEmbeddings_0', 'WeightedCompoundEmbeddings_1'])
            expanded_embeddings['WeightedCompoundEmbeddings_0'] = expanded_embeddings.groupby([result['TeamOfficialName'], result['Driver']])['WeightedCompoundEmbeddings_0'].transform(lambda x: x.shift(1).expanding().mean())
            expanded_embeddings['WeightedCompoundEmbeddings_1'] = expanded_embeddings.groupby([result['TeamOfficialName'], result['Driver']])['WeightedCompoundEmbeddings_1'].transform(lambda x: x.shift(1).expanding().mean())

            result_df['WeightedCumulativeMeanCompoundEmbeddings'] = expanded_embeddings.apply(lambda row: [row['WeightedCompoundEmbeddings_0'], row['WeightedCompoundEmbeddings_1']], axis=1)
            result_df['NumberOfRaces'] = result_df.groupby(['Driver']).cumcount()
            result_df['StintNumber'] = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver'])['Stint'].max().values
            result_df["MaxNumber"] = max_lap_event
            self.result_drivers_df.append(result_df)

    def run(self):
        """
        Execute the feature engineering process for all events and drivers.
        """
        drivers = self.df["Driver"].unique().tolist()
        events = self.df["EventName"].unique().tolist()

        for event in events:
            for driver in drivers:
                self.process_event_and_driver(event, driver)

        df_final = pd.concat(self.result_drivers_df).reset_index(drop=True).dropna()
        df_final['TeamEmbeddings'] = df_final['TeamOfficialName'].map(f1_teams_embeddings)
        df_final['DriverEmbeddings'] = df_final['Driver'].map(f1_drivers_embeddings)
        df_final['EventNameEmbeddings'] = df_final['EventName'].map(grand_prix_embeddings)

        return df_final
    

class TireCompoundFeatureEngineeringPipeline:
    def __init__(self, df):
        """
        Initialize the pipeline with the input dataframe.
        Args:
        df (pd.DataFrame): The input dataframe containing F1 data.
        """
        self.df = df
        self.result_drivers_df = []

    @staticmethod
    def calculate_weighted_feature(df, feature, group_cols, method='mean'):
        """
        Generalized function to calculate the weighted cumulative mean or max for a given feature.
        
        Args:
        df (pd.DataFrame): The input dataframe.
        feature (str): The feature column to process.
        group_cols (list of str): Columns to group by.
        method (str): 'mean' or 'max' to apply to the feature.
        
        Returns:
        pd.DataFrame: A DataFrame with weighted cumulative mean/max results.
        """
        if method == 'mean':
            mean_per_year = df.groupby(['Year'] + group_cols, as_index=False)[feature].mean().reset_index()
            if (feature == "GridPosition") or (feature == 'Stint'):
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.expanding().mean())
            else:
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.shift(1).expanding().mean())

        elif method == 'max':
            mean_per_year = df.groupby(['Year'] + group_cols, as_index=False)[feature].max().reset_index()
            if (feature == "GridPosition") or (feature == 'Stint'):
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.expanding().mean())
            else:
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.shift(1).expanding().mean())

        return weighted.reset_index(drop=True)

    @staticmethod
    def generalized_weighted_mean_compound(group):
        """
        Function to compute the weighted average of compound embeddings.
        
        Args:
        group (pd.DataFrame): Grouped dataframe containing compound embeddings.
        
        Returns:
        pd.Series: A pandas Series containing the weighted average embeddings.
        """
        embeddings = np.array(group['CompoundEmbeddings'].tolist())
        compound_counts = group['Compound'].value_counts()

        weighted_sum = np.zeros(embeddings[0].shape)
        total_weight = 0

        for compound, count in compound_counts.items():
            compound_embeddings = embeddings[group['Compound'] == compound]
            weighted_sum += compound_embeddings.sum(axis=0)
            total_weight += count

        weighted_avg = weighted_sum / total_weight
        return pd.Series([list(weighted_avg)], index=['WeightedCompoundEmbeddings'])

    def process_event_and_driver(self, event, driver):
        """
        Process feature engineering for a single event and driver.
        
        Args:
        event (str): The event name.
        driver (str): The driver's name.
        """
        df_driver = self.df[(self.df["EventName"] == event) & (self.df["Driver"] == driver)]
        max_lap_event = self.df[self.df["EventName"] == event]["LapNumber"].max()
        if len(df_driver):
            
            result_df = df_driver[['Year', 'TeamOfficialName', 'EventName', 'Driver', 'Stint', 'Compound', 'CumulativePoints']].groupby(
                ['Year', 'EventName', 'TeamOfficialName', 'Driver', 'Stint'], as_index=False
            ).agg({"CumulativePoints": "last", "Compound":"last"})

            result_df['WeightedCumulativeMeanLapTimeSeconds'] = self.calculate_weighted_feature(df_driver, 'LapTimeSeconds', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanLapDeltaSeconds'] = self.calculate_weighted_feature(df_driver, 'LapDelta', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanFinalPosition'] = self.calculate_weighted_feature(df_driver, 'Position_result', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanLapPosition'] = self.calculate_weighted_feature(df_driver, 'Position_lap', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanGridPosition'] = self.calculate_weighted_feature(df_driver, 'GridPosition', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['GridPosition'] = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver', 'Stint'])['GridPosition'].max().values
            result_df['GridPositionEmbedding'] = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver', 'Stint'])['GridPositionEmbedding'].last().values
            result_df['WeightedCumulativeMeanSpeedI1'] = self.calculate_weighted_feature(df_driver, 'SpeedI1', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanSpeedI2'] = self.calculate_weighted_feature(df_driver, 'SpeedI2', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanSpeedFL'] = self.calculate_weighted_feature(df_driver, 'SpeedFL', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanSpeedST'] = self.calculate_weighted_feature(df_driver, 'SpeedST', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanMaxStint'] = self.calculate_weighted_feature(df_driver, 'Stint', ['TeamOfficialName', 'Driver', 'Stint'], method='max')

            result = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver', 'Stint'], as_index=False).apply(self.generalized_weighted_mean_compound)

            expanded_embeddings = pd.DataFrame(result['WeightedCompoundEmbeddings'].tolist(), columns=['WeightedCompoundEmbeddings_0', 'WeightedCompoundEmbeddings_1'])
            expanded_embeddings['WeightedCompoundEmbeddings_0'] = expanded_embeddings.groupby([result['TeamOfficialName'], result['Driver'], result['Stint']])['WeightedCompoundEmbeddings_0'].transform(lambda x: x.shift(1).expanding().mean())
            expanded_embeddings['WeightedCompoundEmbeddings_1'] = expanded_embeddings.groupby([result['TeamOfficialName'], result['Driver'], result['Stint']])['WeightedCompoundEmbeddings_1'].transform(lambda x: x.shift(1).expanding().mean())

            result_df['WeightedCumulativeMeanCompoundEmbeddings'] = expanded_embeddings.apply(lambda row: [row['WeightedCompoundEmbeddings_0'], row['WeightedCompoundEmbeddings_1']], axis=1)
            result_df['NumberOfRacesOnStint'] = result_df.groupby(['Driver', 'Stint']).cumcount()
            
            race_number_df = result_df.drop_duplicates(["Year", "Driver", "TeamOfficialName", "EventName"]).groupby(['Driver'], as_index=False).cumcount()
            race_number_df = race_number_df.reset_index(drop=True).to_frame(name='RaceNumber')
            race_number_df = pd.concat([result_df.groupby(["Year", "Driver", "TeamOfficialName", "EventName"], as_index=False).count()[["Year", "Driver", "TeamOfficialName", "EventName"]], race_number_df], axis=1)
            result_df = pd.merge(result_df, race_number_df, on=["Year", "Driver", "TeamOfficialName", "EventName"])
            
            stint_number_df = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver'], as_index=False)['Stint'].max()
            result_df = pd.merge(result_df, stint_number_df, on=['Year', 'TeamOfficialName', 'Driver'], suffixes=('', '_max_stint'))
            result_df['StintNumber'] = result_df['Stint_max_stint']
            
            tyre_compounds_embeddings_tmp = {
                "HARD": [0.9592574834823608, 0.2825334072113037],
                "MEDIUM": [0.9999638795852661, 0.008497972041368484],
                "SOFT": [0.982682466506958, 0.1852976679801941],
                "SUPERSOFT": [0.9986254572868347, -0.05241385102272034],
                "HYPERSOFT": [-0.8185507655143738, 0.5744342803955078],
                "ULTRASOFT": [0.9489226341247559, 0.31550878286361694],
                "INTERMEDIATE": [0.8133333325386047, 0.5817979574203491],
                "WET": [0.5332446694374084, 0.8459610342979431],
                "nan": [0.0, 0.0],
                "NaN": [0.0, 0.0],
            }
            
            result_df['ShiftedCompound'] = result_df.groupby(['Year', 'TeamOfficialName', 'Driver'])['Compound'].shift(1)
            result_df['ShiftedCompound'] = result_df['ShiftedCompound'].astype(str)
            result_df['ShiftedCompoundEmbedding'] = result_df['ShiftedCompound'].map(tyre_compounds_embeddings_tmp)

            result_df["MaxNumber"] = max_lap_event

            self.result_drivers_df.append(result_df)

    def run(self):
        """
        Execute the feature engineering process for all events and drivers.
        """
        drivers = self.df["Driver"].unique().tolist()
        events = self.df["EventName"].unique().tolist()

        for event in events:
            for driver in drivers:
                try:
                    self.process_event_and_driver(event, driver)
                except:
                    print(driver, event)

        df_final = pd.concat(self.result_drivers_df).reset_index(drop=True)
        df_final = df_final.dropna()

        df_final['TeamEmbeddings'] = df_final['TeamOfficialName'].map(f1_teams_embeddings)
        df_final['DriverEmbeddings'] = df_final['Driver'].map(f1_drivers_embeddings)
        df_final['EventNameEmbeddings'] = df_final['EventName'].map(grand_prix_embeddings)

        return df_final

class LapNumberFeatureEngineeringPipeline:
    def __init__(self, df):
        """
        Initialize the pipeline with the input dataframe.
        Args:
        df (pd.DataFrame): The input dataframe containing F1 data.
        """
        self.df = df
        self.result_drivers_df = []

    @staticmethod
    def calculate_weighted_feature(df, feature, group_cols, method='mean'):
        """
        Generalized function to calculate the weighted cumulative mean or max for a given feature.
        
        Args:
        df (pd.DataFrame): The input dataframe.
        feature (str): The feature column to process.
        group_cols (list of str): Columns to group by.
        method (str): 'mean' or 'max' to apply to the feature.
        
        Returns:
        pd.DataFrame: A DataFrame with weighted cumulative mean/max results.
        """
        if method == 'mean':
            mean_per_year = df.groupby(['Year'] + group_cols, as_index=False)[feature].mean().reset_index()
            if (feature == "GridPosition") or (feature == 'Stint'): # Add Compound
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.expanding().mean())
            else:
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.shift(1).expanding().mean())

        elif method == 'max':
            mean_per_year = df.groupby(['Year'] + group_cols, as_index=False)[feature].max().reset_index()
            if (feature == "GridPosition") or (feature == 'Stint'): # Add Compound
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.expanding().mean())
            else:
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.shift(1).expanding().mean())

        return weighted.reset_index(drop=True)

    @staticmethod
    def generalized_weighted_mean_compound(group):
        """
        Function to compute the weighted average of compound embeddings.
        
        Args:
        group (pd.DataFrame): Grouped dataframe containing compound embeddings.
        
        Returns:
        pd.Series: A pandas Series containing the weighted average embeddings.
        """
        embeddings = np.array(group['CompoundEmbeddings'].tolist())
        compound_counts = group['Compound'].value_counts()

        weighted_sum = np.zeros(embeddings[0].shape)
        total_weight = 0

        for compound, count in compound_counts.items():
            compound_embeddings = embeddings[group['Compound'] == compound]
            weighted_sum += compound_embeddings.sum(axis=0)
            total_weight += count

        weighted_avg = weighted_sum / total_weight
        return pd.Series([list(weighted_avg)], index=['WeightedCompoundEmbeddings'])

    def process_event_and_driver(self, event, driver):
        """
        Process feature engineering for a single event and driver.
        
        Args:
        event (str): The event name.
        driver (str): The driver's name.
        """
        max_lap_event = self.df[self.df["EventName"] == event]["LapNumber"].max()
        df_driver = self.df[(self.df["EventName"] == event) & (self.df["Driver"] == driver)]

        if len(df_driver):
            # print(f"Processing driver: {driver} at event: {event}")
            
            result_df = df_driver[['Year', 'TeamOfficialName', 'EventName', 'Driver', 'Stint', 'Compound', 'CumulativePoints', 'LapNumber']].groupby(
                ['Year', 'EventName', 'TeamOfficialName', 'Driver', 'Stint'], as_index=False
            ).agg({"CumulativePoints": "last", "Compound":"last", 'LapNumber': 'max'})

            result_df['WeightedCumulativeMeanLapTimeSeconds'] = self.calculate_weighted_feature(df_driver, 'LapTimeSeconds', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanLapDeltaSeconds'] = self.calculate_weighted_feature(df_driver, 'LapDelta', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanFinalPosition'] = self.calculate_weighted_feature(df_driver, 'Position_result', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanLapPosition'] = self.calculate_weighted_feature(df_driver, 'Position_lap', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanGridPosition'] = self.calculate_weighted_feature(df_driver, 'GridPosition', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['GridPosition'] = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver', 'Stint'])['GridPosition'].max().values
            result_df['GridPositionEmbedding'] = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver', 'Stint'])['GridPositionEmbedding'].last().values
            result_df['WeightedCumulativeMeanSpeedI1'] = self.calculate_weighted_feature(df_driver, 'SpeedI1', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanSpeedI2'] = self.calculate_weighted_feature(df_driver, 'SpeedI2', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanSpeedFL'] = self.calculate_weighted_feature(df_driver, 'SpeedFL', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanSpeedST'] = self.calculate_weighted_feature(df_driver, 'SpeedST', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanMaxStint'] = self.calculate_weighted_feature(df_driver, 'Stint', ['TeamOfficialName', 'Driver', 'Stint'], method='max')

            result = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver', 'Stint'], as_index=False).apply(self.generalized_weighted_mean_compound)

            expanded_embeddings = pd.DataFrame(result['WeightedCompoundEmbeddings'].tolist(), columns=['WeightedCompoundEmbeddings_0', 'WeightedCompoundEmbeddings_1'])
            expanded_embeddings['WeightedCompoundEmbeddings_0'] = expanded_embeddings.groupby([result['TeamOfficialName'], result['Driver'], result['Stint']])['WeightedCompoundEmbeddings_0'].transform(lambda x: x.expanding().mean()) # .transform(lambda x: x.shift(1).expanding().mean())
            expanded_embeddings['WeightedCompoundEmbeddings_1'] = expanded_embeddings.groupby([result['TeamOfficialName'], result['Driver'], result['Stint']])['WeightedCompoundEmbeddings_1'].transform(lambda x: x.expanding().mean())

            result_df['WeightedCumulativeMeanCompoundEmbeddings'] = expanded_embeddings.apply(lambda row: [row['WeightedCompoundEmbeddings_0'], row['WeightedCompoundEmbeddings_1']], axis=1)
            result_df['NumberOfRacesOnStint'] = result_df.groupby(['Driver', 'Stint']).cumcount()
            
            race_number_df = result_df.drop_duplicates(["Year", "Driver", "TeamOfficialName", "EventName"]).groupby(['Driver'], as_index=False).cumcount()
            race_number_df = race_number_df.reset_index(drop=True).to_frame(name='RaceNumber')
            race_number_df = pd.concat([result_df.groupby(["Year", "Driver", "TeamOfficialName", "EventName"], as_index=False).count()[["Year", "Driver", "TeamOfficialName", "EventName"]], race_number_df], axis=1)
            result_df = pd.merge(result_df, race_number_df, on=["Year", "Driver", "TeamOfficialName", "EventName"])
            
            stint_number_df = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver'], as_index=False)['Stint'].max()
            result_df = pd.merge(result_df, stint_number_df, on=['Year', 'TeamOfficialName', 'Driver'], suffixes=('', '_max_stint'))
            result_df['StintNumber'] = result_df['Stint_max_stint']
            
            tyre_compounds_embeddings_tmp = {
                "HARD": [0.9592574834823608, 0.2825334072113037],
                "MEDIUM": [0.9999638795852661, 0.008497972041368484],
                "SOFT": [0.982682466506958, 0.1852976679801941],
                "SUPERSOFT": [0.9986254572868347, -0.05241385102272034],
                "HYPERSOFT": [-0.8185507655143738, 0.5744342803955078],
                "ULTRASOFT": [0.9489226341247559, 0.31550878286361694],
                "INTERMEDIATE": [0.8133333325386047, 0.5817979574203491],
                "WET": [0.5332446694374084, 0.8459610342979431],
                "nan": [0.0, 0.0],
                "NaN": [0.0, 0.0],
            }
            
            result_df['Compound'] = result_df['Compound'].astype(str)
            result_df['CompoundEmbedding'] = result_df['Compound'].map(tyre_compounds_embeddings_tmp)

            result_df['LapsByStint'] = 0

            # Calculate the number of laps by stint
            for (year, driver, event), group in result_df.groupby(['Year', 'Driver', 'EventName']):
                group = group.sort_values(by='Stint')  # Ensure stints are in correct order
                previous_lap = 0  # Set initial previous lap for the first stint in the event
                lap_by_stint = []
                
                for index, row in group.iterrows():
                    laps_in_stint = row['LapNumber'] - previous_lap  # Calculate the number of laps in the current stint
                    lap_by_stint.append(laps_in_stint)
                    previous_lap = row['LapNumber']  # Update the previous lap for the next stint
                
                result_df.loc[group.index, 'LapsByStint'] = lap_by_stint  # Assign calculated laps to the corresponding rows

            result_df['ShiftLapNumber'] = result_df.groupby(['Year', 'TeamOfficialName', 'Driver'])['LapNumber'].shift(1)
            result_df['ShiftLapNumberByStint'] = result_df.groupby(['Year', 'TeamOfficialName', 'Driver'])['LapsByStint'].shift(1)

            result_df['ShiftLapNumber'] = result_df['ShiftLapNumber'].fillna(0)
            result_df['ShiftLapNumberByStint'] = result_df['ShiftLapNumberByStint'].fillna(0)
            result_df['ShiftLapNumberToGo'] = max_lap_event - result_df['ShiftLapNumber']

            result_df["MaxNumber"] = max_lap_event

            self.result_drivers_df.append(result_df)

    def run(self):
        """
        Execute the feature engineering process for all events and drivers.
        """
        drivers = self.df["Driver"].unique().tolist()
        events = self.df["EventName"].unique().tolist()

        for event in events:
            for driver in drivers:
                try:
                    self.process_event_and_driver(event, driver)
                except:
                    print(driver, event)

        df_final = pd.concat(self.result_drivers_df).reset_index(drop=True)
        df_final = df_final.dropna()

        df_final['TeamEmbeddings'] = df_final['TeamOfficialName'].map(f1_teams_embeddings)
        df_final['DriverEmbeddings'] = df_final['Driver'].map(f1_drivers_embeddings)
        df_final['EventNameEmbeddings'] = df_final['EventName'].map(grand_prix_embeddings)

        return df_final

class LapTimeFeatureEngineeringPipeline:
    def __init__(self, df):
        """
        Initialize the pipeline with the input dataframe.
        Args:
        df (pd.DataFrame): The input dataframe containing F1 data.
        """
        self.df = df
        self.result_drivers_df = []

    @staticmethod
    def calculate_weighted_feature(df, feature, group_cols, method='mean'):
        """
        Generalized function to calculate the weighted cumulative mean or max for a given feature.
        
        Args:
        df (pd.DataFrame): The input dataframe.
        feature (str): The feature column to process.
        group_cols (list of str): Columns to group by.
        method (str): 'mean' or 'max' to apply to the feature.
        
        Returns:
        pd.DataFrame: A DataFrame with weighted cumulative mean/max results.
        """
        if method == 'mean':
            mean_per_year = df.groupby(['Year'] + group_cols, as_index=False)[feature].mean().reset_index()
            if (feature == "GridPosition") or (feature == 'Stint'): # Add Compound
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.expanding().mean())
            else:
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.shift(1).expanding().mean())

        elif method == 'max':
            mean_per_year = df.groupby(['Year'] + group_cols, as_index=False)[feature].max().reset_index()
            if (feature == "GridPosition") or (feature == 'Stint'): # Add Compound
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.expanding().mean())
            else:
                weighted = mean_per_year.groupby(group_cols)[feature].transform(lambda x: x.shift(1).expanding().mean())

        return weighted.reset_index(drop=True)

    @staticmethod
    def generalized_weighted_mean_compound(group):
        """
        Function to compute the weighted average of compound embeddings.
        
        Args:
        group (pd.DataFrame): Grouped dataframe containing compound embeddings.
        
        Returns:
        pd.Series: A pandas Series containing the weighted average embeddings.
        """
        embeddings = np.array(group['CompoundEmbeddings'].tolist())
        compound_counts = group['Compound'].value_counts()

        weighted_sum = np.zeros(embeddings[0].shape)
        total_weight = 0

        for compound, count in compound_counts.items():
            compound_embeddings = embeddings[group['Compound'] == compound]
            weighted_sum += compound_embeddings.sum(axis=0)
            total_weight += count

        weighted_avg = weighted_sum / total_weight
        return pd.Series([list(weighted_avg)], index=['WeightedCompoundEmbeddings'])

    def process_event_and_driver(self, event, driver):
        """
        Process feature engineering for a single event and driver.
        
        Args:
        event (str): The event name.
        driver (str): The driver's name.
        """
        max_lap_event = self.df[self.df["EventName"] == event]["LapNumber"].max()
        df_driver = self.df[(self.df["EventName"] == event) & (self.df["Driver"] == driver)]

        if len(df_driver):
            
            result_df = df_driver[['Year', 'TeamOfficialName', 'EventName', 'Driver', 'Stint', 'Compound', 'CumulativePoints', 'LapNumber', 'LapTimeSeconds']].groupby(
                ['Year', 'EventName', 'TeamOfficialName', 'Driver', 'Stint'], as_index=False
            ).agg({"CumulativePoints": "last", "Compound":"last", 'LapNumber': 'max', 'LapTimeSeconds':'mean'})

            result_df['WeightedCumulativeMeanLapTimeSeconds'] = self.calculate_weighted_feature(df_driver, 'LapTimeSeconds', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanLapDeltaSeconds'] = self.calculate_weighted_feature(df_driver, 'LapDelta', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanFinalPosition'] = self.calculate_weighted_feature(df_driver, 'Position_result', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanLapPosition'] = self.calculate_weighted_feature(df_driver, 'Position_lap', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanGridPosition'] = self.calculate_weighted_feature(df_driver, 'GridPosition', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['GridPosition'] = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver', 'Stint'])['GridPosition'].max().values
            result_df['GridPositionEmbedding'] = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver', 'Stint'])['GridPositionEmbedding'].last().values
            result_df['WeightedCumulativeMeanSpeedI1'] = self.calculate_weighted_feature(df_driver, 'SpeedI1', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanSpeedI2'] = self.calculate_weighted_feature(df_driver, 'SpeedI2', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanSpeedFL'] = self.calculate_weighted_feature(df_driver, 'SpeedFL', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanSpeedST'] = self.calculate_weighted_feature(df_driver, 'SpeedST', ['TeamOfficialName', 'Driver', 'Stint'], method='mean')
            result_df['WeightedCumulativeMeanMaxStint'] = self.calculate_weighted_feature(df_driver, 'Stint', ['TeamOfficialName', 'Driver', 'Stint'], method='max')

            result = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver', 'Stint'], as_index=False).apply(self.generalized_weighted_mean_compound)

            expanded_embeddings = pd.DataFrame(result['WeightedCompoundEmbeddings'].tolist(), columns=['WeightedCompoundEmbeddings_0', 'WeightedCompoundEmbeddings_1'])
            expanded_embeddings['WeightedCompoundEmbeddings_0'] = expanded_embeddings.groupby([result['TeamOfficialName'], result['Driver'], result['Stint']])['WeightedCompoundEmbeddings_0'].transform(lambda x: x.expanding().mean())
            expanded_embeddings['WeightedCompoundEmbeddings_1'] = expanded_embeddings.groupby([result['TeamOfficialName'], result['Driver'], result['Stint']])['WeightedCompoundEmbeddings_1'].transform(lambda x: x.expanding().mean())

            result_df['WeightedCumulativeMeanCompoundEmbeddings'] = expanded_embeddings.apply(lambda row: [row['WeightedCompoundEmbeddings_0'], row['WeightedCompoundEmbeddings_1']], axis=1)
            result_df['NumberOfRacesOnStint'] = result_df.groupby(['Driver', 'Stint']).cumcount()
            
            race_number_df = result_df.drop_duplicates(["Year", "Driver", "TeamOfficialName", "EventName"]).groupby(['Driver'], as_index=False).cumcount()
            race_number_df = race_number_df.reset_index(drop=True).to_frame(name='RaceNumber')
            race_number_df = pd.concat([result_df.groupby(["Year", "Driver", "TeamOfficialName", "EventName"], as_index=False).count()[["Year", "Driver", "TeamOfficialName", "EventName"]], race_number_df], axis=1)
            result_df = pd.merge(result_df, race_number_df, on=["Year", "Driver", "TeamOfficialName", "EventName"])
            
            stint_number_df = df_driver.groupby(['Year', 'TeamOfficialName', 'Driver'], as_index=False)['Stint'].max()
            result_df = pd.merge(result_df, stint_number_df, on=['Year', 'TeamOfficialName', 'Driver'], suffixes=('', '_max_stint'))
            result_df['StintNumber'] = result_df['Stint_max_stint']
            
            tyre_compounds_embeddings_tmp = {
                "HARD": [0.9592574834823608, 0.2825334072113037],
                "MEDIUM": [0.9999638795852661, 0.008497972041368484],
                "SOFT": [0.982682466506958, 0.1852976679801941],
                "SUPERSOFT": [0.9986254572868347, -0.05241385102272034],
                "HYPERSOFT": [-0.8185507655143738, 0.5744342803955078],
                "ULTRASOFT": [0.9489226341247559, 0.31550878286361694],
                "INTERMEDIATE": [0.8133333325386047, 0.5817979574203491],
                "WET": [0.5332446694374084, 0.8459610342979431],
                "nan": [0.0, 0.0],
                "NaN": [0.0, 0.0],
            }
            
            result_df['Compound'] = result_df['Compound'].astype(str)
            result_df['CompoundEmbedding'] = result_df['Compound'].map(tyre_compounds_embeddings_tmp)

            result_df['LapsByStint'] = 0

            # Calculate the number of laps by stint
            for (year, driver, event), group in result_df.groupby(['Year', 'Driver', 'EventName']):
                group = group.sort_values(by='Stint')  # Ensure stints are in correct order
                previous_lap = 0  # Set initial previous lap for the first stint in the event
                lap_by_stint = []
                
                for index, row in group.iterrows():
                    laps_in_stint = row['LapNumber'] - previous_lap  # Calculate the number of laps in the current stint
                    lap_by_stint.append(laps_in_stint)
                    previous_lap = row['LapNumber']  # Update the previous lap for the next stint
                
                result_df.loc[group.index, 'LapsByStint'] = lap_by_stint  # Assign calculated laps to the corresponding rows

            result_df["LapNumberToGo"] = max_lap_event - result_df["LapNumber"]
            result_df["MaxNumber"] = max_lap_event

            self.result_drivers_df.append(result_df)

    def run(self):
        """
        Execute the feature engineering process for all events and drivers.
        """
        drivers = self.df["Driver"].unique().tolist()
        events = self.df["EventName"].unique().tolist()

        for event in events:
            for driver in drivers:
                try:
                    self.process_event_and_driver(event, driver)
                except:
                    print(driver, event)

        df_final = pd.concat(self.result_drivers_df).reset_index(drop=True)
        df_final = df_final.dropna()

        df_final['TeamEmbeddings'] = df_final['TeamOfficialName'].map(f1_teams_embeddings)
        df_final['DriverEmbeddings'] = df_final['Driver'].map(f1_drivers_embeddings)
        df_final['EventNameEmbeddings'] = df_final['EventName'].map(grand_prix_embeddings)

        return df_final