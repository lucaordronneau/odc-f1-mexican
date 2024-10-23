drivers_input = ['LEC',
 'SAI',
 'VER',
 'NOR',
 'PIA',
 'RUS',
 'PER',
 'HUL',
 'LAW',
 'COL',
 'MAG',
 'GAS',
 'ALO',
 'TSU',
 'STR',
 'ALB',
 'BOT',
 'OCO',
 'ZHO',
 'HAM']

grid_positions_input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

import json
from f1_mexican.data.lap_data_loader import LapDataLoader
from f1_mexican.data.result_data_loader import ResultDataLoader
from f1_mexican.preprocessing.preprocess_pipeline import PreprocessPipeline
from f1_mexican.constants import unique_grand_prix_names

from f1_mexican.preprocessing.feature_pipeline import (
    StintNumberFeatureEngineeringPipeline,
    TireCompoundFeatureEngineeringPipeline,
    LapNumberFeatureEngineeringPipeline,
    LapTimeFeatureEngineeringPipeline,
)
from f1_mexican.modeling.stint_number_modeling import StintNumberModelTrainingPipeline
from f1_mexican.modeling.tire_compound_modeling import TireCompoundModelTrainingPipeline
from f1_mexican.modeling.lap_number_modeling import LapNumberModelTrainingPipeline
from f1_mexican.modeling.lap_time_modeling import LapTimeModelTrainingPipeline

from f1_mexican.inference.stint_number_inference_pipeline import StintNumberInferencePipeline
from f1_mexican.inference.tire_compound_inference_pipeline import TireCompoundInferencePipeline
from f1_mexican.inference.lap_number_inference_pipeline import LapNumberInferencePipeline
from f1_mexican.inference.lap_time_inference_pipeline import LapTimeInferencePipeline

train = False

# Define the driver and grid position details
# drivers_input = ['VER']  # User input: could be a list or a single driver
# grid_positions_input = [12]  # User input: could be a list or a single position
event = 'Mexican Grand Prix'  # Event
year = 2024  # Current year

# Handle single input or list input for drivers and grid positions
drivers = drivers_input if isinstance(drivers_input, list) else [drivers_input]
grid_positions = grid_positions_input if isinstance(grid_positions_input, list) else [grid_positions_input]

final_result = {}  # Dictionary to store results

# Main loop to process each driver and grid position
for driver in drivers:
    final_result[driver] = []
    for grid_position in grid_positions:

        try:
            # Load the data
            lap_data_loader = LapDataLoader('data/lap_times_2018_to_2023.csv')
            lap_data = lap_data_loader.load()

            result_data_loader = ResultDataLoader('data/race_results_2018_to_2023.csv')
            result_data = result_data_loader.load()

            # Reverse the dictionary: map circuit names to their corresponding Grand Prix
            circuit_to_grand_prix = {circuit: gp for gp, circuits in unique_grand_prix_names.items() for circuit in circuits}

            # Run the preprocessing pipeline
            pipeline = PreprocessPipeline(lap_data, result_data, circuit_to_grand_prix)
            preprocessed_data = pipeline.run()

            lapnumber_max = preprocessed_data[preprocessed_data["EventName"] == event]['LapNumber'].max()
            print("Lap Number Max: ", lapnumber_max)

            # Initialize the stint number inference pipeline
            model_path = 'final_stint_number_xgboost_model.pkl'  # Path to your stint number model
            stint_number_inference_pipeline = StintNumberInferencePipeline(preprocessed_data, model_path, event)
            predicted_stint_number = stint_number_inference_pipeline.predict_stint(driver, grid_position, lapnumber_max)
            if predicted_stint_number[0] >=1 and predicted_stint_number[0] <= 2.75:
                predicted_stint_number = 2
            elif predicted_stint_number[0] >2.75 and predicted_stint_number[0] <= 3.7:
                predicted_stint_number = 3
            # import math
            # predicted_stint_number = math.floor(predicted_stint_number[0])
    
            # predicted_stint_number = 2
            print(f"Predicted Stint Number for {driver} at {grid_position} grid position at {event} in {year}: {predicted_stint_number}")

            # Initialize the tire compound inference pipeline
            model_path = 'final_tire_compound_xgboost_model.pkl'  # Path to your tire compound model
            tire_compound_inference_pipeline = TireCompoundInferencePipeline(preprocessed_data, model_path, event)
            predicted_compounds = tire_compound_inference_pipeline.predict_compounds(driver, grid_position, predicted_stint_number, lapnumber_max)
            print(f"Predicted Compounds for {driver} at grid position {grid_position} at {event} in {year}:")
            for stint, compound in enumerate(predicted_compounds, start=1):
                print(f"Stint {stint}: {compound}")

            # Initialize the lap number inference pipeline
            model_path = 'final_lap_number_xgboost_model.pkl'  # Path to your lap number model
            lap_number_inference_pipeline = LapNumberInferencePipeline(preprocessed_data, model_path, event)
            predicted_lap_number = lap_number_inference_pipeline.predict_lap_number(driver, grid_position, predicted_stint_number, predicted_compounds, lapnumber_max)
            predicted_lap_number = [float(number) for number in predicted_lap_number]
            print(f"Predicted Lap Number for {driver} at {grid_position} grid position at {event} in {year}: {predicted_lap_number}")

            # Initialize the lap time inference pipeline
            model_path = 'final_lap_time_xgboost_model.pkl'  # Path to your lap time model
            lap_time_inference_pipeline = LapTimeInferencePipeline(preprocessed_data, model_path, event)
            predicted_lap_time = lap_time_inference_pipeline.predict_lap_time(driver, grid_position, predicted_stint_number, predicted_compounds, lapnumber_max, predicted_lap_number)
            predicted_lap_time = [float(time) for time in predicted_lap_time]
            print(f"Predicted Lap Time for {driver} at {grid_position} grid position at {event} in {year}: {predicted_lap_time}")

            # Store the result for this driver and grid position
            final_result[driver].append({
                'grid_position': grid_position,
                'stint_number': predicted_stint_number,
                'predicted_compounds': predicted_compounds,
                'predicted_lap_number': predicted_lap_number,
                'predicted_lap_time': predicted_lap_time
            })

        except Exception as e:
            print(f"An error occurred for driver {driver} at grid position {grid_position}: {str(e)}")
            continue  # Continue with the next driver or grid position

# Save the final results to a JSON file
file_path = "data.json"
with open(file_path, 'w') as json_file:
    json.dump(final_result, json_file, indent=4)

print(f"Results saved to {file_path}")