import argparse
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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="F1 Mexican GP Prediction Script")
parser.add_argument('--drivers_grid', nargs='+', required=True, help="List of drivers followed by their grid positions (e.g., LEC 1 VER 4 SAI 7)")
args = parser.parse_args()

# Extract drivers and grid positions from the input
if len(args.drivers_grid) % 2 != 0:
    raise ValueError("Please provide an equal number of drivers and grid positions")

drivers_input = args.drivers_grid[::2]  # Extract drivers (every second item)
grid_positions_input = list(map(int, args.drivers_grid[1::2]))  # Extract grid positions (as integers)

train = False

# Define the event and year (hard-coded)
event = 'Mexican Grand Prix'
year = 2024

final_result = {}

# Main loop to process each driver and their respective grid position
for driver, grid_position in zip(drivers_input, grid_positions_input):
    final_result[driver] = []
    
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
        model_path = 'models/final_stint_number_xgboost_model.pkl'
        stint_number_inference_pipeline = StintNumberInferencePipeline(preprocessed_data, model_path, event)
        predicted_stint_number = stint_number_inference_pipeline.predict_stint(driver, grid_position, lapnumber_max)
        if predicted_stint_number[0] >= 1 and predicted_stint_number[0] <= 2.75:
            predicted_stint_number = 2
        elif predicted_stint_number[0] > 2.75 and predicted_stint_number[0] <= 3.7:
            predicted_stint_number = 3

        print(f"Predicted Stint Number for {driver} at {grid_position} grid position at {event} in {year}: {predicted_stint_number}")

        # Initialize the tire compound inference pipeline
        model_path = 'models/final_tire_compound_xgboost_model.pkl'
        tire_compound_inference_pipeline = TireCompoundInferencePipeline(preprocessed_data, model_path, event)
        predicted_compounds = tire_compound_inference_pipeline.predict_compounds(driver, grid_position, predicted_stint_number, lapnumber_max)
        print(f"Predicted Compounds for {driver} at grid position {grid_position} at {event} in {year}:")
        for stint, compound in enumerate(predicted_compounds, start=1):
            print(f"Stint {stint}: {compound}")

        # Initialize the lap number inference pipeline
        model_path = 'models/final_lap_number_xgboost_model.pkl'
        lap_number_inference_pipeline = LapNumberInferencePipeline(preprocessed_data, model_path, event)
        predicted_lap_number = lap_number_inference_pipeline.predict_lap_number(driver, grid_position, predicted_stint_number, predicted_compounds, lapnumber_max)
        predicted_lap_number = [float(number) for number in predicted_lap_number]
        print(f"Predicted Lap Number for {driver} at {grid_position} grid position at {event} in {year}: {predicted_lap_number}")

        # Initialize the lap time inference pipeline
        model_path = 'models/final_lap_time_xgboost_model.pkl'
        lap_time_inference_pipeline = LapTimeInferencePipeline(preprocessed_data, model_path, event)
        predicted_lap_time = lap_time_inference_pipeline.predict_lap_time(driver, grid_position, predicted_stint_number, predicted_compounds, lapnumber_max, predicted_lap_number)
        predicted_lap_time = [float(time) for time in predicted_lap_time]
        print(f"Predicted Lap Time for {driver} at {grid_position} grid position at {event} in {year}: {predicted_lap_time}")

        final_result[driver].append({
            'grid_position': grid_position,
            'stint_number': predicted_stint_number,
            'predicted_compounds': predicted_compounds,
            'predicted_lap_number': predicted_lap_number,
            'predicted_lap_time': predicted_lap_time
        })

    except Exception as e:
        print(f"An error occurred for driver {driver} at grid position {grid_position}: {str(e)}")
        continue

# Save the final results to a JSON file
file_path = "results.json"
with open(file_path, 'w') as json_file:
    json.dump(final_result, json_file, indent=4)

print(f"Results saved to {file_path}")