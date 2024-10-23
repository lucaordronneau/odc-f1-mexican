from f1_mexican.data.lap_data_loader import LapDataLoader
from f1_mexican.data.result_data_loader import ResultDataLoader
from f1_mexican.preprocessing.preprocess_pipeline import PreprocessPipeline

from f1_mexican.constants import unique_grand_prix_names

from f1_mexican.preprocessing.feature_pipeline import StintNumberFeatureEngineeringPipeline
from f1_mexican.modeling.stint_number_modeling import StintNumberModelTrainingPipeline
from f1_mexican.inference.stint_number_inference_pipeline import StintNumberInferencePipeline

from f1_mexican.preprocessing.feature_pipeline import TireCompoundFeatureEngineeringPipeline
from f1_mexican.modeling.tire_compound_modeling import TireCompoundModelTrainingPipeline
from f1_mexican.inference.tire_compound_inference_pipeline import TireCompoundInferencePipeline

from f1_mexican.preprocessing.feature_pipeline import LapNumberFeatureEngineeringPipeline
from f1_mexican.modeling.lap_number_modeling import LapNumberModelTrainingPipeline
from f1_mexican.inference.lap_number_inference_pipeline import LapNumberInferencePipeline

from f1_mexican.preprocessing.feature_pipeline import LapTimeFeatureEngineeringPipeline
from f1_mexican.modeling.lap_time_modeling import LapTimeModelTrainingPipeline
from f1_mexican.inference.lap_time_inference_pipeline import LapTimeInferencePipeline
 
import math

train = True

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

if train:
    stint_number_feature_pipeline = StintNumberFeatureEngineeringPipeline(preprocessed_data)
    final_data = stint_number_feature_pipeline.run()
    print(len(final_data))

    training_columns = [
        'Year',
        'CumulativePoints',
        'WeightedCumulativeMeanLapTimeSeconds',
        'WeightedCumulativeMeanPitTimeSeconds',
        'WeightedCumulativeMeanLapDeltaSeconds',
        'WeightedCumulativeMeanFinalPosition',
        'WeightedCumulativeMeanLapPosition',
        'WeightedCumulativeMeanGridPosition',
        'GridPosition',
        'GridPositionEmbedding',
        'WeightedCumulativeMeanSpeedI1',
        'WeightedCumulativeMeanSpeedI2',
        'WeightedCumulativeMeanSpeedFL',
        'WeightedCumulativeMeanSpeedST',
        'WeightedCumulativeMeanMaxStint',
        'WeightedCumulativeMeanCompoundEmbeddings',
        'NumberOfRaces',
        'StintNumber',
        'TeamEmbeddings',
        'DriverEmbeddings',
        'EventNameEmbeddings',
        'MaxNumber'
    ]

    training_pipeline = StintNumberModelTrainingPipeline(final_data[training_columns])
    training_pipeline.run()

if train:
    tire_compound_feature_pipeline = TireCompoundFeatureEngineeringPipeline(preprocessed_data)
    final_data = tire_compound_feature_pipeline.run()
    print(len(final_data))

    training_columns = [
        'Year',
        'CumulativePoints',
        'WeightedCumulativeMeanLapTimeSeconds',
        'WeightedCumulativeMeanLapDeltaSeconds',
        'WeightedCumulativeMeanFinalPosition',
        'WeightedCumulativeMeanLapPosition',
        'WeightedCumulativeMeanGridPosition',
        'GridPosition',
        'GridPositionEmbedding',
        'WeightedCumulativeMeanSpeedI1',
        'WeightedCumulativeMeanSpeedI2',
        'WeightedCumulativeMeanSpeedFL',
        'WeightedCumulativeMeanSpeedST',
        'WeightedCumulativeMeanCompoundEmbeddings',
        'NumberOfRacesOnStint',
        'RaceNumber',
        'Stint',
        'StintNumber',
        'TeamEmbeddings',
        'DriverEmbeddings',
        'EventNameEmbeddings',
        'ShiftedCompoundEmbedding',
        'Compound',
        'MaxNumber'
    ]

    training_pipeline = TireCompoundModelTrainingPipeline(final_data[training_columns])
    training_pipeline.run()

if train:
    feature_pipeline = LapNumberFeatureEngineeringPipeline(preprocessed_data)
    final_data = feature_pipeline.run()
    print(len(final_data))

    training_columns = [
        'Year',
        'CumulativePoints',
        'WeightedCumulativeMeanLapTimeSeconds',
        'WeightedCumulativeMeanLapDeltaSeconds',
        'WeightedCumulativeMeanFinalPosition',
        'WeightedCumulativeMeanLapPosition',
        'WeightedCumulativeMeanGridPosition',
        'GridPosition',
        'GridPositionEmbedding',
        'WeightedCumulativeMeanSpeedI1',
        'WeightedCumulativeMeanSpeedI2',
        'WeightedCumulativeMeanSpeedFL',
        'WeightedCumulativeMeanSpeedST',
        'WeightedCumulativeMeanCompoundEmbeddings',
        'NumberOfRacesOnStint',
        'RaceNumber',
        'Stint',
        'StintNumber',
        'TeamEmbeddings',
        'DriverEmbeddings',
        'EventNameEmbeddings',
        'CompoundEmbedding',
        'ShiftLapNumber',
        'ShiftLapNumberByStint',
        'ShiftLapNumberToGo',
        'LapsByStint',
        'MaxNumber'
    ]

    training_pipeline = LapNumberModelTrainingPipeline(final_data[training_columns])
    training_pipeline.run()

if train:
    feature_pipeline = LapTimeFeatureEngineeringPipeline(preprocessed_data)
    final_data = feature_pipeline.run()
    print(len(final_data))

    training_columns = [
        'Year',
        'CumulativePoints',
        'WeightedCumulativeMeanLapTimeSeconds',
        'WeightedCumulativeMeanLapDeltaSeconds',
        'WeightedCumulativeMeanFinalPosition',
        'WeightedCumulativeMeanLapPosition',
        'WeightedCumulativeMeanGridPosition',
        'GridPosition',
        'GridPositionEmbedding',
        'WeightedCumulativeMeanSpeedI1',
        'WeightedCumulativeMeanSpeedI2',
        'WeightedCumulativeMeanSpeedFL',
        'WeightedCumulativeMeanSpeedST',
        'WeightedCumulativeMeanCompoundEmbeddings',
        'NumberOfRacesOnStint',
        'RaceNumber',
        'Stint',
        'StintNumber',
        'TeamEmbeddings',
        'DriverEmbeddings',
        'EventNameEmbeddings',
        'CompoundEmbedding',
        'LapNumberToGo',
        'LapsByStint',
        'LapNumber',
        'LapTimeSeconds',
        'MaxNumber'
    ]

    training_pipeline = LapTimeModelTrainingPipeline(final_data[training_columns])
    training_pipeline.run()