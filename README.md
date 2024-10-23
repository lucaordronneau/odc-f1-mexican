
# F1 Mexican Grand Prix Prediction

This project predicts stint numbers, tire compounds, lap numbers, and lap times for drivers during the Mexican Grand Prix. It uses machine learning models for inference based on historical F1 data from 2018 to 2023.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Examples](#examples)

## Requirements
- Python 3.8+
- Poetry (for dependency management)

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install dependencies using Poetry:**

   Make sure you have Poetry installed. If not, install Poetry using the following command:

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   or

   ```bash
   pip install poetry
   ```

   Then, install the project dependencies:

   ```bash
   poetry install
   ```

3. **Activate the Poetry environment:**

   Once dependencies are installed, activate the Poetry shell:

   ```bash
   poetry shell
   ```

## Usage

### Running the Prediction Script

You can run the prediction script by passing the drivers and their grid positions via command-line arguments. The event (Mexican Grand Prix) and year (2024) are hard-coded.

### Command to Run:

```bash
python f1_mexican/launch.py --drivers_grid <DRIVER1> <GRID_POSITION1> <DRIVER2> <GRID_POSITION2> ...
```

For example, to predict for Charles Leclerc (LEC) starting at grid position 1, Max Verstappen (VER) at grid position 4, and Carlos Sainz (SAI) at grid position 7, run:

```bash
python f1_mexican/launch.py --drivers_grid LEC 1 VER 4 SAI 7
```

### Output

The predictions will be printed to the console, and the results will be saved to a JSON file named `resuslts.json` in the project directory.

```json
{
    "VER": [
        {
            "grid_position": 1,
            "stint_number": 2,
            "predicted_compounds": [
                "MEDIUM",
                "HARD"
            ],
            "predicted_lap_number": [
                24.0,
                47.0
            ],
            "predicted_lap_time": [
                82.92504119873047,
                81.77375030517578
            ]
        }
    ],
    "LEC": [
        {
            "grid_position": 2,
            "stint_number": 2,
            "predicted_compounds": [
                "MEDIUM",
                "HARD"
            ],
            "predicted_lap_number": [
                25.0,
                46.0
            ],
            "predicted_lap_time": [
                83.66840362548828,
                82.06167602539062
            ]
        }
    ]
}
```

## File Structure

- `data/`: Contains the historical F1 data required for the models.
  - `lap_times_2018_to_2023.csv`: Lap time data.
  - `race_results_2018_to_2023.csv`: Race result data.
- `f1_mexican/`: Contains the core modules for data loading, preprocessing, modeling, and inference.
- `models/`: Contains the pre-trained models used for inference.
  - `final_stint_number_xgboost_model.pkl`
  - `final_tire_compound_xgboost_model.pkl`
  - `final_lap_number_xgboost_model.pkl`
  - `final_lap_time_xgboost_model.pkl`
- `launch.py`: Main script for running predictions.
- `train.py`: Main script for training.

### Training

```bash
python f1_mexican/train.py
```

## Examples

### Example 1: Predict for a Single Driver

```bash
python f1_mexican/launch.py --drivers_grid LEC 1
```

### Example 2: Predict for Multiple Drivers

```bash
python f1_mexican/launch.py --drivers_grid LEC 1 VER 4 SAI 7
```

## License
This project is licensed under the MIT License.
