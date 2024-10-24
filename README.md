
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
   git clone https://github.com/lucaordronneau/odc-f1-mexican.git
   cd odc-f1-mexican/
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

You can run the prediction script by passing the drivers and their grid positions via command-line arguments. Additionally, you can provide stint numbers, compounds, and lap numbers for each driver. If no stint numbers or compounds are provided, the script will infer them using pre-trained models.

### Command to Run:

```bash
python f1_mexican/main.py --drivers_grid <DRIVER1> <GRID_POSITION1> <DRIVER2> <GRID_POSITION2> ...
```

#### Optional Parameters:

- `--stint_numbers`: A list of stint numbers for each driver. Must match the number of drivers.
- `--compounds`: A list of compounds for each stint for each driver. Must be repeated for each driver. Valid compounds: `MEDIUM`, `HARD`, `SOFT`, `INTERMEDIATE`, `WET`.
- `--lap_numbers`: A list of lap numbers corresponding to each stint for each driver. Must be repeated for each driver.

### Examples:

1. **Basic Prediction with Grid Positions Only**:

   Predict stint numbers, compounds, and lap numbers for Charles Leclerc (LEC) at grid position 1, Max Verstappen (VER) at grid position 4, and Carlos Sainz (SAI) at grid position 7:

   ```bash
   python f1_mexican/main.py --drivers_grid LEC 1 VER 4 SAI 7
   ```

2. **Prediction with Custom Stint Numbers to evaluate the model independently**:

   Provide custom stint numbers (2 stints for LEC, 3 stints for VER, 2 stints for SAI):

   ```bash
   python f1_mexican/main.py --drivers_grid LEC 1 VER 4 SAI 7 --stint_numbers 2 3 2
   ```

3. **Prediction with custom stint numbers, compounds, and lap numbers to evaluate the model independently**:

   Specify custom compounds and lap numbers for each driver:

   ```bash
   python f1_mexican/main.py --drivers_grid LEC 1 VER 4 SAI 7    --stint_numbers 2 3 2    --compounds MEDIUM HARD --compounds SOFT HARD SOFT --compounds MEDIUM HARD    --lap_numbers 25 50 --lap_numbers 30 60 70 --lap_numbers 20 40
   ```

   This sets:
   - LEC with 2 stints (`MEDIUM`, `HARD`) and lap numbers 25 and 50.
   - VER with 3 stints (`SOFT`, `HARD`, `SOFT`) and lap numbers 30, 60, 70.
   - SAI with 2 stints (`MEDIUM`, `HARD`) and lap numbers 20 and 40.

### Output

The predictions will be printed to the console, and the results will be saved to a JSON file named `results.json` in the project directory.

Example output:

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
- `main.py`: Main script for running predictions.
- `train.py`: Main script for training.

### Training

You can train the models using the `train.py` script. Ensure you have the necessary training data in the `data/` folder.

```bash
python f1_mexican/train.py
```

## Examples

### Example 1: Predict for a Single Driver

```bash
python f1_mexican/main.py --drivers_grid LEC 1
```

### Example 2: Predict for Multiple Drivers with Custom Stints

```bash
python f1_mexican/main.py --drivers_grid LEC 1 VER 4 SAI 7 --stint_numbers 2 3 2
```

### Example 3: Predict for Multiple Drivers with Custom Compounds and Lap Numbers

```bash
python f1_mexican/main.py --drivers_grid LEC 1 VER 4 SAI 7 --stint_numbers 2 3 2 --compounds MEDIUM HARD --compounds SOFT HARD SOFT --compounds MEDIUM HARD --lap_numbers 25 50 --lap_numbers 30 60 70 --lap_numbers 20 40
```

## License
This project is licensed under the MIT License.
