# Surveyor Gradient Chasing

## Overview
The `surveyor_gradient_chasing` project implements gradient chasing using water feature measurements. It simulates or controls an autonomous surface vehicle (ASV) to navigate based on environmental data, such as water quality metrics, using Gaussian Process modeling and gradient tracking.

---

## Project Structure

### **Folder Structure**
```
surveyor_gradient_chasing/
├── README.md                     # Project documentation
├── config.py                     # Configuration variables
├── gradient_tracking.py          # Main script for gradient chasing
├── test/
│   ├── mock_surveyor.py          # Mock implementation of the ASV
├── utils/
│   ├── __init__.py               # Package initialization
│   ├── wp_plotter.py             # Plotting utilities for waypoints and environment
│   ├── water_phenomenon.py       # Gaussian Process modeling for water features
│   ├── send_data_utils.py        # Utilities for sending data to MongoDB
├── surveyor_library/             # External surveyor library
│   ├── surveyor_lib/
│   │   ├── helpers.py            # Helper functions for surveyor operations
│   │   ├── surveyor.py           # Surveyor class for real ASV control
```

---

## Installation

### **Prerequisites**
- Python 3.11 or higher
- Required Python libraries:
  - `Cartopy`
  - `Flask`
  - `geopy`
  - `h5py`
  - `matplotlib`
  - `numpy`
  - `opencv-python`
  - `pandas`
  - `pillow`
  - `pygeomag`
  - `PyKrige`
  - `pymongo`
  - `pynmea2`
  - `pyserial`
  - `pytest`
  - `requests`
  - `requests-mock`
  - `scikit-learn`
  - `tqdm`

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/surveyor_gradient_chasing.git
   cd surveyor_gradient_chasing
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### **Command-Line Arguments**
Run the main script `gradient_tracking.py` with the following arguments:

```bash
python gradient_tracking.py <filename> <erp_filename> <extent_filename> [--mission_postfix]
```

#### **Arguments**
- `filename`: Path to the main data CSV file containing waypoints.
- `erp_filename`: Path to the ERP data CSV file.
- `extent_filename`: Path to the extent coordinates CSV file.
- `--mission_postfix`: Optional postfix for mission-specific data saving.

#### **Example**
```bash
python gradient_tracking.py waypoints.csv erp.csv extent.csv --mission_postfix "mission1"
```

---

## Features

### **Gradient Tracking**
- Uses Gaussian Process modeling to predict water feature gradients.
- Dynamically calculates the next waypoint based on the gradient.

### **Simulation Mode**
- Simulates ASV movement using the `MockSurveyor` class.

### **Real ASV Control**
- Interfaces with a real ASV using the `Surveyor` class.

### **Plotting**
- Visualizes the environment, waypoints, and ASV path using `matplotlib` and `cartopy`.

### **Data Handling**
- Saves collected data locally.
- Optionally sends data to MongoDB for storage.

---

## Modules

### **`gradient_tracking.py`**
The main script orchestrates the gradient chasing mission:
- Initializes the ASV (real or simulated).
- Processes waypoints and updates the Gaussian Process model.
- Visualizes the ASV path and environment.

### **`config.py`**
Contains global configuration variables:
- `THROTTLE`: Speed of the ASV.
- `FEATURE_TO_CHASE`: Water feature to optimize (e.g., "ODO (%Sat)").
- `NUM_WAYPOINTS`: Number of waypoints for gradient chasing.
- `STEP_SIZE`: Step size for waypoint calculation.

### **`utils/wp_plotter.py`**
Handles plotting of waypoints, environment, and ASV path:
- `plot_initialization`: Initializes the plot.
- `get_plot_arrow`: Creates an arrow to represent the ASV.

### **`utils/water_phenomenon.py`**
Implements Gaussian Process modeling for water features:
- `next_point`: Calculates the next waypoint based on the gradient.
- `_rotate_within_domain`: Ensures waypoints stay within the defined domain.

### **`test/mock_surveyor.py`**
Simulates ASV movement:
- `go_to_waypoint`: Moves the ASV to a specified waypoint.
- `station_keep`: Stops the ASV at its current location.

---

## Example Workflow

1. **Prepare Input Files**:
   - Create CSV files for waypoints, ERP data, and extent coordinates.

2. **Run the Script**:
   - Execute `gradient_tracking.py` with the prepared input files.

3. **Monitor Progress**:
   - View the ASV path and environment visualization in real-time.

4. **Save Data**:
   - Collected data is saved locally or sent to MongoDB.

---

## Configuration

### **Plot Settings**
Customize plot appearance in `config.py`:
```python
PATH_PLOT_ARGS = {"color": "black", "linestyle": "--"}
CONTOURF_ARGS = {"cmap": "viridis", "levels": 15}
```

---

## Contributing

### **Guidelines**
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description.

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or support, contact:
- **Name**: Jose Fuentes
- **Email**: jfuen099@fiu.edu
- **GitHub**: [Xioeng](https://github.com/xioeng)
