# README
Code repository of article `A robotic sensory system with high spatiotemporal resolution for texture recognitiona`.

### 1. `data_preprocess_evaluation.py`

#### Overview
This Python file serves as the backbone for data ingestion and preprocessing, as well as implementing machine learning algorithms for texture classification.

#### Dependencies
- numpy
- pandas
- scikit-learn
- tsfresh

#### Methods and Classes

- `load_data(filepath: str) -> np.ndarray`
  - Description: Manages the data import from the specified file path.
  - Input: File path as a string.
  - Output: Data as a numpy array.

- `preprocess(data: np.ndarray) -> np.ndarray`
  - Description: Conducts data preprocessing including normalization and outlier removal.
  - Input: Raw data as a numpy array.
  - Output: Processed data as a numpy array.

- `Classifier`
  - Description: A class defining the structure of the texture classification algorithm.
  - Methods: 
    - `train()`: Trains the classifier.
    - `predict(data: np.ndarray) -> str`: Makes a prediction based on input data.

---

### 2. `GUI_demo.py`

#### Overview
This Python file is focused on real-time data visualization and signal classification. It also provides GUI elements for better user interaction.

#### Dependencies
- numpy
- matplotlib
- os
- serial

#### Methods and Classes

- `update_data()`
  - Description: Real-time data visualization logic. Updates the plot and GUI elements.
  - Dependencies: Requires a matplotlib plot and tkinter window to be initialized.

- `signal_classification()`
  - Description: Executes the classification logic based on the input signal and updates the GUI.
  - Output: Predicted label and confidence percentage are displayed on the GUI.

- `random_start()`
  - Description: Initializes plot with random signals for demonstration purposes.
  - Output: Updates plot and GUI elements with random data.
  
---

### 3. `wavelet_transformation.m`

#### Overview
The Matlab script primarily focuses on data import, signal processing, and wavelet-based time-frequency analysis. It also conducts high-pass filtering to remove noise and performs data visualization.

#### Dependencies
- Matlab Signal Processing Toolbox
- Excel, for data export

#### Functionality Breakdown

1. **Data Import**: Reads time and amplitude from a text file.
2. **Data Preprocessing**: Prepares data for FFT analysis.
3. **Fourier Analysis**: Executes FFT and visualizes amplitude spectrum.
4. **Filter Design**: Creates a Butterworth high-pass filter.
5. **Filter Application**: Applies the designed filter to the signal.
6. **WSST Analysis**: Performs Wavelet Synchrosqueezed Transform for time-frequency representation.
7. **Data Visualization**: Plots processed data and WSST results.
8. **Data Export**: Saves processed data to an Excel file.

#### Usage
The script can be run in a Matlab environment. Make sure to place the '10um.txt' file in the specified directory path. To export data, ensure the target directory for the Excel file exists or modify the script to fit your directory structure.
