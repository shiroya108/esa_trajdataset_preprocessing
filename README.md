# Data preprocessing for IMU and Vicon data
Do preprocessing to IMU and Vicon data and package to NPZ format for training and testing dataset

## Prerequisites 
- Numpy
- Pandas
- Plotly
- Matplotlib
- Magwick AHRS https://github.com/morgil/madgwick_py
- Matlab

## Preprocessing
precrocessing.ipynb
1. Read IMU and Vicon files
2. Find Vicon tracker postitions
3. Find common quaterion of Vicon trackers
4. Run AHRS on IMU data
5. Align IMU and Vicon data
6. Save preprocessed data

## Packaging
npz_packaging.ipynb
1. Open folder containing preprocessed data (syn_folder)
2. Obtain a list of data folders
3. Copy the list and divide it into training set and testing set
4. Enter training list file and training dataset destination and start packaging training dataset, obtain training dataset file train.npz
5. Enter testing list file and testing dataset destination and start testing training dataset, obtain testing files starting from 0.npz
