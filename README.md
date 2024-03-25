# BN6202 Group 2 Data Processing

Note: You can set up a virtual environment, but not really needed I guess...

## Libraries Required
- numpy
- matplotlib

## How it Works
- Data of different lengths are all interpolated to maintain data, while providing a standardised number of datapoints to allow averaging of datapoints
- 

## Generating all interpolated datafiles
- Get the list of all csv files within the data_raw file
```
folder = '/bn6202-g02-dataprocessing/data_raw'
files_list = list_csv_files_in_folder(folder, True)
```
- Run data interpolation on all files within data_raw
```
raw_data_processing(files_list)
```
- Perform data transformation on ankle interpolated files
```
anklefolder = '/bn6202-g02-dataprocessing/data_interpolated'
ankle_files_list = list_csv_files_in_folder(folder, False, 'ANKLE', 'ANKLE')
ankle_process(ankle_files_list)
```


## To find the list of maximum or miniumum angles in a set of files
- Select the types of file that needs to be used, e.g. 'CONTROL' and 'KNEE'
```
folder = '/bn6202-g02-dataprocessing/data_interpolated'
files_list = list_csv_files_in_folder(folder, False, 'KNEE', 'CONTROL')
```
- Select the range of indexing, and type of search (min/max), and select the output source
```
minmax_angle_list(files_list, 200, 501, '/bn6202-g02-dataprocessing/processed/max_knee_swingphase_control.csv', 'max')
```
