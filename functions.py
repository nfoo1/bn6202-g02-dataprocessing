import numpy as np
import csv
import matplotlib.pyplot as plt
import os

#################################
### DATA PROCESSING FUNCTIONS ###
#################################


def list_csv_files_in_folder(folder_path):
    """
    Inputs:
    - folder_path: A string representing the path to the folder
    
    Outputs:
    - A list containing the pathnames of all CSV files within the folder
    """
    csv_file_paths = []  # Initialize an empty list to store CSV file paths
    
    # Walk through the directory tree rooted at folder_path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                csv_file_paths.append(os.path.join(root, file))  # Append the CSV file path to the list
    
    return csv_file_paths

def interpolate_to_501_points(input_list):
    # Use is to allow averaging of trials of different durations
    # 501 chosen as len([0.0, 0.2, 0.4, ..., 99.8, 100.0) == 500
    # Allows normalisation to 100% gait cycle

    # Convert input list to numpy array for easier manipulation
    input_array = np.array(input_list, dtype=float)

    # Find indices of non-empty values
    non_empty_indices = np.where(~np.isnan(input_array))[0]

    # Create interpolation function
    interpolation_func = np.interp

    # Generate indices for output
    output_indices = np.linspace(0, len(input_array) - 1, 501)

    # Interpolate values
    interpolated_values = interpolation_func(output_indices, non_empty_indices, input_array[non_empty_indices])

    return interpolated_values

def raw_data_processing(input_files):
    # Function processes data of 3 trials for a single participant at a single bag placement
    for input_file in input_files:
        output_file = os.path.splitext(input_file)[0] + '_INTERPOLATED.csv'

        # Lists to store the data from each column
        column1 = []
        column2 = []
        column3 = []

        # Read data from the input CSV file
        with open(input_file, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip the header if it exists
            for row in csv_reader:
                # Convert values from string to float if necessary
                value1 = float(row[0]) if row[0] else None
                value2 = float(row[1]) if row[1] else None
                value3 = float(row[2]) if row[2] else None

                # Append non-null values to respective lists
                if value1 is not None:
                    column1.append(value1)
                if value2 is not None:
                    column2.append(value2)
                if value3 is not None:
                    column3.append(value3)

        # Apply interpolation function to each list
        column1_processed = interpolate_to_501_points(column1)
        column2_processed = interpolate_to_501_points(column2)
        column3_processed = interpolate_to_501_points(column3)

        # Write processed data to the output CSV file
        with open(output_file, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            for values in zip(column1_processed, column2_processed, column3_processed):
                csv_writer.writerow(values)

        print(f"Processed data saved to: {output_file}")

def knee_angle_max_list(csv_files, output_file='max_values.csv'):
    max_list = []

    # Iterate through each CSV file
    for file_path in csv_files:
        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip header if present

            # Initialize maximum values for each column
            max_values = [-float('inf')] * 3

            # Iterate through each row in the CSV file
            for row in reader:
                # Extract data from each column
                for col_index in range(3):
                    # Update maximum value for each column
                    value = float(row[col_index])
                    if value > max_values[col_index]:
                        max_values[col_index] = value

            # Append maximum values for indices 201 to 501
            max_list.extend(max_values)

    # Save max_list as a CSV file with values in a single column
    with open(output_file, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(max_list)

    return max_list

##########################
### PLOTTING FUNCTIONS ###
##########################

# Notes:
# For line-plots, ensure x-axis scale is from 0% - 100%, x-axis label is 'Percentage of Gait Cycle'
#   - Trunk angle: y-axis flexion/extension, scale tbd
#   - Hip angle: y-axis flexion/extension, scale tbd
#   - Knee angle: y-axis flexion/extension, scale tbd
#   - Ankle angle: y-axis flexion/extension, scale tbd
# For bar-plots, TBDAOSICBCOISN

def individual_line_plot():
    pass





#####################
### WORKING SPACE ###
#####################

# raw_data_processing('/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_raw/01_ST_CONTROL_ANKLE.csv', 
#            '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_interpolated/01_ST_CONTROL_ANKLE_INTERPOLATED.csv', True)

# knee_list = ['/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_interpolated/01_ST_CONTROL_KNEE_INTERPOLATED.csv']

# knee_angle_max_list(knee_list, '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/processed/max_knee_angles_swingphase.csv')

folder = '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_raw'
files_list = list_csv_files_in_folder(folder)
# print(files_list)

raw_data_processing(files_list)