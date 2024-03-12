import numpy as np
import csv
import matplotlib.pyplot as plt

#################################
### DATA PROCESSING FUNCTIONS ###
#################################

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

def raw_data_processing(input_file, output_file, save_file = False):
    # Function processes data of 3 trials for a single participant at a single bag placement

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

    # Creating an average column:
    column_average = []
    for i in range(len(column1_processed)):
        column_average.append((column1_processed[i]+column2_processed[i]+column3_processed[i])/3)

    # Write processed data to the output CSV file, if needed
    if save_file == True:
        with open(output_file, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            for values in zip(column1_processed, column2_processed, column3_processed, column_average):
                csv_writer.writerow(values)

    return column1_processed, column2_processed, column3_processed, column_average


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

raw_data_processing('/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_raw/01_ST_LOW_TRUNK.csv', 
           '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_interpolated/01_ST_LOW_TRUNK_INTERPOLATED.csv', True)

