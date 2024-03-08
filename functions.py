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

def csv_to_list(file_path):
    # Simply converts a single-column csv to a 1-D list

    data_list = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header if it exists
        for row in csv_reader:
            for value in row:
                data_list.append(float(value))  # Convert to float if necessary
    return data_list

def list_to_csv(input_list, file_path):
    # Simply saves a single-column csv file, given a 1-D list

    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        for item in input_list:
            csv_writer.writerow([item])

def data_processing_single_column(input_file, output_file):
    data_list = csv_to_list(input_file)
    processed_data = interpolate_to_501_points(data_list)
    list_to_csv(processed_data, output_file)
    print('List written to CSV file: ', output_file)



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

