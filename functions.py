import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd

#################################
### DATA PROCESSING FUNCTIONS ###
#################################


def list_csv_files_in_folder(folder_path, include_all=True, keyword1=None, keyword2=None):
    """
    Inputs:
    - folder_path: A string representing the path to the folder
    - include_all: A boolean indicating whether to include all CSV files or filter by keywords (default: True)
    - keyword1: The first keyword to filter by (default: None)
    - keyword2: The second keyword to filter by (default: None)
    
    Outputs:
    - A list containing the pathnames of CSV files within the folder that match the specified criteria
    """
    csv_file_paths = []  # Initialize an empty list to store CSV file paths
    
    # Walk through the directory tree rooted at folder_path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                if include_all:
                    csv_file_paths.append(os.path.join(root, file))  # Append the CSV file path to the list
                elif keyword1 and keyword2 and (keyword1 in file) and (keyword2 in file):
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
    # Define the output folder path
    output_folder = '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_interpolated'

    # Function processes data of 3 trials for a single participant at a single bag placement
    for input_file in input_files:
        # Extract filename from input_file path
        filename = os.path.basename(input_file)
        # Generate output file path with the new folder and interpolated suffix
        output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '_INTERPOLATED.csv')

        # Lists to store the data from each column
        column1 = []
        column2 = []
        column3 = []

        # Read data from the input CSV file
        with open(input_file, 'r', encoding='utf-8-sig') as file:
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

def ankle_process(csv_files):
    for csv_file in csv_files:
        # Check if the file exists
        if os.path.exists(csv_file):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file, header=None)
            
            # Perform the operation (90 - value) for each value in the DataFrame
            df = 90 - df
            
            # Write the new DataFrame back to the CSV file
            df.to_csv(csv_file, header=False, index=False)
            print(f"Processed: {csv_file}")



def minmax_angle_list(file_list, start_row, end_row, output_file, search_type='max'):
    result_list = []

    for file_name in file_list:
        with open(file_name, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip header row if exists
            
            # Initialize values based on search_type
            if search_type == 'max':
                values = [-float('inf')] * 501
                comparison_func = max
            elif search_type == 'min':
                values = [float('inf')] * 501
                comparison_func = min
            else:
                raise ValueError("Invalid search_type. Use 'max' or 'min'.")

            # Iterate through each row in the CSV file
            for i, row in enumerate(csv_reader):
                # Check if the current row is within the specified range
                if start_row <= i <= end_row:
                    # Update values with the maximum or minimum values found in the current row
                    values = [comparison_func(cur_val, float(val)) for cur_val, val in zip(values, row)]
            
            # Extend result_list with the values from the current file
            result_list.extend(values)

    # Write result_list to the specified output file as a single column
    with open(output_file, 'w', newline='') as output_csv:
        csv_writer = csv.writer(output_csv)
        csv_writer.writerows([[val] for val in result_list])

    return result_list


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

def create_boxplot(csv_file, save_path, title='Boxplot', xlabel='X-axis', ylabel='Y-axis'):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Extract data from columns
    data = [df[col].dropna() for col in df.columns]
    
    # Create boxplot
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    plt.boxplot(data, labels=df.columns, patch_artist=True, boxprops=dict(facecolor='white', color='black'), whiskerprops=dict(color='black'), capprops=dict(color='black'), medianprops=dict(color='black'))
    
    # Customize plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    
    # Save plot as PNG at 600 dpi
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')



#####################
### WORKING SPACE ###
#####################

# Performing data processing (probably can run each time without wasting THAT much time and computing power)
folder = '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_raw'
files_list = list_csv_files_in_folder(folder, True)
raw_data_processing(files_list)
anklefolder = '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_interpolated'
ankle_files_list = list_csv_files_in_folder(folder, False, 'ANKLE', 'ANKLE')
ankle_process(ankle_files_list)

# folder = '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_interpolated'
# files_list = list_csv_files_in_folder(folder, False, 'KNEE', 'HIGH')



# create_boxplot('/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/processed_compiled/MAX_SWING_KNEE.csv', '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/figures/MAX_SWING_KNEE.png', 'Maximum Knee Joint Angle During Swing Phase', 'Bag Position', 'Angle (deg)')