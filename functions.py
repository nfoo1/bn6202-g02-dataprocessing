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
            print(f"Processed Ankles: {csv_file}")

def combine_csv(folder_path, term1, term2, combined_file_path):
    # OUTPUT IS AS FOLLOWS: EACH COLUMN OF EACH CSV FILE APPENDED COLUMNWISE, LAST TWO COLUMNS ARE AVERAGE AND SAMPLE STDEV RESPECTIVELY
    # FUNCTION CAN BE USED, AND RETURNS LIST OF COLLECTIVE AVERAGE AND SAMPLE STDEV 

    avg_list = []
    std_dev_list = []

    # Search for CSV files with specified search terms
    filtered_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and term1 in f and term2 in f]

    if not filtered_files:
        print("No matching CSV files found.")
        return [], []

    # Initialize list to store data from filtered files
    combined_data = []

    # Iterate through filtered files
    for file_name in filtered_files:
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            data = list(csv_reader)

        # Append data from filtered file to combined data list
        combined_data.append(data)

    # Transpose combined data to stack horizontally
    combined_data = np.array(combined_data).transpose((1, 0, 2))

    # Initialize list to store row averages and standard deviations
    row_avg_list = []
    row_std_dev_list = []

    # Iterate through rows in combined data
    for row in combined_data:
        # Convert row values to float
        row_values = np.array(row).astype(float)

        # Calculate row average and standard deviation
        row_avg = np.mean(row_values)
        row_std_dev = np.std(row_values, ddof=1)

        # Append row average and standard deviation to respective lists
        row_avg_list.append(row_avg)
        row_std_dev_list.append(row_std_dev)

    # Write combined data to a new CSV file
    with open(combined_file_path, 'w', newline='') as combined_file:
        csv_writer = csv.writer(combined_file)

        # Write combined data rows without header
        for i in range(len(combined_data)):
            combined_row = combined_data[i].flatten().tolist()
            combined_row += [row_avg_list[i], row_std_dev_list[i]]
            csv_writer.writerow(combined_row)

    return row_avg_list, row_std_dev_list

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

def rawdata_batch_interpolation():
    # Performing data processing (probably can run each time without wasting THAT much time and computing power)
    # Or you can perform this each time data_raw is updated

    folder = '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_raw'
    files_list = list_csv_files_in_folder(folder, True)
    raw_data_processing(files_list)
    anklefolder = '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_interpolated'
    ankle_files_list = list_csv_files_in_folder(anklefolder, False, 'ANKLE', 'INTERPOLATED')
    ankle_process(ankle_files_list)


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


def individual_line_plot(average_list, std_deviation_list, save_path, title, xlabel, ylabel):
    # Convert lists to numpy arrays
    average_list = np.array(average_list)
    std_deviation_list = np.array(std_deviation_list)

    # Generating x values
    x_values = np.arange(0, 100.1, 0.2)  # 501 values from 0 to 100 (inclusive)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, average_list, color='blue', label='Average')
    plt.fill_between(x_values, average_list - std_deviation_list, average_list + std_deviation_list, color='lightblue', label='Standard Deviation')

    # Adding labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Setting x-axis range
    plt.xlim(0, 100)

    # Display grid
    plt.grid(True)

    # Save the plot
    plt.savefig(save_path, dpi=600)
    print(f"Plot saved at: {save_path}")
    plt.close()

def individual_line_plot_trunk(average_list, std_deviation_list, save_path, title, xlabel, ylabel):
    # Convert lists to numpy arrays
    average_list = np.array(average_list)
    std_deviation_list = np.array(std_deviation_list)

    # Generating x values
    x_values = np.arange(0, 100.1, 0.2)  # 501 values from 0 to 100 (inclusive)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, average_list, color='blue', label='Average')
    plt.fill_between(x_values, average_list - std_deviation_list, average_list + std_deviation_list, color='lightblue', label='Standard Deviation')

    # Adding labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Setting x-axis range
    plt.ylim(-20, 20)
    plt.xlim(0, 100)

    # Display grid
    plt.grid(True)

    # Save the plot
    plt.savefig(save_path, dpi=600)
    print(f"Plot saved at: {save_path}")
    plt.close()

def batch_individual_line_plots():
    joints = ['KNEE', 'HIP', 'ANKLE']
    levels = ['CONTROL', 'HIGH', 'MEDIUM', 'LOW']

    for joint in joints:
        for level in levels:
            path = f'/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/processed_compiled/LONGITUDINAL_{level}_{joint}.csv'
            average, stdev = combine_csv('/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_interpolated', joint, level, path)
            filename = f'/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/figures/LONGITUDINAL_{joint}_{level}.png'
            individual_line_plot(average, stdev, filename, f'Average {joint.capitalize()} Joint Angle - {level.capitalize()}', 'Percentage of Gait Cycle (%)', f'{joint.capitalize()} Flexion Angle (deg)')

def batch_individual_line_plots_trunk():
    joints = ['TRUNK']
    levels = ['CONTROL', 'HIGH', 'MEDIUM', 'LOW']

    for joint in joints:
        for level in levels:
            path = f'/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/processed_compiled/LONGITUDINAL_{level}_{joint}.csv'
            average, stdev = combine_csv('/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_interpolated', joint, level, path)
            filename = f'/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/figures/LONGITUDINAL_{joint}_{level}.png'
            individual_line_plot_trunk(average, stdev, filename, f'Average {joint.capitalize()} Joint Angle - {level.capitalize()}', 'Percentage of Gait Cycle (%)', f'{joint.capitalize()} Flexion Angle (deg)')

def longitudinal_comparison(folder_path, joint_descriptor, save_filepath, title="Line Plot", x_label="X", y_label="Y"):
    # Search for CSV files containing the specified joint descriptor
    csv_files = [file for file in os.listdir(folder_path) if file.startswith('LONGITUDINAL_') and joint_descriptor in file]
    
    # Ensure there are exactly four CSV files
    if len(csv_files) != 4:
        print("Error: There should be exactly four CSV files matching the joint descriptor.")
        return
    
    # Define colors for different types of data
    colors = {'CONTROL': 'blue', 'HIGH': 'green', 'MEDIUM': 'orange', 'LOW': 'purple'}
    
    # Initialize a matplotlib figure
    fig, ax = plt.subplots()

    # Setting x-axis to 0-100% gait cycle, and limiting the limits of x-axis.
    x_vals = np.arange(0, 100, 0.2)
    ax.set_xlim(0, 100)

    # Set y-axis limits for TRUNK joint descriptor
    if joint_descriptor == "TRUNK":
        ax.set_ylim(-20, 20)
    
    # Iterate over each CSV file
    for idx, file in enumerate(csv_files):
        # Read CSV file
        df = pd.read_csv(os.path.join(folder_path, file))
        # Extract relevant columns (last two columns)
        avg_column = df.iloc[:, -2]
        std_dev_column = df.iloc[:, -1]
        # Extract type of data (CONTROL, HIGH, MEDIUM, LOW)
        data_type = file.split('_')[1]
        # Plot average line
        ax.plot(x_vals, avg_column, color=colors[data_type], label=data_type)

    # Add shaded area for standard deviation outside the loop to ensure it doesn't affect legend order
    for idx, file in enumerate(csv_files):
        df = pd.read_csv(os.path.join(folder_path, file))
        avg_column = df.iloc[:, -2]
        std_dev_column = df.iloc[:, -1]
        data_type = file.split('_')[1]
        # Shaded area for stdev
        # ax.fill_between(x_vals, avg_column - std_dev_column, avg_column + std_dev_column, alpha=0.2, color=colors[data_type])
    
    # Set plot title and axis labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Adjust legend order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper right')  # Reverse order
    
    # Save plot as PNG file at 600 dpi
    plt.savefig(save_filepath, dpi=600)
    
    # Show plot
    # plt.show()
    plt.close()

def batch_logitudinal_comparison():
    joints = ['KNEE', 'HIP', 'TRUNK', 'ANKLE']

    for joint in joints:
        if joint == 'TRUNK':
            y_label = 'Trunk Flexion Angle (deg)'
        elif joint == 'ANKLE':
            y_label = 'Ankle Plantarflexion Angle (deg)'
        else:
            y_label = f'{joint.capitalize()} Flexion Angle (deg)'
        
        save_filepath = f'/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/figures/COMPARISON_{joint}.png'
        title = f'{joint.capitalize()} Joint Angles'

        longitudinal_comparison('/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/processed_compiled',
                                joint_descriptor=joint,
                                save_filepath=save_filepath,
                                title=title,
                                x_label='Percentage of Gait Cycle (%)',
                                y_label=y_label)

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
    plt.close()


#####################
### WORKING SPACE ###
#####################

# Run if needed
rawdata_batch_interpolation()

# Batch longitudinal plots
batch_individual_line_plots()
batch_individual_line_plots_trunk()

# Batch comparison plots
batch_logitudinal_comparison()

folder = '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/data_interpolated'
files_list = list_csv_files_in_folder(folder, False, 'KNEE', 'CONTROL')
minmax_angle_list(files_list, 300, 501, '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/processed/max_knee_swingphase_control.csv', 'max')
files_list = list_csv_files_in_folder(folder, False, 'KNEE', 'HIGH')
minmax_angle_list(files_list, 300, 501, '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/processed/max_knee_swingphase_high.csv', 'max')
files_list = list_csv_files_in_folder(folder, False, 'KNEE', 'MEDIUM')
minmax_angle_list(files_list, 300, 501, '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/processed/max_knee_swingphase_medium.csv', 'max')
files_list = list_csv_files_in_folder(folder, False, 'KNEE', 'LOW')
minmax_angle_list(files_list, 300, 501, '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/processed/max_knee_swingphase_low.csv', 'max')

create_boxplot('/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/processed_compiled/MAX_SWING_KNEE.csv', '/Users/nigelfoo/Documents/bn6202-g02-dataprocessing/bn6202-g02-dataprocessing/figures/MAX_SWING_KNEE.png', 'Maximum Knee Joint Angle During Swing Phase', 'Bag Position', 'Angle (deg)')