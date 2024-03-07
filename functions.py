import numpy as np
import csv

def interpolate_to_501_points(input_list):
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

# # Example usage:
# input_list = [1, 2, np.nan, 4, 5, np.nan, 7, 8]  # Example list with NaN for missing values
# interpolated_values = interpolate_to_501_points(input_list)
# print(interpolated_values)

def csv_to_list(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header if it exists
        for row in csv_reader:
            for value in row:
                data_list.append(float(value))  # Convert to float if necessary
    return data_list

def list_to_csv(input_list, file_path):
    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        for item in input_list:
            csv_writer.writerow([item])

# Example usage:
input_file = '/Users/nigelfoo/Documents/low knee nigel 3.csv'  # Change this to your input CSV file path
output_file = '/Users/nigelfoo/Documents/low knee nigel 3 interpolated.csv' # Change this to your output CSV file path 
data_list = csv_to_list(input_file)
processed = interpolate_to_501_points(data_list)
list_to_csv(processed, output_file)
print("List written to CSV file:", output_file)