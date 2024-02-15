import pandas as pd

def calculate_averages(log_file_path, output_file):
    # Read the log file into a DataFrame
    df = pd.read_csv(log_file_path, sep='\t', header=None, names=['ID', 'Value1', 'Value2'])

    # Lists to store the averages of each section
    avg_values1 = []
    avg_values2 = []

    # Temp lists to store values for current section
    current_values1 = []
    current_values2 = []

    for index, row in df.iterrows():
        if row['ID'] == 0 and current_values1 and current_values2:
            # Calculate and append the averages of the current section
            avg_values1.append(sum(current_values1) / len(current_values1))
            avg_values2.append(sum(current_values2) / len(current_values2))
            # Reset the temp lists for the next section
            current_values1 = []
            current_values2 = []

        current_values1.append(row['Value1'])
        current_values2.append(row['Value2'])

    # Don't forget to calculate averages for the last section if it exists
    if current_values1 and current_values2:
        avg_values1.append(sum(current_values1) / len(current_values1))
        avg_values2.append(sum(current_values2) / len(current_values2))

    # Create a DataFrame from the averages
    averages_df = pd.DataFrame({
        'AvgValue1': avg_values1,
        'AvgValue2': avg_values2
    })

    # Save the averages to a new CSV file
    averages_df.to_csv(output_file, sep=";", index=False)


model_value = '9'
# Loop through the log files
for i in range(1, 7):  # Assuming loss files are numbered 1 through 6
    log_file_path = f'C:/DeepL/models_{model_value}/loss_{str(i)}.log'
    output_file = f'C:/DeepL/models_{model_value}/avg_{str(i)}.csv'
    calculate_averages(log_file_path, output_file)
    print(f'Averages for {log_file_path} calculated and saved to {output_file}.')
