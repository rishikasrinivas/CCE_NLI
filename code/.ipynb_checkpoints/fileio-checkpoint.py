import csv
import os
def log_to_csv(file, data, header):
    if os.path.exists(file):
        mode = 'a'
    else:
        mode = 'w'
    with open(file, mode=mode, newline='') as file:
        writer = csv.writer(file, delimiter=',')

        # Write the header
        if mode == 'w':
            writer.writerow(header)
        # Write the data rows
        writer.writerow(data)
