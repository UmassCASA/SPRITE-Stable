import csv
from datetime import datetime


def sort_csv_by_filename(csv_file_path, output_file_path):
    with open(csv_file_path, newline="") as file:
        # Read the CSV into memory
        reader = csv.reader(file)
        headers = next(reader)  # Grab the header row
        data = list(reader)

    # Sort the data based on the datetime in the filename
    data.sort(key=lambda row: datetime.strptime(row[0], "%Y%m%d_%H%M%S.nc"))

    # Write the sorted data back to a new CSV file
    with open(output_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)


# Example usage
csv_file_path = "/work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/max_values_casa_updated.csv"

output_file_path = "/work/pi_mzink_umass_edu/SPRITE/skillful_nowcasting/output/sorted_max_values_casa_updated.csv"
sort_csv_by_filename(csv_file_path, output_file_path)

print("Sorting completed.")
