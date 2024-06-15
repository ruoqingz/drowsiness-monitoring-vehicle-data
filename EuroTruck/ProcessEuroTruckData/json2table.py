import os
import json
import pandas as pd

os.chdir("/Users/ruotsing/PycharmProjects/DMS/EuroTruck/ProcessEuroTruckData")
# Define the folder path
folder_path = "logRaw"

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".log"):
        file_path = os.path.join(folder_path, filename)
        
        # Read JSON data from the file, skipping lines with extra data
        json_data = []
        with open(file_path, "r") as file:
            for line in file:
                try:
                    json_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {filename}: {line.strip()}")
        
        # Convert JSON data to DataFrame
        df = pd.DataFrame(json_data)
        
        # Write DataFrame to Excel file
        excel_filename = os.path.splitext(filename)[0] + ".xlsx"  # Change extension to .xlsx
        excel_filepath = os.path.join(folder_path, excel_filename)
        df.to_excel(excel_filepath, index=False)

df = pd.DataFrame()
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(folder_path, filename)
        df=pd.concat([df,pd.read_excel(file_path)], ignore_index=True)

df = df.set_index("TIME")
df = df.sort_index()
df.to_excel("device_history_0614_catia.xlsx", index=True)


print("Conversion completed for all files in the folder.")
