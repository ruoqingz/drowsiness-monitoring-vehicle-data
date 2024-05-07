import json
import requests
import csv
import sseclient


def process_data(data):
    if data["game"]["connected"]:
        print("Game connected")
        with open("telemetry.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([data["game"]["time"], data["game"]["timeScale"],
                             data["truck"]["speed"], data["truck"]["cruiseControlSpeed"],
                             data["truck"]["cruiseControlOn"],
                             data["truck"]["userSteer"], data["truck"]["userThrottle"], data["truck"]["userBrake"],
                             data["truck"]["gameSteer"], data["truck"]["placement"]["x"],
                             data["truck"]["placement"]["y"],
                             data["truck"]["acceleration"]["x"], data["truck"]["acceleration"]["y"]])

    else:
        print("Error: Game not connected")


if __name__ == '__main__':
    url = "http://localhost:25555/api/ets2/telemetry"  # not public internet, please connect to local server
    response = requests.get(url)
    client = sseclient.SSEClient(response)

    with open("telemetry.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time", "timeScale", "speed", "cruiseControlSpeed", "cruiseControlOn", "userSteer", "userThrottle",
             "userBrake",
             "gameSteer", "placement_x", "placement_y", "acceleration_x", "acceleration_y"])

    for event in client:
        data = json.loads(event.data)
        process_data(data)



