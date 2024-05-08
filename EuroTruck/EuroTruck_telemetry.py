import json
import requests
import csv
import aiohttp
import asyncio
import datetime


async def get_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                print("Failed to fetch data from the URL.")
                return None


async def write_data(data):
    if data["game"]["connected"]:
        print("Game connected")
        with open("telemetry.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.datetime.now(),data["game"]["time"], data["game"]["timeScale"],
                             data["truck"]["speed"], data["truck"]["cruiseControlSpeed"],
                             data["truck"]["cruiseControlOn"],
                             data["truck"]["userSteer"], data["truck"]["userThrottle"], data["truck"]["userBrake"],
                             data["truck"]["gameSteer"], data["truck"]["placement"]["x"],
                             data["truck"]["placement"]["y"],
                             data["truck"]["acceleration"]["x"], data["truck"]["acceleration"]["y"]])

    else:
        print("Error: Game not connected")


async def job(url):
    game_data = await get_data(url)
    await write_data(game_data)


async def main(url):
    while True:
        start_time = asyncio.get_event_loop().time()
        await asyncio.gather(job(url), asyncio.sleep(0.1))
        end_time = asyncio.get_event_loop().time()
        elapsed = end_time - start_time
        if elapsed < 0.1:
            await asyncio.sleep(0.1 - elapsed)


if __name__ == '__main__':
    url = "http://localhost:25555/api/ets2/telemetry"
    with open("telemetry.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["realwordTime","time", "timeScale", "speed", "cruiseControlSpeed", "cruiseControlOn", "userSteer", "userThrottle",
             "userBrake",
             "gameSteer", "placement_x", "placement_y", "acceleration_x", "acceleration_y"])

    asyncio.run(main(url))




