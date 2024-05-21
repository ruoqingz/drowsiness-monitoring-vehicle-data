import csv
import aiohttp
import asyncio
import datetime


async def get_data(session,url):
    async with session.get(url) as response:
        if response.status == 200:
            return await response.json()
        else:
            print("Failed to fetch data from the URL.")
            return None


async def write_data(data):
    global lock
    async with lock:
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


async def job(session, url):
    while True:
        asyncio.create_task(get_and_write_data(session,url))
        await asyncio.sleep(0.1)


async def get_and_write_data(session,url):
    game_data = await get_data(session,url)
    await write_data(game_data)


async def main(url):
    async with aiohttp.ClientSession() as session:
        await job(session, url)


if __name__ == '__main__':
    url = "http://localhost:25555/api/ets2/telemetry"
    with open("telemetry.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["realwordTime","time", "timeScale", "speed", "cruiseControlSpeed", "cruiseControlOn", "userSteer", "userThrottle",
             "userBrake",
             "gameSteer", "placement_x", "placement_y", "acceleration_x", "acceleration_y"])
    lock = asyncio.Lock()
    asyncio.run(main(url))




