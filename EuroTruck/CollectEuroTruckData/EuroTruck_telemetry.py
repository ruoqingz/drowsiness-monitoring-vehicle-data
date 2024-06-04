import csv
import aiohttp
import asyncio
import datetime


async def get_data(session, url):
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
            with open(name, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.datetime.now(), data["game"]["time"], data["game"]["timeScale"],
                                 data["truck"]["speed"], data["truck"]["cruiseControlSpeed"],
                                 data["truck"]["cruiseControlOn"],
                                 data["truck"]["userSteer"], data["truck"]["userThrottle"], data["truck"]["userBrake"],
                                 data["truck"]["gameSteer"], data["truck"]["placement"]["x"],
                                 data["truck"]["placement"]["y"],data["truck"]["placement"]["heading"],
                                 data["truck"]["acceleration"]["x"], data["truck"]["acceleration"]["y"]])

        else:
            print("Error: Game not connected")


async def job(session, url):
    while True:
        asyncio.create_task(get_and_write_data(session, url))
        await asyncio.sleep(0.025)


async def get_and_write_data(session, url):
    game_data = await get_data(session, url)
    await write_data(game_data)


async def main(url):
    async with aiohttp.ClientSession() as session:
        await job(session, url)


if __name__ == '__main__':
    # 10.130.21.233 is the IP address of the server
    url = "http://10.130.21.233:25555/api/ets2/telemetry"
    name="telemetry_0604_4_michele.csv"
    with open(name, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["realwordTime", "time", "timeScale", "speed", "cruiseControlSpeed", "cruiseControlOn", "userSteer",
             "userThrottle",
             "userBrake",
             "gameSteer", "placement_x", "placement_y","placement_yaw", "acceleration_x", "acceleration_y"])
    lock = asyncio.Lock()
    asyncio.run(main(url))
