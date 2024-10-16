# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import socket
import time
from utils import monitor


class IUManager:

    def __init__(self):
        self.iu = None

    def setmonitor(self, ServerName: str):
        self.monitor = monitor.ResourceMonitor(ServerName)

    async def start_iu(self, IP: str, Port: int, ServerIP: str, ServerPort: int):
        self.iu = IU(IP, Port, ServerIP, ServerPort)

        print(
            f"Starting {self.iu.__class__.__name__} on {self.iu.IP}:{self.iu.Port}"
        )
        await asyncio.gather(self.iu.start(), self.monitor.monitor_program())

    async def stop_iu(self):
        await self.iu.stop()
        print(
            f"Stop {self.iu.__class__.__name__} on {self.iu.IP}:{self.iu.Port}"
        )


class IU:
    IP: str = ""
    Port: int = 0
    status: str = ""
    client = None
    NodeID: int = -1
    change: bool = False
    Server: dict = {"ip": "", "port": 0}
    manager = None
    changed = False

    def __init__(self, ip: str, port: int, ServerIP: str, ServerPort: int) -> None:
        self.IP = ip
        self.Port = port
        self.status = "NIU"
        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client.bind((ip, port))
        self.Server["ip"], self.Server["port"] = ServerIP, ServerPort
        self.NodeID = -1
        self.manager = IUManager()

    async def start(self):
        await self.Initiate()
        await asyncio.gather(
            self.listen(),
            self.send_info()
        )

    async def stop(self):
        if self.client:
            self.client.close()
            print("UDP Socket has been closed.")

    async def Initiatelisten(self):
        loop = asyncio.get_running_loop()
        while self.NodeID == -1:
            data, addr = await loop.run_in_executor(None, self.client.recvfrom, 2048)
            message = data.decode()
            if message.startswith("IUInitiate"):
                parts = message.split()
                if len(parts) < 3:
                    raise ValueError("Unexpected message format")
                status, ID = parts[1], parts[2]
                if status == "Success":
                    self.NodeID = int(ID)
                    print("IU Initiate Successfully")
                elif status == "Fail":
                    raise ConnectionError("IU initiate Fail")
                else:
                    raise ValueError("Unknown status in Server initiate response")

    async def Initiate(self):
        message = "IUInitiate"
        timeout = 1
        ServerIP, ServerPort = self.Server["ip"], self.Server["port"]
        while True:
            await self.udp_send(message, ServerIP, ServerPort)
            try:
                await asyncio.wait_for(self.Initiatelisten(), timeout)
                break
            except asyncio.TimeoutError:
                if self.NodeID == -1:
                    print("No response received, sending again...")
            except Exception as e:
                print(f"Error during initiation: {str(e)}")
                break

    async def InitiateChanged(self):
        message = "IUInitiate"
        ServerIP, ServerPort = self.Server["ip"], self.Server["port"]
        await self.udp_send(message, ServerIP, ServerPort)

    async def udp_send(self, message: str, addr: str, port: int) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.client.sendto, message.encode(), (addr, port))

    async def listen(self):
        """ Listen for incoming UDP messages. """
        loop = asyncio.get_event_loop()
        while True:
            data, addr = await loop.run_in_executor(None, self.client.recvfrom, 2048)
            message = data.decode()
            await self.handle_message(message, addr)

    async def handle_message(self, message: str, addr) -> None:
        parts = message.split()
        com = parts[1]
        if message.startswith("Abnormal"):
            if com == "ChangeStatus":
                self._handleIUChangeStatus()
            elif com == "ChangeServer":
                if self.changed is False:
                    ServerIP = parts[2]
                    ServerPort = int(parts[3])
                    self.Server["ip"], self.Server["port"] = ServerIP, ServerPort
                    self.NodeID = -1
                    self.changed = True
        if message.startswith("IUInitiate"):
            if com == "Success":
                self.NodeID = int(parts[2])
                self.changed = False
            elif com == "Fail":
                raise ConnectionError("IU initiate Fail")
            else:
                raise ValueError("Unknown status in Server initiate response")

    def _handleIUChangeStatus(self):
        if self.status == "NIU":
            self.status = "EIU"
        elif self.status == "EIU":
            self.status = "NIU"

    async def send_info(self):
        while True:
            if self.NodeID != -1:
                ip, port = self.Server["ip"], self.Server["port"]

                if self.NodeID == -1:
                    continue

                send_count = 0
                start_time = time.time()
                await self.udp_send(f"Start {self.NodeID} {self.status}", ip, port)

                while time.time() - start_time < 1:
                    if self.NodeID == -1:
                        break
                    await self.udp_send(f"Onhold {self.NodeID}", ip, port)
                    send_count += 1
                    await asyncio.sleep(0.2)

                if self.NodeID != -1:
                    end_time = time.time()
                    message = f"Stop {self.NodeID} {send_count} {start_time:.2f} {end_time:.2f}"
                    await self.udp_send(message, ip, port)
            else:
                await self.InitiateChanged()
