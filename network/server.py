# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import socket
from typing import Dict, Union
import asyncio
from utils import monitor


class ServerManager:
    def __init__(self):
        self.server = None

    def setmonitor(self, ServerName: str):
        self.monitor = monitor.ResourceMonitor(ServerName)

    async def start_server(self, Role: str, IP: str, Port: int, DRDIP: str, DRDPort: int):
        self.server = Server(IP, Port, DRDIP, DRDPort, Role)
        print(
            f"Starting {self.server.__class__.__name__} on {self.server.IP}:{self.server.Port}"
        )
        await asyncio.gather(self.monitor.monitor_program(), self.server.start())

    async def stop_server(self):
        await self.server.stop()
        print(
            f"Stop {self.server.__class__.__name__} on {self.server.IP}:{self.server.Port}"
        )


class Server:
    server: socket.socket = None
    Role: str = None
    Status: str = None
    ServerId: int = None
    IP: str = None
    Port: int = None
    DRD: Dict[str, Union[str, int]] = {"ip": "", "port": 0}
    Clients = set()
    ClientsInfo = {}
    manager = ServerManager()
    IUs: Dict[int, dict] = {}
    IUsRecord: Dict[int, dict] = {}
    IUsSend: Dict[int, int] = {}
    pRecord = {}
    IUsStatus = {}
    AMSFlag = False

    def __init__(self, IP: str, Port: int, DRDIP: str, DRDPort: int, Role: str):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server.bind((IP, Port))
        self.DRD = {"ip": DRDIP, "port": DRDPort}
        self.ServerId = -1
        self.IP = IP
        self.Port = Port
        self.setRole(Role)

    def allocationNodeid(self, ip: str, port: int) -> int:
        device = (ip, port)

        if device in self.Clients:
            for nodeid, device_info in self.ClientsInfo.items():
                if device_info["ip"] == ip and device_info["port"] == port:
                    return nodeid

        NewID = 0
        for nodeid in self.ClientsInfo.keys():
            if NewID != nodeid:
                break
            NewID += 1
        self.ClientsInfo[NewID] = {"ip": ip, "port": port}
        self.ClientsInfo = dict(sorted(self.ClientsInfo.items()))
        self.Clients.add(device)
        return NewID

    def setRole(self, Role: str):
        if Role not in {"AMS", "MSP"}:
            raise ValueError("Please input a valid role: 'AMS' or 'MSP'")
        self.Role = Role

    def deleteNode(self, Nodeid) -> None:
        ClientInfo = self.ClientsInfo.get(Nodeid)
        ip = ClientInfo.get("ip")
        port = ClientInfo.get("port")
        device = (ip, port)
        self.Clients.discard(device)
        self.ClientsInfo.pop(Nodeid)

    async def udp_send(self, message: str, addr: str, port: int) -> None:
        """Send a UDP message to the specified address and port."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.server.sendto, message.encode(), (addr, port))

    async def Initiatelisten(self):
        loop = asyncio.get_running_loop()
        while self.ServerId == -1:
            data, addr = await loop.run_in_executor(None, self.server.recvfrom, 2048)
            message = data.decode()
            if message.startswith("AMSInitiate") or message.startswith("MSPInitiate"):
                parts = message.split()
                if len(parts) < 3:
                    raise ValueError("Unexpected message format")
                status, ID = parts[1], parts[2]
                if status == "Success":
                    self.ServerId = int(ID)
                    print("Server Initiate Successfully")
                elif status == "Fail":
                    raise ConnectionError(f"{self.Role} initiate Fail")
                else:
                    raise ValueError(f"Unknown status in {self.Role} initiate response")

    async def Initiate(self):
        Type = self.Role
        if Type == "AMS":
            message = "AMSInitiate"
            self.AMSFlag = True
        elif Type == "MSP":
            message = "MSPInitiate"
        else:
            raise ValueError("Unknown server type")
        timeout = 1
        DRDIP, DRDPort = self.DRD["ip"], self.DRD["port"]
        while self.ServerId == -1:
            await self.udp_send(message, DRDIP, DRDPort)
            try:
                await asyncio.wait_for(self.Initiatelisten(), timeout)
                break
            except asyncio.TimeoutError:
                if self.ServerId == -1:
                    print("No response received, sending again...")
            except Exception as e:
                print(f"Error during initiation: {str(e)}")
                break
    async def start(self):
        await self.Initiate()
        print("Start Listening")
        await self.listen()

    async def stop(self):
        if self.server:
            self.server.close()
    def clear(self):
        self.IUs.clear()
        self.IUsRecord.clear()
        self.IUsSend.clear()
        self.pRecord.clear()
        self.IUsStatus.clear()

    async def listen(self):
        """ Listen for incoming UDP messages. """
        loop = asyncio.get_event_loop()
        while True:
            data, addr = await loop.run_in_executor(None, self.server.recvfrom, 2048)
            message = data.decode()
            await self.handle_message(message, addr)

    async def handle_message(self, message: str, addr):
        if message.startswith("Abnormal"):
            parts = message.split()
            if len(parts) > 1:
                com = parts[1]
                if com == "ChangeStatus":
                    self._handleChangeStatus()
                elif com == "ChangeRole":
                    self._handleChangeRole(message)
                elif com == "IUChangeStatus":
                    await self._handleIUChangeStatus(message)
                elif com == "IUChangeServer":
                    await self._handleIUChangeServer(message)

        elif message.startswith("IUInitiate"):
            await self._handleIUInitiate(addr)
        elif message.startswith("Start"):
            self._handleIUSendStart(message)
        elif message.startswith("Onhold"):
            self._handleIUSendOnhold(message)
        elif message.startswith("Stop"):
            await self._handleIUSendStop(message)

    def _handleChangeRole(self, message: str):
        parts = message.split()
        Role = parts[2]
        if Role == "AMS" and self.AMSFlag is not False:
            self.Role = "MSP"
        elif Role == "MSP":
            self.Role = "AMS"
        else:
            raise ValueError("Invalid role")

    def _handleChangeStatus(self):
        if self.Role == "AMS":
            if self.Status == "NAMS":
                self.Status = "EAMS"
            elif self.Status == "EAMS":
                self.Status = "NAMS"
            else:
                raise ValueError("Invalid status")
        elif self.Role == "MSP":
            if self.Status == "NMSP":
                self.Status = "EMSP"
            elif self.Status == "EMSP":
                self.Status = "NMSP"
            else:
                raise ValueError("Invalid status")
        else:
            raise ValueError("Invalid role")

    async def _handleIUChangeStatus(self, message: str):
        parts = message.split()
        try:
            IUIP, IUPort = parts[2], int(parts[3])
            IUStatus = parts[4]
            for IUNodeid, IUInfo in self.IUs.items():
                if IUInfo["ip"] == IUIP and IUInfo["port"] == IUPort:
                    msg = f'Abnormal ChangeStatus {IUStatus}'
                    self.IUsStatus[IUNodeid] = IUStatus
                    await self.udp_send(msg, IUIP, IUPort)
                    break

        except (IndexError, ValueError) as e:
            print(f"Invalid ChangeServer message format: {e}")
            return

    async def _handleIUChangeServer(self, message: str):
        print(message)
        parts = message.split()
        try:
            IUid = None
            IUIP, IUPort = parts[2], int(parts[3])
            NewAMSIP, NewAMSPort = parts[4], int(parts[5])
            for IUNodeid, IUInfo in self.IUs.items():
                if IUInfo["ip"] == IUIP and IUInfo["port"] == IUPort:
                    IUid = IUNodeid
                    break

        except (IndexError, ValueError) as e:
            print(f"Invalid ChangeServer message format: {e}")
            return

        # 处理 ChangeServer 消息
        msg = f'Abnormal ChangeServer {NewAMSIP} {NewAMSPort}'
        await self.udp_send(msg, IUIP, IUPort)
        if self.IUs.get(IUid):
            self.IUs.pop(IUid)
        if self.IUsRecord.get(IUid):
            self.IUsRecord.pop(IUid)
        if self.IUsSend.get(IUid):
            self.IUsSend.pop(IUid)
        if self.IUsStatus.get(IUid):
            self.IUsStatus.pop(IUid)
        if self.pRecord.get(IUid):
            self.pRecord.pop(IUid)

    async def _handleIUInitiate(self, addr):
        IUip, IUport = addr[0], addr[1]
        Nodeid = self.allocationNodeid(IUip, IUport)
        self.IUs[Nodeid] = {"ip": IUip, "port": IUport}
        Initmsg = f"IUInitiate Success {Nodeid}"
        await self.udp_send(Initmsg, IUip, IUport)
        msg = f"IUsInfo {self.ServerId} {self.IUs}"
        await self.udp_send(msg, self.DRD.get("ip"), self.DRD.get("port"))
        self.IUsRecord[Nodeid] = {"Start": 0, "Stop": 0, "Record": 0}

    def _handleIUSendStart(self, message: str):
        # Start 0 NIU
        parts = message.split()
        Node = int(parts[1])
        if self.IUsRecord.get(Node) is None:
            return
        else:
            self.IUsRecord[Node]["Start"] = time.time()
            self.IUsSend[Node] = 0
            self.IUsRecord, self.IUsSend = self.IUsRecord, self.IUsSend
            self.IUsStatus[Node] = parts[2]

    def _handleIUSendOnhold(self, message: str):
        parts = message.split()
        Node = int(parts[1])
        self.IUsSend[Node] = self.IUsSend.get(Node, 0) + 1

    async def _handleIUSendStop(self, message: str):
        parts = message.split()
        Node = int(parts[1])
        if self.IUsRecord.get(Node) is None:
            return
        ServerStart = self.IUsRecord.get(Node).get("Start")
        ServerStop = time.time()
        ServerRecv = self.IUsSend.get(Node)

        IUSendpackets = int(parts[2])
        IUStart = float(parts[3])
        IUStop = float(parts[4])

        if ServerStart is None or ServerRecv is None:
            return

        p = 0
        if int(IUStart) / 10 == int(ServerStart) / 10 and int(IUStop) / 10 == int(ServerStop) / 10:
            IUdiff = IUStop - IUStart
            Serverdiff = ServerStop - ServerStart
            diff = Serverdiff - IUdiff
            diff = abs(diff)

            if diff > 0.5:
                p = 0
            else:
                p = float(ServerRecv) / float(IUSendpackets)
        if self.IUs.get(Node):
            self.pRecord[Node] = p
        msg = f"IUsPRecord {self.ServerId} {self.pRecord}"
        await self.udp_send(msg, self.DRD.get("ip"), self.DRD.get("port"))
        msg = f"IUsStatus {self.ServerId} {self.IUsStatus}"
        await self.udp_send(msg, self.DRD.get("ip"), self.DRD.get("port"))
