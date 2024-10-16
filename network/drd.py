# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import socket
from algorithm.PROSTM import PROSTMNetwork
from utils.tools import splitData
from utils import monitor
import logging


class DRDManager:

    def __init__(self):
        self.drd = None

    async def start_server(self, Type, IP: str, Port: int, N=0, T=0, theta1=0, theta2=0, theta3=0, theta4=0, delta=0,
                           k=0, u=0,
                           v=0, X=0, Y=0):
        if Type == "Model":
            self.drd = DRDWM(IP, Port, N, T)
            self.drd.setBasicVar(theta1, theta2, theta3, theta4, delta, k, u, v, X, Y)
            self.drd.ConnectMartix()
        elif Type == "NoModel":
            self.drd = DRDNM(IP, Port)
        else:
            raise ValueError("Invalid Type(Model,NoModel)")
        print(
            f"Starting {self.drd.__class__.__name__} on {self.drd.IP}:{self.drd.Port}"
        )
        await asyncio.gather(self.monitor.monitor_program(), self.drd.start())

    def setmonitor(self, ServerName: str):
        self.monitor = monitor.ResourceMonitor(ServerName)

    async def stop_server(self):
        await self.drd.stop()


class drdbase:
    server: socket.socket = None
    IP, Port = "", 0
    Servers = set()
    ServersInfo = {}
    ServerNodes = {}
    ServerType = {}
    ServerStatus = {}
    ServerAMSStatus = {}
    ServersP = {}
    ServerConIUNum = {}

    IUs = set()
    IUsInfo = {}
    IUsStatus = {}
    IUsP = {}
    IUConServer = {}

    def setup_logger(self):
        name = "P"
        file_name = "PRecord.log"
        level = logging.INFO
        logger = logging.getLogger(name)
        logger.setLevel(level)
        handler = logging.FileHandler(file_name, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s,%(message)s'))
        handler.setLevel(level)
        logger.addHandler(handler)
        return logger

    def __init__(self, ip: str, port: int):
        self.IP, self.Port = ip, port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server.bind((ip, port))
        self.Plogger = self.setup_logger()

    async def start(self):
        print("DRD Starting Listening")
        await asyncio.gather(
            self.listen()
        )

    async def stop(self):
        if self.server:
            self.server.close()
            print("UDP Socket has been closed.")

    def allocationNodeid(self, ip: str, port: int, Type: str, ServerID=None, innerID=None) -> int:
        NewID = 0
        if Type == "Server":
            device = (ip, port)

            if device in self.Servers:
                for NodeID, device_info in self.ServersInfo.copy().items():
                    if device_info["ip"] == ip and device_info["port"] == port:
                        return NodeID

            for NodeID in self.ServersInfo.keys():
                if NewID != NodeID:
                    break
                NewID += 1

            self.ServersInfo[NewID] = {"ip": ip, "port": port}
            self.ServersInfo = dict(sorted(self.ServersInfo.copy().items()))
            self.Servers.add(device)
        elif Type == "Client":
            device = (ip, port)

            if device in self.IUs:
                for NodeID, device_info in self.IUsInfo.copy().items():
                    if device_info["ip"] == ip and device_info["port"] == port:
                        return NodeID

            for NodeID in self.IUsInfo.keys():
                if NewID != NodeID:
                    break
                NewID += 1

            self.IUsInfo[NewID] = {"ip": ip, "port": port, 'index': ServerID, "inner": innerID}
            self.IUsInfo = dict(sorted(self.IUsInfo.copy().items()))
            self.IUsStatus[NewID] = "NIU"
            self.IUs.add(device)
        return NewID

    async def IUChangeServer(self, ID):
        IUInfo = self.IUsInfo.get(ID)
        IUIP, IUPort, index = IUInfo["ip"], IUInfo["port"], IUInfo["index"]
        serverTypes = self.ServerType.values()
        sServerInfo = self.ServersInfo.get(index)
        sIP, sPort = sServerInfo["ip"], sServerInfo["port"]
        if "MSP" in serverTypes:
            MSPindex = list(serverTypes).index("MSP")
            ServerInfo = self.ServersInfo.get(index)
            ip, port = ServerInfo["ip"], ServerInfo["port"]
            await self.ServerChangeRole(MSPindex, ip, port, "AMS")
            ServerInfo = self.ServersInfo.get(MSPindex)
            ip, port = ServerInfo["ip"], ServerInfo["port"]
            message = f"Abnormal IUChangeServer {IUIP} {IUPort} {ip} {port}"
            await self.udp_send(message, sIP, sPort)
        else:
            for ID_ in self.ServerType.keys():
                if index == ID_:
                    continue
                else:
                    ServerInfo = self.ServersInfo.get(ID_)
                    ip, port = ServerInfo["ip"], ServerInfo["port"]
                    message = f"Abnormal IUChangeServer {IUIP} {IUPort} {ip} {port}"
                    await self.udp_send(message, sIP, sPort)
                    return

    async def ServerChangeRole(self, ID, ip, port, role):
        message = ""
        if role == "AMS":
            message = "Abnormal ChangeRole AMS"
        elif role == "MSP":
            message = "Abnormal ChangeRole MSP"
        if self.ServerAMSStatus.get(ID):
            return
        await self.udp_send(message, ip, port)
        self.ServerType[ID] = role

    def deleteServer(self, Nodeid) -> None:
        ServerInfo = self.ServersInfo.get(Nodeid)
        ip = ServerInfo.get("ip")
        port = ServerInfo.get("port")
        device = (ip, port)
        self.Servers.discard(device)
        self.ServersInfo.pop(Nodeid)

    def deleteClient(self, Nodeid) -> None:
        ClientInfo = self.IUsInfo.get(Nodeid)
        ip = ClientInfo.get("ip")
        port = ClientInfo.get("port")
        device = (ip, port)
        self.IUs.discard(device)
        self.IUsInfo.pop(Nodeid)

    async def udp_send(self, message: str, addr: str, port: int) -> None:
        """Send a UDP message to the specified address and port."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.server.sendto, message.encode(), (addr, port))

    async def listen(self):
        loop = asyncio.get_event_loop()
        while True:
            data, addr = await loop.run_in_executor(None, self.server.recvfrom, 2048)
            message = data.decode()
            await self._handle_message(message, addr)

    async def _handle_message(self, message: str, addr: tuple) -> None:
        if message == "MSPInitiate":
            await self._handleMSPInitiate(addr)
        elif message == "AMSInitiate":
            await self._handleAMSInitiate(addr)
        elif message.startswith("IUsInfo"):
            self._handleIUsInfo(message)
        elif message.startswith("IUsPRecord"):
            await self._handleIUsP(message)

    async def _handleMSPInitiate(self, addr):
        ip, port = addr[0], addr[1]
        ID = self.allocationNodeid(ip, port, "Server")
        self.ServerNodes[ID] = {"ip": ip, "port": port}
        self.ServerType[ID] = 'MSP'
        self.ServerConIUNum[ID] = 0
        self.ServerStatus[ID] = "NMSP"
        await self.udp_send(f"MSPInitiate Success {ID}", ip, port)

    async def _handleAMSInitiate(self, addr):
        ip, port = addr[0], addr[1]
        ID = self.allocationNodeid(ip, port, "Server")
        self.ServerNodes[ID] = {"ip": ip, "port": port}
        self.ServerType[ID] = 'AMS'
        self.ServerAMSStatus[ID] = True
        self.ServerConIUNum[ID] = 0
        self.ServerStatus[ID] = "NAMS"
        await self.udp_send(f"AMSInitiate Success {ID}", ip, port)

    def _handleIUsInfo(self, message):
        ServerID = splitData(message, 2, 1)
        Info = splitData(message, 2, 2)  # 信息
        for IUid, IUInfo in Info.copy().items():
            ip, port = IUInfo['ip'], IUInfo['port']
            Nodeid = self.allocationNodeid(ip, port, "Client", ServerID, IUid)
            self.IUConServer[ServerID, IUid] = Nodeid
            self.ServerConIUNum[ServerID] += 1

    async def _handleIUsP(self, message):
        ServerId = splitData(message, 2, 1)  # AMS的ID
        P = splitData(message, 2, 2)
        ServerPSum, ServerNum = 0, 0
        for IUid, Pvalue in P.copy().items():
            if Pvalue == 0:
                Pvalue = 1
            IUindex = self.IUConServer.get(ServerId, IUid)
            self.IUsP[IUindex] = Pvalue
            ServerPSum += Pvalue
            ServerNum += 1
        self.ServersP[ServerId] = ServerPSum / ServerNum
        await self.Check()
        record = ""
        P, Status = 0, ""
        for i in range(0, 10):
            if i < 4:
                ID = i
                if self.ServersP.get(ID) is None:
                    P = 0
                else:
                    P = self.ServersP.get(ID)
                if self.ServerStatus.get(ID) is None:
                    Status = "NAMS"
                else:
                    Status = self.ServerStatus.get(ID)
            else:
                ID = i - 4
                if self.IUsP.get(ID) is None:
                    P = 0
                else:
                    P = self.IUsP.get(ID)
                if self.IUsStatus.get(ID) is None:
                    Status = "NIU"
                else:
                    Status = self.IUsStatus.get(ID)
            record += f"!{i}!{P}!{Status}!"
        self.Plogger.info(record)

    async def Check(self):
        raise NotImplementedError()


class DRDNM(drdbase):
    IUECount, ServerECount = {}, {}
    ServerNCount = {}
    IULimitMin, ServerLimitMin = 0.4, 0.9
    IULimitMax, ServerLimitMax = 1.5, 1.1

    async def deleteIU(self, ID):

        IUInfo = self.IUsInfo.get(ID)
        ip, port, serverid, innerid = IUInfo["ip"], IUInfo["port"], IUInfo["index"], IUInfo["inner"]
        device = (ip, port)

        self.IUs.discard(device)
        self.IUsInfo.pop(ID)
        self.IUsStatus.pop(ID)
        self.IUsP.pop(ID)
        self.IUConServer.pop((serverid, innerid))
        self.ServerConIUNum[serverid] -= 1
        self.IUECount.pop(ID)

    async def ServerStatusCheck(self):
        for ID, P in self.ServersP.copy().items():
            if P < self.ServerLimitMin or P > self.ServerLimitMax:
                if self.ServerType.get(ID) == "AMS":
                    self.ServerStatus[ID] = "EAMS"
                elif self.ServerType.get(ID) == "MSP":
                    self.ServerStatus[ID] = "EMSP"
                self.ServerNCount[ID] = 0
                if self.ServerECount.get(ID) is None:
                    self.ServerECount[ID] = 1
                else:
                    self.ServerECount[ID] += 1

            elif self.ServerLimitMin <= P <= self.ServerLimitMax:
                if self.ServerType.get(ID) == "AMS":
                    self.ServerStatus[ID] = "NAMS"
                elif self.ServerType.get(ID) == "MSP":
                    self.ServerStatus[ID] = "NMSP"
                self.ServerECount[ID] = 0
                if self.ServerNCount.get(ID) is None:
                    self.ServerNCount[ID] = 1
                else:
                    self.ServerNCount[ID] += 1

    async def ServerRecycle(self):
        for ID, Status in self.ServerStatus.copy().items():
            if 'E' in Status:
                if Status == "EAMS":
                    if self.ServerConIUNum[ID] == 0:
                        self.ServerStatus[ID] = "NAMS"
                        self.ServersP[ID] = 1
                    if self.ServerECount.get(ID) > 2:  # 可做调整
                        serverStatus = self.ServerStatus.values()
                        if "NMSP" not in serverStatus:
                            print("No MSP can Deploy")
                        else:
                            index = list(serverStatus).index("NMSP")
                            ServerInfo = self.ServersInfo.get(index)
                            ip, port = ServerInfo["ip"], ServerInfo["port"]
                            await self.ServerChangeRole(ID, ip, port, "AMS")
            if 'N' in Status:
                if Status == "NAMS":
                    if self.ServerConIUNum[ID] == 0:
                        if self.ServerNCount.get(ID) is None:
                            self.ServerNCount[ID] = 1
                        if self.ServerNCount.get(ID) > 10:
                            ServerInfo = self.ServersInfo.get(ID)
                            ip, port = ServerInfo["ip"], ServerInfo["port"]
                            await self.ServerChangeRole(ID, ip, port, "MSP")

    async def IUStatusCheck(self):
        for ID, P in self.IUsP.copy().items():
            if P < self.IULimitMin or P > self.IULimitMax:
                self.IUsStatus[ID] = "EIU"
                if self.IUsInfo.get(ID) is None:
                    return
                if self.IUECount.get(ID) is None:
                    self.IUECount[ID] = 1
                else:
                    self.IUECount[ID] += 1
            elif self.IULimitMin <= P <= self.IULimitMax:
                self.IUsStatus[ID] = "NIU"
                self.IUECount[ID] = 0

    async def IUCheck(self):
        for ID, Status in self.IUsStatus.copy().items():
            if Status == "EIU":
                if self.IUECount.get(ID, 0) > 2:  # 可做调整
                    await self.IUChangeServer(ID)
                    await self.deleteIU(ID)

    async def Check(self):
        await self.ServerStatusCheck()
        await self.ServerRecycle()
        await self.IUStatusCheck()
        await self.IUCheck()


class DRDWM(drdbase):
    Model = None
    IULimitMin, ServerLimitMin = 0.4, 0.9
    IULimitMax, ServerLimitMax = 1.5, 1.1

    def __init__(self, ip: str, port: int, N, T):
        super().__init__(ip, port)
        self.N = N
        self.T = T
        self.Model = ITSNetwork(N, T)

    def setBasicVar(self, theta1, theta2, theta3, theta4, delta, k, u, v, X, Y):
        self.Model.setTheta(theta1, theta2, theta3, theta4)
        self.Model.setBasicValue(delta, k, u, v, X, Y)

    def ConnectMartix(self):
        LAConnect = [(1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)]
        for i, j in LAConnect:
            self.Model.connection_matrix("LA", i, j)

    async def StatusCheck(self):
        for ID, P in self.ServersP.copy().items():
            if P < self.ServerLimitMin or P > self.ServerLimitMax:
                if self.ServerType.get(ID) == "AMS":
                    self.ServerStatus[ID] = "EAMS"
                elif self.ServerType.get(ID) == "MSP":
                    self.ServerStatus[ID] = "EMSP"


            elif self.ServerLimitMin <= P <= self.ServerLimitMax:
                if self.ServerType.get(ID) == "AMS":
                    self.ServerStatus[ID] = "NAMS"
                elif self.ServerType.get(ID) == "MSP":
                    self.ServerStatus[ID] = "NMSP"

            for ID, P in self.IUsP.copy().items():
                if P < self.IULimitMin or P > self.IULimitMax:
                    self.IUsStatus[ID] = "EIU"

                elif self.IULimitMin <= P <= self.IULimitMax:
                    self.IUsStatus[ID] = "NIU"

    async def Check(self):
        await self.StatusCheck()
        if len(self.ServersInfo) + len(self.IUsInfo) == 10:
            for ID, Type in self.ServerStatus.items():
                if ID == 0:
                    if Type == "NMSP":
                        P = self.ServersP.get(ID, 1)
                        self.Model.setP(Type, ID, P)
                        self.Model.setP("EMSP", ID, 0)
                    else:
                        P = self.ServersP.get(ID, 1)
                        self.Model.setP(Type, ID, P)
                        self.Model.setP("NMSP", ID, 0)
                    self.Model.setP("NAMS", ID, 0)
                    self.Model.setP("EAMS", ID, 0)
                    self.Model.setP("NIU", ID, 0)
                    self.Model.setP("EIU", ID, 0)
                else:
                    if Type == "NAMS":
                        P = self.ServersP.get(ID, 1)
                        self.Model.setP(Type, ID, P)
                        self.Model.setP("EAMS", ID, 0)
                    else:
                        P = self.ServersP.get(ID, 1)
                        self.Model.setP(Type, ID, P)
                        self.Model.setP("NAMS", ID, 0)
                    self.Model.setP("NMSP", ID, 0)
                    self.Model.setP("EMSP", ID, 0)
                    self.Model.setP("NIU", ID, 0)
                    self.Model.setP("EIU", ID, 0)

            index = len(self.ServerStatus)
            for ID, Type in self.IUsStatus.items():
                if Type == "NIU":
                    P = self.IUsP.get(ID, 1)
                    self.Model.setP(Type, ID + index, P)
                    self.Model.setP("EIU", ID + index, 0)
                else:
                    P = self.IUsP.get(ID, 1)
                    self.Model.setP(Type, ID + index, P)
                    self.Model.setP("NIU", ID + index, 0)
                self.Model.setP("NAMS", ID + index, 0)
                self.Model.setP("EAMS", ID + index, 0)
                self.Model.setP("NMSP", ID + index, 0)
                self.Model.setP("EMSP", ID + index, 0)
            self.Model.calculate_Next()
            self.Model.calculateAndPrint_TH()
            t = 0
            while t < 3:
                count = {}
                for i in range(4, self.N):
                    pNIU = self.Model.pNIU[i]
                    pEIU = self.Model.pEIU[i]
                    if pNIU < pEIU:
                        if count.get(i) is None:
                            count[i] = 1
                        else:
                            count[i] += 1
                    if count[i] == 2:
                        await self.IUChangeServer(i)
                if t == 3:
                    break
                self.Model.calculate_Next()
                self.Model.calculateAndPrint_TH()
                t += 1
