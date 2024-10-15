# !/usr/bin/env python3
# -*- coding: utf-8 -*-


import asyncio
import os
from network import iu, server, drd
from utils.tools import load_settings


def iu_main(IUName: str):
    settings = load_settings(os.getcwd())
    ip = settings['ip']
    port = settings['port']
    serverip = settings['ServerIP']
    serverport = settings['ServerPort']
    IU = iu.IUManager()
    IU.setmonitor(IUName)
    asyncio.run(IU.start_iu(ip, port, serverip, serverport))


def server_main(ServerName: str):
    settings = load_settings(os.getcwd())
    role = settings['role']
    ip = settings['ip']
    port = settings['port']
    DRDip = settings['DRDIP']
    DRDport = settings['DRDPort']
    Server = server.ServerManager()
    Server.setmonitor(ServerName)
    asyncio.run(Server.start_server(role, ip, port, DRDip, DRDport))


def drd_main(DRDName: str):
    settings = load_settings(os.getcwd())
    ip = settings['ip']
    port = settings['port']
    Type = settings['Type']
    drdt = drd.DRDManager()
    drdt.setmonitor(DRDName)
    if Type == "Model":
        N = settings['N']
        T = settings['T']
        theta1 = settings["theta1"]
        theta2 = settings["theta2"]
        theta3 = settings["theta3"]
        theta4 = settings["theta4"]
        delta, k, u, v, X, Y = settings["delta"], settings["k"], settings["u"], settings["v"], settings["X"], settings[
            "Y"]
        asyncio.run(drdt.start_server(Type, ip, port, N, T, theta1, theta2, theta3, theta4, delta, k, u, v, X, Y))

    else:
        asyncio.run(drdt.start_server(Type, ip, port))
