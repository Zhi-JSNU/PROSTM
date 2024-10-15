# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import psutil
import asyncio
import logging
import time


class ResourceMonitor:
    pid_cpuinfo = {}

    _time = getattr(time, 'monotonic', time.time)
    sum_cpus = psutil.cpu_count() or 1

    def timer(self):
        return self._time() * self.sum_cpus

    def __init__(self, ServerName: str):

        self.loggers = {
            'memory': self.setup_logger('memory_logger', ServerName + '_memory.log'),
            'bandwidth': self.setup_logger('bandwidth_logger', ServerName + '_bandwidth.log'),
            'cpu': self.setup_logger('cpu_logger', ServerName + '_cpu.log'),
            'init': self.setup_logger('init_logger', 'init.log'),
        }

    def setup_logger(self, name, file_name, level=logging.INFO):

        logger = logging.getLogger(name)
        logger.setLevel(level)
        handler = logging.FileHandler(file_name, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s,%(message)s'))
        handler.setLevel(level)
        logger.addHandler(handler)
        return logger

    async def monitor_program(self):

        pid = psutil.Process().pid
        while True:
            try:
                memory_usage, (sent_diff, recv_diff), cpu_usage = await asyncio.gather(
                    self.check_memory_usage(pid),
                    self.check_network_change(pid),
                    self.check_cpu_usage(pid)
                )
                self.loggers['memory'].info(f"{memory_usage:.2f}")
                self.loggers['bandwidth'].info(
                    f"{sent_diff / 1024:.2f},{recv_diff / 1024:.2f}")
                self.loggers['cpu'].info(f"{cpu_usage:.2f}")

            except Exception as e:
                self.loggers['init'].error(f"Error in monitor_network_usage: {str(e)}")
            await asyncio.sleep(1)

    async def check_memory_usage(self, pid):
        """检查内存使用量"""
        process = psutil.Process(pid)
        memory_info = process.memory_percent()
        memory_info *= 100
        round(memory_info, 2)
        return memory_info

    async def check_network_change(self, pid):

        process = psutil.Process(pid)
        connections = process.connections(kind='all')
        bytes_sent = sum(conn.raddr[1] for conn in connections if conn.raddr)
        bytes_recv = sum(conn.laddr[1] for conn in connections if conn.laddr)
        sent_diff = bytes_sent
        recv_diff = bytes_recv
        # 更新旧的字节数
        return sent_diff, recv_diff

    async def check_cpu_usage(self, pid):

        p = psutil.Process(pid)
        tt = p.cpu_times()
        st0, tt0_0, tt0_1 = self.pid_cpuinfo.get(pid, (0, 0, 0))
        st1, tt1_0, tt1_1 = self.timer(), tt.user, tt.system

        delta_proc = (tt1_0 - tt0_0) + (tt1_1 - tt0_1)
        delta_time = st1 - st0
        try:
            cpus_percent = ((delta_proc / delta_time) * 100)
        except:
            cpus_percent = 0.0
        self.pid_cpuinfo[pid] = [st1, tt1_0, tt1_1]
        cpus_percent *= 100
        return cpus_percent

    async def p_record(self, message):
        self.loggers['p'].info(message)
