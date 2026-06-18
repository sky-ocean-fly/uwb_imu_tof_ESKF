import socket
import struct
import threading
import time
import numpy as np
from typing import Dict, List, Optional

class NatNetClient:
    def __init__(self, server_ip: str = "192.168.50.208", 
                 multicast_address: str = "239.255.42.99"):
        """
        初始化NatNet客户端
        Args:
            server_ip: Motive软件运行的电脑IP
            multicast_address: NatNet组播地址
        """
        self.server_ip = server_ip
        self.multicast_address = multicast_address
        self.port = 1511  # NatNet数据端口
        
        self.rigid_bodies = {}  # 存储刚体位姿
        self.running = False
        self.thread = None
        
    def connect(self):
        """连接到NatNet服务器"""
        # 创建UDP套接字
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 绑定到端口
        self.socket.bind(('', self.port))
        
        # 加入组播组
        mreq = struct.pack("4sl", socket.inet_aton(self.multicast_address), socket.INADDR_ANY)
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        print(f"连接到NatNet服务器 {self.server_ip}...")
        
    def start(self):
        """开始接收数据"""
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def _receive_loop(self):
        """数据接收循环"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(65536)  # 接收数据
                if len(data) > 0:
                    self._parse_packet(data)
            except Exception as e:
                print(f"接收数据错误: {e}")
                
    def _parse_packet(self, data: bytes):
        """解析NatNet数据包"""
        # 这里简化了实际解析过程
        # 实际需要根据NatNet协议文档解析数据包结构
        
        # 示例：解析刚体数据（简化版）
        offset = 0
        
        # 1. 解析数据包头
        # 实际实现需要根据NatNet SDK的DataDescriptions进行解析
        
        # 临时示例：打印原始数据大小
        print(f"收到数据包: {len(data)} 字节")
        
    def get_rigid_body_pose(self, rigid_body_id: int) -> Optional[Dict]:
        """
        获取指定刚体的位姿
        Returns: {
            'position': [x, y, z],
            'rotation': [qx, qy, qz, qw],
            'timestamp': float,
            'tracking_valid': bool
        }
        """
        if rigid_body_id in self.rigid_bodies:
            return self.rigid_bodies[rigid_body_id]
        return None
        
    def stop(self):
        """停止接收"""
        self.running = False
        if self.thread:
            self.thread.join()
        self.socket.close()

# 使用示例
if __name__ == "__main__":
    # 创建客户端
    client = NatNetClient(server_ip="192.168.50.208")
    
    try:
        # 连接并开始接收
        client.connect()
        client.start()
        
        # 主循环：获取数据并用于控制
        while True:
            # 获取无人机位姿（假设刚体ID为1）
            # print("123\n")
            pose = client.get_rigid_body_pose(29)
            
            if pose and pose['tracking_valid']:
                # 提取位置和姿态
                x, y, z = pose['position']
                qx, qy, qz, qw = pose['rotation']
                
                # 这里可以加入控制算法
                print(f"位置: [{x:.3f}, {y:.3f}, {z:.3f}]")
                print(f"姿态: [{qw:.3f}, {qx:.3f}, {qy:.3f}, {qz:.3f}]")
                
                # 计算控制指令（示例）
                # control_command = calculate_control(x, y, z, qx, qy, qz, qw)
                # send_to_drone(control_command)
            
            time.sleep(0.01)  # 100Hz循环
            
    except KeyboardInterrupt:
        print("程序停止")
    finally:
        client.stop()