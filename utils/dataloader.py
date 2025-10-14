# # MIT License
# #
# # Copyright (c) 2025 Yibin Wu.
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
from .loadanddump import loadConfig as Config
from pathlib import Path
from pathlib import PurePosixPath 
from rosbags.highlevel import AnyReader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_types_from_msg, register_types
import numpy as np
from datetime import datetime
import os
from datetime import datetime
from tqdm import tqdm
class DataLoader:
    def __init__(self, config: Config, data_path: Path):
        self.config = config
        self.data_path = data_path
        self.add_types = {}
        go2_msg_folder = "./unitree_go/msg"
        for root, dirs, files in os.walk(go2_msg_folder):
            for file in files:
                if file.endswith(".msg"):
                    msgpath = Path(root) / file
                    msgdef  = msgpath.read_text(encoding="utf-8")

    
                    self.add_types.update(get_types_from_msg(msgdef, self._guess_msgtype(msgpath)))

        register_types(self.add_types)

    def _guess_msgtype(self ,path: Path) -> str:
        """Guess message type name from path."""
        name = path.relative_to(path.parents[2]).with_suffix("")
        if "msg" not in name.parts:
            name = name.parent / "msg" / name.name
        return str(PurePosixPath(name))
    
    
    def _rostime2gpst(self, cur_ts) -> float:
        sec = cur_ts // 10**9
        nanosec = cur_ts % 10**9

        
        gps_epoch = datetime(1980, 1, 6)   
        leap_seconds = 18

     
        ros_datetime = datetime.fromtimestamp(sec + nanosec*1e-9)

      
        delta = ros_datetime - gps_epoch
        total_seconds = delta.total_seconds()

       
        offset_seconds = (datetime.fromtimestamp(sec) - 
                        datetime.utcfromtimestamp(sec)).total_seconds()

        gps_week = int(total_seconds // (7 * 24 * 3600))
        sow = total_seconds % (7 * 24 * 3600) - offset_seconds + leap_seconds

        return sow

    def _rosMsg2np(self, lowstate_msg, lowstate_dim,cur_ts):

        robot_sensor_np = np.zeros((1, lowstate_dim), dtype=np.float64)

        

        ts = self._rostime2gpst(cur_ts)

        robot_sensor_np[0, 0] = ts

        robot_sensor_np[0, 1:4] = np.array(lowstate_msg.imu_state.gyroscope)
        robot_sensor_np[0, 4:7] = -np.array(lowstate_msg.imu_state.accelerometer)

        robot_sensor_np[0, 7] = lowstate_msg.motor_state[3].q
        robot_sensor_np[0, 8] = lowstate_msg.motor_state[4].q
        robot_sensor_np[0, 9] = lowstate_msg.motor_state[5].q
        robot_sensor_np[0, 10] = lowstate_msg.motor_state[0].q
        robot_sensor_np[0, 11] = lowstate_msg.motor_state[1].q
        robot_sensor_np[0, 12] = lowstate_msg.motor_state[2].q
        robot_sensor_np[0, 13] = lowstate_msg.motor_state[9].q
        robot_sensor_np[0, 14] = lowstate_msg.motor_state[10].q
        robot_sensor_np[0, 15] = lowstate_msg.motor_state[11].q
        robot_sensor_np[0, 16] = lowstate_msg.motor_state[6].q
        robot_sensor_np[0, 17] = lowstate_msg.motor_state[7].q
        robot_sensor_np[0, 18] = lowstate_msg.motor_state[8].q
        robot_sensor_np[0, 19] = lowstate_msg.motor_state[3].dq
        robot_sensor_np[0, 20] = lowstate_msg.motor_state[4].dq
        robot_sensor_np[0, 21] = lowstate_msg.motor_state[5].dq
        robot_sensor_np[0, 22] = lowstate_msg.motor_state[0].dq
        robot_sensor_np[0, 23] = lowstate_msg.motor_state[1].dq
        robot_sensor_np[0, 24] = lowstate_msg.motor_state[2].dq
        robot_sensor_np[0, 25] = lowstate_msg.motor_state[9].dq
        robot_sensor_np[0, 26] = lowstate_msg.motor_state[10].dq
        robot_sensor_np[0, 27] = lowstate_msg.motor_state[11].dq
        robot_sensor_np[0, 28] = lowstate_msg.motor_state[6].dq
        robot_sensor_np[0, 29] = lowstate_msg.motor_state[7].dq
        robot_sensor_np[0, 30] = lowstate_msg.motor_state[8].dq

        robot_sensor_np[0, 31] = lowstate_msg.foot_force[1]
        robot_sensor_np[0, 32] = lowstate_msg.foot_force[0]
        robot_sensor_np[0, 33] = lowstate_msg.foot_force[3]
        robot_sensor_np[0, 34] = lowstate_msg.foot_force[2]
        

        return robot_sensor_np

    def load_data(self):
        
        bag = AnyReader([self.data_path])
        bag.open()

        datatopic = self.config.topic

        msg_counts = bag.topics[datatopic].msgcount

        data_connections = [x for x in bag.connections if x.topic == datatopic]
        lowstate_msgs = bag.messages(connections=data_connections)

        lowstate_dim = 35  # 1 timestamp + 6 imu + 24 motor state + 4 foot force
        
        lowstate_np = np.zeros((msg_counts, lowstate_dim))
        

        for i in tqdm(range(msg_counts), total=msg_counts, desc="Loading bag", unit="msg"):
            try:
                lowstate_connection, cur_ts, lowstate_data = next(lowstate_msgs)  # ns

                try:
                    lowstate_msg = deserialize_cdr(
                        lowstate_data, lowstate_connection.msgtype
                    )
                except Exception as e:
                    print(f"Error deserialize_cdr {i}: {e}")


                lowstate_np[i] = self._rosMsg2np(
                    lowstate_msg, lowstate_dim,cur_ts
                )
                
                
            except Exception as e:
                print(f"Error processing message {i}: {e}")
                continue
        
        return lowstate_np

