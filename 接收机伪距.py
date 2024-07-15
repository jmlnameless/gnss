import numpy as np
from collections import defaultdict
from datetime import datetime
from matplotlib import pyplot as plt, rcParams
from 电离层误差参数计算 import main


def split(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("错误: 文件未找到")
        return None

    data_by_time = defaultdict(dict)
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 8:
            continue
        date_time = parts[0]
        satellite_id = parts[1]
        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])
        pseudo_range = float(parts[5])
        clock_bias = float(parts[6])
        weight = float(parts[7])
        data_by_time[date_time][satellite_id] = [x, y, z, pseudo_range, clock_bias, weight]
    return data_by_time


def read_iono_delay(file_path):
    iono_delay = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            timestamp = f"{parts[0]} {parts[1]}"
            satellite = parts[2].strip(':')
            delay = float(parts[4])
            if timestamp not in iono_delay:
                iono_delay[timestamp] = {}
            iono_delay[timestamp][satellite] = delay
    return iono_delay


def calculate_recieve(txt, xyzcoord):
    x0, y0, z0 = xyzcoord
    X = np.zeros((4, 1))
    count = 0
    c = 299792458  # 光速，单位：米/秒

    while True:
        count += 1
        listA = []
        listL = []
        listP = []

        for item in txt:
            x, y, z, pseudo_range, clock_bias, weight = txt[item]

            distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

            l = (x - x0) / distance
            m = (y - y0) / distance
            n = (z - z0) / distance

            listA.append([-l, -m, -n, 1])
            listL.append([pseudo_range - distance + c * clock_bias + X[3, 0]])
            listP.append(weight)

        if len(listA) == 0 or len(listL) == 0:
            print("错误: 没有足够的卫星数据来计算接收机坐标。")
            return None

        A = np.mat(listA)
        L = np.mat(listL)
        P = np.mat(np.diag(listP))

        try:
            X = np.linalg.inv(A.T * P * A) * (A.T * P * L)
        except np.linalg.LinAlgError:
            print("错误: 矩阵不可逆，无法计算。")
            return None

        x0 += X[0, 0]
        y0 += X[1, 0]
        z0 += X[2, 0]

        if all(abs(X[i, 0]) < 0.001 for i in range(3)) or count > 100:
            result = [x0, y0, z0, count, len(A)]
            break
    return result


XYZ = [1, 1, 1]
data_by_time12 = split('12组卫星位置钟差伪距(S).txt')
data_by_time10 = split('10组卫星位置钟差伪距(S).txt')
P12 = [-2615466.6, 4732793.7, 3371105.6]
P10 = [-2615265.2433, 4732896.7836, 3371103.4755]
dx_list = []
dy_list = []
dz_list = []
distances_raw = []
distances_diff = []
timestamps = []
results = []
distances_corrected = []

# 获取共同的时间戳集合，并按时间排序
timestamps_12_set = set(data_by_time12.keys())
timestamps_10_set = set(data_by_time10.keys())
common_timestamps = sorted(timestamps_12_set.intersection(timestamps_10_set))

for timestamp in common_timestamps:
    satellite_data_12 = data_by_time12[timestamp]
    satellite_data_10 = data_by_time10[timestamp]

    # 计算 12 组数据修正后的 xyz 坐标
    xyz12 = calculate_recieve(satellite_data_12, XYZ)
    dx = xyz12[0] - P12[0]
    dy = xyz12[1] - P12[1]
    dz = xyz12[2] - P12[2]

    # 计算 10 组数据的修正后 xyz 坐标
    xyz_10 = calculate_recieve(satellite_data_10, XYZ)
    xyz_10 = xyz_10[:3]  # 确保只包含坐标
    xyz_10_new = [xyz_10[0] - dx, xyz_10[1] - dy, xyz_10[2] - dz]

    # 计算欧式距离
    distance_raw = np.linalg.norm(np.array(xyz_10) - np.array(P10))
    distance_diff = np.linalg.norm(np.array(xyz_10_new) - np.array(P10))
    results.append({
        'timestamp': timestamp,
        'xyz_10': xyz_10_new
    })
    # 存储时间和距离
    timestamps.append(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))
    distances_raw.append(distance_raw)
    distances_diff.append(distance_diff)
    print(f"{timestamp}: 粗坐标距离 = {distance_raw}, 差分后距离 = {distance_diff}")

# 设置字体以支持中文
rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
# 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(timestamps, distances_raw, label='粗坐标P10与精确P10坐标的欧式距离')
# plt.plot(timestamps, distances_diff, label='差分后坐标P10与精确P10坐标的欧式距离')
plt.xlabel('时间')
plt.ylabel('欧式距离 (米)')
plt.legend()
plt.title('粗坐标与精确坐标的欧式距离')
plt.show()

# 将结果写入文件
output_filename = '接收机伪距定位结果.txt'
with open(output_filename, 'w') as f:
    for result in results:
        timestamp = result['timestamp']
        xyz_10 = result['xyz_10']
        x, y, z = xyz_10
        f.write(f"{timestamp}: {x}, {y}, {z}\n")

print(f"结果已写入到文件 {output_filename}")

main()


def calculate_receiver_position_with_iono_delay(satellite_data, initial_coords, iono_delay, timestamp):
    x0, y0, z0 = initial_coords
    X = np.zeros((4, 1))
    count = 0
    c = 299792458  # Speed of light in m/s

    while True:
        count += 1
        A_list = []
        L_list = []
        P_list = []

        for sat_id, data in satellite_data.items():
            x, y, z, pseudo_range, clock_bias, weight = data

            distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

            l = (x - x0) / distance
            m = (y - y0) / distance
            n = (z - z0) / distance

            iono_correction = iono_delay.get(timestamp, {}).get(sat_id, 0)

            A_list.append([-l, -m, -n, 1])
            L_list.append([pseudo_range - distance + c * clock_bias + X[3, 0] - iono_correction])
            P_list.append(weight)

        if len(A_list) == 0 or len(L_list) == 0:
            print("Error: Not enough satellite data to calculate receiver position.")
            return None

        A = np.mat(A_list)
        L = np.mat(L_list)
        P = np.mat(np.diag(P_list))

        try:
            X = np.linalg.inv(A.T * P * A) * (A.T * P * L)
        except np.linalg.LinAlgError:
            print("Error: Matrix is not invertible, cannot compute.")
            return None

        x0 += X[0, 0]
        y0 += X[1, 0]
        z0 += X[2, 0]

        if all(abs(X[i, 0]) < 0.001 for i in range(3)) or count > 100:
            result = [x0, y0, z0, count, len(A)]
            break

    return result


iono_delay = read_iono_delay('iono_delay_results.txt')

distances_raw = []
timestamps = []
# 计算原始和修正后的距离
for timestamp in common_timestamps:
    satellite_data_10 = data_by_time10[timestamp]

    # 计算修正电离层误差前的坐标
    xyz_raw = calculate_recieve(satellite_data_10, XYZ)
    if xyz_raw:
        xyz_raw = xyz_raw[:3]  # 确保只包含坐标

        # 计算修正电离层误差后的坐标
        xyz_corrected = calculate_receiver_position_with_iono_delay(satellite_data_10, XYZ, iono_delay, timestamp)
        if xyz_corrected:
            xyz_corrected = xyz_corrected[:3]  # 确保只包含坐标

            # 计算距离
            distance_raw = np.linalg.norm(np.array(xyz_raw) - np.array(P10))
            distance_corrected = np.linalg.norm(np.array(xyz_corrected) - np.array(P10))
            # 存储时间和距离
            timestamps.append(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))
            distances_raw.append(distance_raw)
            distances_corrected.append(distance_corrected)

# 设置字体以支持中文
rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(timestamps, distances_raw, label='修正电离层误差前的距离')
plt.plot(timestamps, distances_corrected, label='修正电离层误差后的距离')
plt.xlabel('时间')
plt.ylabel('欧式距离 (米)')
plt.legend()
plt.title('修正电离层误差前后与10组真实坐标的欧式距离')
plt.show()
