import numpy as np
import datetime


# 解析卫星数据函数
def parse_satellite_data(data_str):
    satellites = []
    lines = data_str.strip().split("\n")
    for line in lines:
        parts = line.strip().split(",")
        utc_time_str = parts[0]  # UTC时间字符串
        prn = parts[1]  # 卫星编号
        coords = list(map(float, parts[2:5]))  # X, Y, Z 坐标
        clock_bias = float(parts[5])  # 钟差数据
        utc_time = datetime.datetime.strptime(utc_time_str, '%Y-%m-%d %H:%M:%S')
        satellites.append({
            "utc_time": utc_time,
            "prn": prn,
            "coords": coords,
            "clock_bias": clock_bias
        })
    return satellites


# 读取卫星数据函数
def read_satellite_data(filename):
    with open(filename, 'r') as file:
        data_str = file.read()
    return parse_satellite_data(data_str)


# 读取接收机坐标数据函数
def read_receiver_coords(filename):
    receiver_coords = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(": ")
            if len(parts) == 2:
                utc_time_str = parts[0]
                coords = list(map(float, parts[1].split(", ")))
                receiver_coords[utc_time_str] = coords
    return receiver_coords


# 转换矩阵函数，获取站点到地心的转换矩阵
def get_transformation_matrix(station_coords):
    X, Y, Z = station_coords
    lat = np.arctan2(Z, np.sqrt(X ** 2 + Y ** 2))
    lon = np.arctan2(Y, X)

    H = np.array([
        [-np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon), np.sin(lon)],
        [-np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon), -np.cos(lon)],
        [np.cos(lat), np.sin(lat), 0]
    ])
    return H


# 计算方位角的函数
def calculate_azimuth_angle(station_coords, satellite_coords):
    delta_x = satellite_coords[0] - station_coords[0]
    delta_y = satellite_coords[1] - station_coords[1]
    azimuth = np.arctan2(delta_y, delta_x)
    return np.degrees(azimuth) % 360


# 计算仰角的函数
def calculate_altitude_angle(station_coords, satellite_coords):
    H = get_transformation_matrix(station_coords)
    delta_coords = np.array(satellite_coords) - np.array(station_coords)
    transformed_coords = np.dot(H, delta_coords)

    delta_x_T, delta_y_T, delta_z_T = transformed_coords
    lat = np.arctan2(station_coords[2], np.sqrt(station_coords[0] ** 2 + station_coords[1] ** 2))
    lon = np.arctan2(station_coords[1], station_coords[0])

    numerator = np.cos(lat) * (np.cos(lon) * delta_x_T + np.sin(lon) * delta_y_T) + np.sin(lat) * delta_z_T
    denominator = np.sqrt(delta_x_T ** 2 + delta_y_T ** 2 + delta_z_T ** 2)

    h_T = (np.pi / 2) - np.arccos(numerator / denominator)
    return np.degrees(h_T)


# Klobuchar电离层模型计算函数
def calculate_klobuchar_iono_delay(station_coords, satellite_coords, utc_time, azimuth, elevation):
    alpha = [0.1770e-7, 0.2235e-7, -0.1192e-6, -0.1192e-6]
    beta = [0.1229e6, 0.1638e6, -0.1966e6, -0.2621e6]

    E = np.radians(elevation)
    psi = 0.0137 / (E + 0.11) - 0.022

    lat_u = np.degrees(np.arctan2(station_coords[2], np.sqrt(station_coords[0] ** 2 + station_coords[1] ** 2)))
    lon_u = np.degrees(np.arctan2(station_coords[1], station_coords[0]))
    phi_i = lat_u + psi * np.cos(np.radians(azimuth))
    if phi_i > 0.416:
        phi_i = 0.416
    elif phi_i < -0.416:
        phi_i = -0.416

    lambda_i = lon_u + (psi * np.sin(np.radians(azimuth)) / np.cos(np.radians(phi_i)))

    phi_m = phi_i + 0.064 * np.cos(np.radians(lambda_i - 1.617))

    t = 43200 * lambda_i + utc_time.hour * 3600 + utc_time.minute * 60 + utc_time.second
    if t > 86400:
        t -= 86400
    elif t < 0:
        t += 86400

    amp = np.polyval(alpha[::-1], phi_m)
    if amp < 0:
        amp = 0

    period = np.polyval(beta[::-1], phi_m)
    if period < 72000:
        period = 72000

    x = 2 * np.pi * (t - 50400) / period

    F = 1 + 16 * (0.53 - E / np.pi) ** 3

    if abs(x) < 1.57:
        T = F * (5e-9 + amp * (1 - (x ** 2) / 2 + (x ** 4) / 24))
    else:
        T = F * 5e-9

    c = 299792458
    iono_delay = T * c
    return iono_delay


# 主程序
def main():
    # 示例卫星数据文件路径
    filename_satellite = r"10组卫星位置钟差伪距(S).txt"
    # 示例接收机坐标数据文件路径
    filename_receiver = r"接收机伪距定位结果.txt"
    # 输出文件路径
    output_filename = r"iono_delay_results.txt"

    # 读取卫星数据和接收机坐标数据
    satellites = read_satellite_data(filename_satellite)
    receiver_coords = read_receiver_coords(filename_receiver)

    # 打开文件以写入电离层延迟结果
    with open(output_filename, 'w') as output_file:
        for data in satellites:
            utc_time = data["utc_time"]
            prn = data["prn"]
            sat_coords = data["coords"]

            # 查找与卫星时间匹配的接收机坐标
            utc_time_str = utc_time.strftime('%Y-%m-%d %H:%M:%S')
            if utc_time_str in receiver_coords:
                station_coords = receiver_coords[utc_time_str]
                azimuth = calculate_azimuth_angle(station_coords, sat_coords)
                elevation = calculate_altitude_angle(station_coords, sat_coords)
                iono_delay = calculate_klobuchar_iono_delay(station_coords, sat_coords, utc_time, azimuth, elevation)

                # 将结果写入文件，电离层延迟精确到小数点后三位
                output_file.write(f"{utc_time_str} {prn} : {iono_delay:.3f} 米\n")
            else:
                print(f"警告: 没有找到与卫星时间匹配的接收机坐标，卫星时间为 {utc_time_str}")


if __name__ == "__main__":
    main()
