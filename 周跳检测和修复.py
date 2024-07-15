from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams

# 设置字体以支持中文
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
file_path = '3291461169C.24o'
output_file_path = 'N（未做历元差）.txt'


# 读取o文件中的数据
def read_C1L1(file_path):
    data = []
    c1c = None
    l1c = None

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        header = True
        time = None

        for line in lines:
            if header:
                if "END OF HEADER" in line:
                    header = False
                continue

            if line.startswith('>'):
                raw_time = line[1:29].strip()  # 只取前28个字符，这样可以去掉多余的部分
                time = datetime.strptime(raw_time, '%Y %m %d %H %M %S.%f0')  # 解析时间
            else:
                prn = line[:3].strip()
                if prn and len(line) > 76:  # 假设列长度至少为76
                    # 提取 C1C 和 L1C 数据
                    c1c = line[5:17].strip().split()[0] if len(line) > 5 else ''
                    l1c = line[51:76].strip().split()[0] if len(line) > 51 else ''
                    l1_2 = float(c1c) / 0.19029367
                    n = float(l1c) - l1_2
                    if c1c and l1c:
                        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S')
                        data.append([formatted_time, prn, c1c, l1c, l1_2, n])

    return data


def write_data_to_file(data, output_file_path):
    with open(output_file_path, 'w') as file:
        for entry in data:
            file.write('\t'.join(map(str, entry)) + '\n')


total_data = read_C1L1(file_path)
write_data_to_file(total_data, output_file_path)


# print(f"Data has been written to {output_file_path}")


def detect_jump(data):
    df = pd.DataFrame(data, columns=['formatted_time', 'prn', 'c1c', 'l1c', 'l1_2', 'n'])
    df['formatted_time'] = pd.to_datetime(df['formatted_time'])
    groups = df.groupby('prn')

    results = []
    for prn, group in groups:
        group = group.sort_values(by='formatted_time')
        group['l1_2'] = group['c1c'].astype(float) / 0.19029367
        group['delta_n'] = group['l1c'].astype(float) - group['l1_2']
        group['delta_n_shifted'] = group['delta_n'].shift(-1)
        group['N'] = group['delta_n_shifted'].astype(float) - group['delta_n'].astype(float)
        group['N_shifted'] = group['N'].shift(1)

        # 根据阈值将异常值设置为NaN
        threshold = 1000
        group.loc[(group['N_shifted'] > threshold) | (group['N_shifted'] < -threshold), 'N_shifted'] = np.nan

        # 去除第一个不完整的N值
        valid_entries = group[['formatted_time', 'prn', 'delta_n', 'delta_n_shifted', 'N', 'N_shifted']].iloc[
                        1:].dropna()

        results.extend(valid_entries.values.tolist())

    write_data_to_file(results, '周跳检测N（去掉第一行）.txt')
    return results


def polynomial_regression_predict(df, degree=3):
    df = df.copy()  # 确保在副本上操作
    # 将时间转变成数值
    df.loc[:, 'ordinal_time'] = df['formatted_time'].apply(lambda x: x.toordinal())
    X = df[['ordinal_time']].values
    y = df['N_shifted'].values

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    poly_reg_model = LinearRegression()

    results = []
    M = 10
    for i in range(len(X) - M):
        X_train = X_poly[i:i + M]
        y_train = y[i:i + M]
        poly_reg_model.fit(X_train, y_train)
        X_test = X_poly[i + M].reshape(1, -1)
        y_test = y[i + M]

        y_pred = poly_reg_model.predict(X_test)[0]

        # 计算 V 和 delta
        y_train_pred = poly_reg_model.predict(X_train)
        V = y_train - y_train_pred
        n = 3
        delta = np.sqrt((V.T @ V) / (M - n + 1))

        jump = abs(y_test - y_pred) > 3 * delta

        results.append([df.iloc[i + M]['formatted_time'], df.iloc[i + M]['prn'], y_test, y_pred, jump])

    return results


total_data = read_C1L1(file_path)
write_data_to_file(total_data, output_file_path)
jump_data = detect_jump(total_data)

# 将跳变检测数据转化为 DataFrame
df_jump = pd.DataFrame(jump_data, columns=['formatted_time', 'prn', 'delta_n', 'delta_n_shifted', 'N', 'N_shifted'])
# 仅选择 GPS 卫星（编号以 'G' 开头）
df_jump_gps = df_jump[df_jump['prn'].str.startswith('G17')]

# 检测周跳
detection_results = polynomial_regression_predict(df_jump_gps)


def correct_predictions(results):
    corrected_results = []
    for result in results:
        formatted_time, prn, N_shifted, N_pred, jump = result

        # 如果 N_shifted 超过了 N_pred，则用 N_pred 替代 N_shifted
        if N_shifted > N_pred:
            N_shifted = N_pred

        corrected_results.append([formatted_time, prn, N_shifted, N_pred, jump])

    return corrected_results


# 修正检测结果
corrected_results = correct_predictions(detection_results)


# 遍历卫星
# 获取所有唯一的 GPS 卫星编号
# 绘制检测结果
def plot_detection_results(results):
    df_results = pd.DataFrame(results, columns=['formatted_time', 'prn', 'N_shifted', 'N_pred', 'jump'])
    for prn in df_results['prn'].unique():
        plt.figure()
        df_prn = df_results[df_results['prn'] == prn]
        plt.plot(df_prn['formatted_time'], df_prn['N_shifted'], label=f'卫星 {prn} 真实值')
        plt.plot(df_prn['formatted_time'], df_prn['N_pred'], label=f'卫星 {prn} 预测值')
        plt.xlabel('时间')
        plt.ylabel('N_shifted')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.title('周跳检测结果')
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)
        plt.show()


# 绘制检测结果
plot_detection_results(detection_results)


def plot_corrected_results(results):
    df_results = pd.DataFrame(results, columns=['formatted_time', 'prn', 'N_shifted', 'N_pred', 'jump'])
    for prn in df_results['prn'].unique():
        plt.figure()
        df_prn = df_results[df_results['prn'] == prn]
        plt.plot(df_prn['formatted_time'], df_prn['N_shifted'], label=f'卫星 {prn} 预测值')
        plt.plot(df_prn['formatted_time'], df_prn['N_pred'], label=f'卫星 {prn} 修正值')
        plt.xlabel('时间')
        plt.ylabel('N_shifted')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.title('周跳修正结果')
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.0)
        plt.show()


# 绘制修正后的图像
plot_corrected_results(corrected_results)
