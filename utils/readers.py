import sys, os
import numpy as np
import time
import re
from utils.enums import *


def read_user_files(user_id):

    """ Read data files with respect to the userId
    :param user_id: strings from "01" to "51"
    :return: dict: {label: str}
    """

    cwd = os.getcwd()
    file_path = os.path.split(cwd)[0] + "\\resources\\"
    file_name_prefix = "p" + user_id
    labels = range(1, 6)
    user_matrices = {}

    for i, label in zip([".t"] * 5, labels):
        postfix = i + str(label)
        file_name = file_path + file_name_prefix + postfix
        with open(file_name) as file:
            content = file.read().strip()
            matrix = formulate_to_matrix(content)
            user_matrices[label] = matrix

    return user_matrices


def formulate_to_matrix(content):

    """
    Formulate the string of "content" to a numpy array
    :param content: String, 单个文件中的所有数据行，一个action的所有传感器event
    :return: numpy array，转化为完整矩阵的状态矩阵
    """

    state_list = {}  # key:sensor, value:[time_interval, state]
    start_time = -1.0
    for a_line in content.split("\n"):
        date, time_, sensor, state = a_line.split("\t")

        # time processing, compute time_interval and give the start_time as 0
        time_ = time_.split(".")[0]
        date_time = "".join([date, " ", time_])
        time_interval = 0  # the seconds distance from the start time
        if start_time == -1:
            start_time = int(time.mktime(time.strptime(date_time, "%Y-%m-%d %H:%M:%S")))
        else:
            time_interval = int(time.mktime(time.strptime(date_time, "%Y-%m-%d %H:%M:%S"))) - start_time
        # state processing, add {sensor:[(time_interval, state)]} to the state_list
        states = state_list.setdefault(sensor, [])
        states.append((time_interval, state))

    # 调用fill_into_array，将序列转换为矩阵
    state_matrix = form_matrix(state_list)
    return state_matrix


def form_matrix(state_list):

    """
    transform the list form to the numpy array form of the action
    :param state_list: dict, the sensor list with the form of {sensor: [(time_interval, state)]}
    :return: numpy array, the sensor matrix with respect to the time interval from the start of action
    """
    state_array = [[] for i in range(len(Sensor))]
    total_time_seconds = 0  # 记录总的时间长度（秒数）

    # process the state array, put the state value into the matrix
    for key, value in state_list.items():

        key_ = re.sub(r"-", "", key)
        try:
            row_index = Sensor[key_].value
        except KeyError:
            print("No such key: ", key_)
            continue

        start_state = 0  # 左边的状态
        start_time = 0  # 左边状态开始的时间

        for time_interval, state in value:
            try:
                current_state = State[state].value  # 当前状态，与下一时刻的状态相同，本次先处理一次
                next_state = State[state].value  # 下一时刻的左边状态，表示状态的延续
            except KeyError as err:  # 状态值是一个数字，而不是枚举类型
                current_state = state  # 当前状态，在当前的处理中，出现一次
                next_state = 0  # 下一时刻的左边状态，数值型的传感器状态不延续
            finally:
                state_array[row_index].extend([start_state] * (time_interval - start_time))  # 上一个状态延续至今
                state_array[row_index].append(current_state)  # 当前状态出现
                start_time = time_interval + 1  # 下一时刻左边状态的起始时间
                start_state = next_state  # 下一时刻的状态

        if len(state_array[row_index]) >= total_time_seconds:
            total_time_seconds = len(state_array[row_index])

    # add some values if the length of the array row is less than the total cols
    for row in state_array:
        if len(row) < total_time_seconds:
            row.extend([0] * (total_time_seconds - len(row)))

    return np.asarray(state_array, float)


if __name__ == "__main__":

    # 用户1的数据读取为矩阵
    data_01 = read_user_files("01")
    with open("test.txt", "w+") as f:
        for label, matrix in data_01.items():
            print("label:", label, "\nmatrix:", matrix, file=f)
