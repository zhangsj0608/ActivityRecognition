import sys, os
import numpy as np
import time
import re
from utils.enums import *


def read_user_files(userId):

    """ Read data files with respect to the userId
    :param userId: strings from "01" to "51"
    :return: dict: {label: str}
    """

    cwd = os.getcwd()
    file_path = os.path.split(cwd)[0] + "\\resources\\"
    file_name_prefix = "p" + userId
    labels = range(1, 5)
    result_data = {}

    for i, label in zip([".t"] * 5, labels):
        postfix = i + str(label)
        file_name = file_path + file_name_prefix + postfix
        with open(file_name) as f:
            content = f.read().strip()
            result_data[label] = content

    return result_data


def formulate_to_matrix(content):

    """
    Formulate the string of "content" to a numpy array
    :param content: String, a line made of all the texts from an action data file
    :return: numpy array of the action
    """

    state_list = {}
    start_time = -1.0
    for line in content.split("\n"):
        date, time_, sensor, state = line.split("\t")

        ## time processing, compute time_interval and give the start_time as 0
        time_ = time_.split(".")[0]
        date_time = "".join([date, " ", time_])
        time_interval = 0 ## the seconds distance from the start time
        if start_time == -1:
            start_time = int(time.mktime(time.strptime(date_time, "%Y-%m-%d %H:%M:%S")))
        else:
            time_interval = int(time.mktime(time.strptime(date_time, "%Y-%m-%d %H:%M:%S"))) - start_time
        ## state processiong, add {sensor:[(time_interval, state)]} to the state_list
        states = state_list.setdefault(sensor, [])
        states.append((time_interval, state))

    state_array = fill_into_array(state_list)
    return state_array


def fill_into_array(state_list):

    """
    transform the list form to the numpy array form of the action
    :param state_list: dict, the sensor list with the form of {sensor: [(time_interval, state)]}
    :return: numpy array, the sensor matrix with respect to the time interval from the start of action
    """
    state_array = [[] for i in range(len(Sensor))]
    total_time_in_seconds = 0

    ## process the state array, put the state value into the matrix
    for key, value in state_list.items():

        key_ = re.sub(r"-", "", key)
        row_index = Sensor[key_].value

        start_state = 0
        start_time = 0

        for time_interval, state in value:
            state_array[row_index].extend([start_state] * (time_interval - start_time))
            start_time = time_interval
            start_state = State[state].value
            if time_interval > total_time_in_seconds:
                total_time_in_seconds = time_interval

    ## add some values if the length of the array row is less than the total cols
    for row in state_array:
        if len(row) < total_time_in_seconds:
            row.extend([0] * (total_time_in_seconds - len(row)))

    return np.asarray(state_array, float)


if __name__ == "__main__":

    data_01 = read_user_files("01")
    state_array = formulate_to_matrix(data_01[1])
    #print(state_list)
    #state_array = fill_into_array(state_list)
    print(state_array.shape)
    f = open("test.txt", "w+")
    for line in state_array:
        print(line, file=f)
    f.close()
