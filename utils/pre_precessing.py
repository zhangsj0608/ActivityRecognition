import numpy as np


def extend_to_maxsize(instances):
    rows = 0  # 行数是相等的
    max_columns = 0  # 列数取最大值
    for _, instance in instances:
        rows = instance.shape[0]
        if max_columns < instance.shape[1]:
            max_columns = instance.shape[1]
    print("rows:", rows, "cols:", max_columns)

    # 将instance扩展到最大的size,统一所有数据的size
    formalized_instances = []
    for label, instance in instances:
        column = instance.shape[1]
        if column < max_columns:
            zero_matrix = np.zeros(shape=(rows, max_columns - column))
            formalized_instance = np.concatenate((instance, zero_matrix), axis=1)
            formalized_instances.append((label, formalized_instance))
        else:
            formalized_instances.append((label, instance))

    return formalized_instances
