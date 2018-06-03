from enum import Enum

class Sensor(Enum):

    M01 = 0
    M02 = 1
    M03 = 2
    M04 = 3
    M05 = 4
    M06 = 5
    M07 = 6
    M08 = 7
    M09 = 8
    M10 = 9
    M11 = 10
    M12 = 11
    M13 = 12
    M14 = 13
    M15 = 14
    M16 = 15
    M17 = 16
    M18 = 17
    M19 = 18
    M20 = 19
    M21 = 20
    M22 = 21
    M23 = 22
    M24 = 23
    M25 = 24
    M26 = 25
    I01 = 26
    I02 = 27
    I03 = 28
    I04 = 29
    I05 = 30
    I06 = 31
    I07 = 32
    I08 = 33
    D01 = 34
    AD1A = 35
    AD1B = 36
    AD1C = 37
    asterisk = 38


class State(Enum):
    ON = 1
    OFF = 0
    START = 1
    END = 0
    ABSENT = 1
    OPEN = 1
    CLOSE = 0


if __name__ == "__main__":
    sen1 = Sensor["D01"]
    print(sen1.value)
    print(len(Sensor))