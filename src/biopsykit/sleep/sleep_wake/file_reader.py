# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import time
from biopsykit.sleep.sleep_wake.base import _SleepWakeBase

"""
Read int the files of the mesa sleep-wake dataset. The filetypes are PSG, Actigraphy and R-points.
"""


def read_psg(file_path):
    psg = {}
    for i in range(20):  # try only with 20 datasets to lower memory consumption

        try:
            psg[i] = xml_reader(file_path + '\mesa-sleep-' + "{:04d}".format(i) + '-nsrr.xml')
        except:
            pass
            # print("{:04d}".format(i) + "not available")

    return psg


def read_actigraphy(file_path):
    actigraphy = {}
    for i in range(20):  # try only with 20 datasets to lower memory consumption
        try:
            actigraphy[i] = pd.read_csv(file_path + "{:04d}".format(i) + '.csv')
        except:
            pass

    return actigraphy


def read_r_point(file_path):
    r_point = {}
    for i in range(20):  # try only with 20 datasets to lower memory consumption
        try:
            r_point[i] = pd.read_csv(file_path + "{:04d}".format(i) + '-rpoint.csv')
        except:
            pass

    return r_point


def xml_reader(file_path):
    psg_data = ET.parse(file_path)
    root = psg_data.getroot()
    data = {}
    time = []
    sleep = []
    j = 0
    for elem in root:
        for subelem in elem:
            if (subelem[0].text == "Stages|Stages"):
                number = float(subelem[3].text)
                for i in range(0, int(number / 30)):  # Use dict!

                    time.append(j)
                    sleep.append(subelem[1].text)

                    j += 30

    data["time"] = time
    data["sleep"] = sleep
    df = pd.DataFrame.from_dict(data)

    return df


if __name__ == '__main__':
    time_1 = time.perf_counter()
    psg = read_psg('D:\Studium\Master\Masterarbeit\mesa\polysomnography/annotations-events-nsrr')

    time_2 = time.perf_counter()
    actigraphy = read_actigraphy()

    time_3 = time.perf_counter()
    r_point = read_r_point()

    time_4 = time.perf_counter()

    # time consumption (for all datasets --> about 2000; numbered from 0000 to 7000)
    print(time_2 - time_1)  # time for psg: 38.733s
    print(time_3 - time_2)  # time for actigraphy: 222.542s
    print(time_4 - time_3)  # time for r_point: 283.863s
