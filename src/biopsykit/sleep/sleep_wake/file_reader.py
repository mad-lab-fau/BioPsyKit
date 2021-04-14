"""
Read in the files of the mesa sleep-wake dataset. The filetypes are PSG, Actigraphy and R-points.
"""
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import time
from biopsykit.sleep.sleep_wake.base import _SleepWakeBase




def read_psg(file_path):
    """
    Read in the XML-files from the mesa-dataset

    Parameters
    ----------
    file_path: str
        file path to the mesa folder with your XML-files. Important: not the filename itself!

    Returns
    -------
    :dict: dict that contains a pandas.DataFrame for every subject
    """

    psg = {}
    for i in range(20):  # try only with 20 datasets to lower memory consumption

        try: #look if a dataset exists
            psg[i] = xml_reader(file_path + '\mesa-sleep-' + "{:04d}".format(i) + '-nsrr.xml')
        except:
            pass
            # print("{:04d}".format(i) + "not available")

    return psg


def read_actigraphy(file_path):
    actigraphy = {}
    for i in range(20):  # try only with 20 datasets to lower memory consumption
        try:
            actigraphy[i] = pd.read_csv(file_path + '\mesa-sleep-'+ "{:04d}".format(i) + '.csv')
        except:
            pass

    return actigraphy


def read_r_point(file_path):
    r_point = {}
    for i in range(20):  # try only with 20 datasets to lower memory consumption
        try:
            r_point[i] = pd.read_csv(file_path + '\mesa-sleep-'+ "{:04d}".format(i) + '-rpoint.csv')
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



