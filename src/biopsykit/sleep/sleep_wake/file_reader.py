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
    Read in all the XML-files from the mesa-dataset

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
    """
    Read in all the csv-files from the actigraphy mesa-dataset.

    Parameters
    ----------
    file_path: str
        file path to the mesa folder with your actigraphy csv-files. Important: not the filename itself!

    Returns
    -------
    :dict: dict that contains a pandas.DataFrame for every subject

    """
    actigraphy = {}
    for i in range(20):  # try only with 20 datasets to lower memory consumption
        try:
            actigraphy[i] = pd.read_csv(file_path + '\mesa-sleep-'+ "{:04d}".format(i) + '.csv')
        except:
            pass

    return actigraphy


def read_r_point(file_path):
    """
    Read in all the csv-files from the r-roint mesa-dataset.

    Parameters
    ----------
    file_path: str
        file path to the mesa folder with your r-point csv-files. Important: not the filename itself!

    Returns
    -------
    :dict: dict that contains a pandas.DataFrame for every subject

    """
    r_point = {}
    for i in range(20):  # try only with 20 datasets to lower memory consumption
        try:
            r_point[i] = pd.read_csv(file_path + '\mesa-sleep-'+ "{:04d}".format(i) + '-rpoint.csv')
        except:
            pass

    return r_point


def xml_reader(file_path):
    psg_data = ET.parse(file_path)  #read xml files in a tree structure
    root = psg_data.getroot()       # root of the tree structure
    data = {}                       # tree structure:
                                    #    <PSGAnnotation>
    time = []                       #       <ScoredEvents>
    sleep = []                      #           <ScoredEvent>
    j = 0                           #               <EventType>Stages|Stages<EventType>
    for elem in root:               #               <EventConcept>Wake|0<EventConcept>
        for subelem in elem:        #               <Start>1500<Start>
            if (subelem[0].text == "Stages|Stages"):#<Duration>90<Duration>
                number = float(subelem[3].text)
                for i in range(0, int(number / 30)):  # Use dict!

                    time.append(j)
                    sleep.append(subelem[1].text)

                    j += 30     #every 30s we have one label, j is the time counting up without resetting.
                                # i is restting after each scored event
    data["time"] = time
    data["sleep"] = sleep
    df = pd.DataFrame.from_dict(data)   #build a dataframe with time and sleep/wake status

    return df



