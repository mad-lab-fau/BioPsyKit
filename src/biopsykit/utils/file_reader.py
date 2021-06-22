"""
Read in the files of the mesa sleep-wake dataset. The filetypes are PSG, Actigraphy and R-points.
"""
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import time
from biopsykit.sleep.sleep_wake_detection.algorithms._base import _SleepWakeBase
import re
from pathlib import Path



def read_all_psg(folder_path):
    """
    Read in all the XML-files from the mesa-dataset

    Parameters
    ----------
    folder_path: str
        file path to the mesa folder with your XML-files. Important: not the filename itself!

    Returns
    -------
    :dict:
        dict that contains a pandas.DataFrame for every subject
    """
    print("start reading psg-data")
    psg = {}
    path_list = list(Path(folder_path).glob('*.xml'))
    #mesa_list = re.findall("(\d{4})", ''.join(str(p))) -->if mesa_id_list wanted
    for data_name in path_list:
        i = re.findall("(\d{4})",data_name.name)[0]
        #i = re.search("(\d{4})",path.name).__getitem__(0)
        psg[i] = xml_reader(folder_path + data_name.name)
        print('file '+str(i)+' readed in!')

    print('Reading psg-data finished')
    return psg



def read_single_psg(file_path, mesaid):
    """
    Read in all the XML-files from the mesa-dataset

    Parameters
    ----------
    file_path: str
        file path to the mesa folder with your XML-files. Important: not the filename itself!

    Returns
    -------
    :dict:
        dict that contains a pandas.DataFrame for every subject
    """

    try: #look if a dataset exists
        psg = xml_reader(file_path + '\polysomnography/annotations-events-nsrr\mesa-sleep-' + "{:04d}".format(mesaid) + '-nsrr.xml')
    except:
        raise ImportError("Dataset don't exist")

    return psg


def read_all_actigraphy(folder_path):
    """
    Read in all the csv-files from the actigraphy mesa-dataset.

    Parameters
    ----------
    folder_path: str
        file path to the mesa folder with your actigraphy csv-files. Important: not the filename itself!

    Returns
    -------
    dict:
        dict that contains a pandas.DataFrame for every subject

    """
    print("Start reading Actigraphy-data!")
    actigraphy = {}
    path_list = list(Path(folder_path).glob('*.csv'))
    
    for data_name in path_list:
        i = re.findall("(\d{4})",data_name.name)[0]
        actigraphy[i] = pd.read_csv(folder_path + data_name.name)
        print('file ' + str(i) + ' readed in!')

    print("Reading Actigraphy data finished!")
    return actigraphy



def read_single_actigraphy(file_path,mesaid):
    """
    Read in all the csv-files from the actigraphy mesa-dataset.

    Parameters
    ----------
    file_path: str
        file path to the mesa folder with your actigraphy csv-files. Important: not the filename itself!

    Returns
    -------
    dict:
        dict that contains a pandas.DataFrame for every subject

    """

    try:
        actigraphy = pd.read_csv(file_path + '/actigraphy\mesa-sleep-'+ "{:04d}".format(mesaid) + '.csv')

    except ImportError as e:
        raise ImportError("Dataset don't exist") from e


    return actigraphy





def read_all_r_point(folder_path):
    """
    Read in all the csv-files from the r-roint mesa-dataset.

    Parameters
    ----------
    folder_path: str
        file path to the mesa folder with your r-point csv-files. Important: not the filename itself!

    Returns
    -------
    dict:
        dict that contains a pandas.DataFrame for every subject

    """
    print("Start reading r-point-data!")
    r_point = {}

    path_list = list(Path(folder_path).glob('*.csv'))
    for data_name in path_list:
        i = re.findall("(\d{4})", data_name.name)[0]
        r_point[i] = pd.read_csv(folder_path + data_name.name)
        print('file ' + str(i) + ' readed in!')


    print("Reading r-point-data finished!")
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




def read_clean_data(folder_path):
    data = {}
    path_list = list(Path(folder_path +"/clean_data/").glob('*.csv'))
    for data_name in path_list:
        i = re.findall("(\d{4})", data_name.name)[0]
        subj_data = pd.read_csv(folder_path +"/clean_data/" + data_name.name)
        subj_data['linetime'] = pd.to_datetime(subj_data['linetime'], format = "%H:%M:%S").dt.time
        data[i] = subj_data

    return data

