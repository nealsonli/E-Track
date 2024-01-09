#!/usr/bin/env python3
"""
Amit Kohli, Julien Martel, and Anastasios Angelopoulos
August 10, 2020
"""

import os
import glob
import struct
from collections import namedtuple
from PIL import Image


'Types of data'
Event = namedtuple('Event', 'polarity row col timestamp label')
Frame = namedtuple('Frame', 'row col img timestamp')


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))
    return imgs


def read_aerdat(filepath):
    """Reads an event file"""
    with open(filepath, mode='rb') as file:
        file_content = file.read()

    ''' Packet format'''
    packet_format = 'BHHI'  # pol = uchar, (x,y) = ushort, t = uint32
    packet_size = struct.calcsize('=' + packet_format)  # 16 + 16 + 8 + 32 bits => 2 + 2 + 1 + 4 bytes => 9 bytes
    num_events = len(file_content) // packet_size
    extra_bits = len(file_content) % packet_size

    '''Remove Extra Bits'''
    if extra_bits:
        file_content = file_content[0:-extra_bits]

    ''' Unpacking'''
    event_list = list(struct.unpack('=' + packet_format * num_events, file_content))
    event_list.reverse()

    return event_list


def get_path_info(path):
    """Parses the filename of the frames"""
    path = path.split('\\')[-1]
    filename = path.split('.')[0]
    path_parts = filename.split('_')
    index = int(path_parts[0])
    stimulus_type = path_parts[3]
    timestamp = int(path_parts[4])
    return {'index': index, 'row': int(path_parts[1]), 'col': int(path_parts[2]), 'stimulus_type': stimulus_type,
            'timestamp': timestamp}


class EyeDataset:
    """Manages both events and frames as a general data object"""
    def __init__(self, data_dir, user):
        """Initialize by creating a time ordered stack of frames and events"""
        self.data_dir = data_dir
        self.user = user

        self.frame_stack = []
        self.event_stack = []

    def __len__(self):
        return len(self.frame_stack) + len(self.event_stack)

    def __getitem__(self, index):
        """Determine if event or frame is next in time by peeking into both stacks"""
        event_timestamp = self.event_stack[-4]
        frame_timestamp = self.frame_stack[-1].timestamp
        frame_label = [self.frame_stack[-1].col, self.frame_stack[-1].row]

        'Returns selected data type'
        if event_timestamp < frame_timestamp:
            polarity = self.event_stack.pop()
            row = self.event_stack.pop()
            col = self.event_stack.pop()
            timestamp = self.event_stack.pop()
            event = Event(polarity, row, col, timestamp, frame_label)
            return event
        else:
            frame = self.frame_stack.pop()
            img = Image.open(frame.img).convert("L")
            frame = frame._replace(img=img)
            return frame

    def collect_data(self, eye=0):
        """Loads in data from the data_dir as filenames"""
        print('Loading Frames....')
        self.frame_stack = self.load_frame_data(eye)
        print('There are ' + str(len(self.frame_stack)) + ' frames \n')
        print('Loading Events....')
        self.event_stack = self.load_event_data(eye)
        print('There are ' + str(int(len(self.event_stack) / 4)) + ' events \n')

    def load_frame_data(self, eye):
        filepath_list = []
        user_name = "user" + str(self.user)
        img_dir = os.path.join(self.data_dir, user_name, str(eye), 'frames')
        print('img_dir: ' + img_dir + '\n')
        img_filepaths = list(glob_imgs(img_dir))
        img_filepaths.sort(key=lambda name: get_path_info(name)['index'])
        img_filepaths.reverse()
        for fpath in img_filepaths:
            path_info = get_path_info(fpath)
            frame = Frame(path_info['row'], path_info['col'], fpath, path_info['timestamp'])
            filepath_list.append(frame)
        return filepath_list

    def load_event_data(self, eye):
        user_name = "user" + str(self.user)
        event_file = os.path.join(self.data_dir, user_name, str(eye), 'events.aerdat')
        print('event_file: ' + event_file + '\n')
        filepath_list = read_aerdat(event_file)
        return filepath_list
