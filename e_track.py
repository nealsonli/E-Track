#!/usr/bin/env python3
"""
Nealson Li, Ashwin Bhat, Arijit Raychowdhury
June 11, 2023
"""

import pickle
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import cv2 as cv
from skimage.morphology import disk

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from multiprocessing import freeze_support
from collections import namedtuple, Counter
from copy import deepcopy
import timeit

from math import pi, cos, sin
from util.ellipse import LsqEllipse
from unet import custom_objects

from dataset.eye_dataset import EyeDataset

seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)
rng = np.random.RandomState(seed)


# Amit Kohli, Julien Martel, and Anastasios Angelopoulos
# August 10, 2020
parser = argparse.ArgumentParser(description='Arguments for using the eye visualizer')
parser.add_argument('--subject',
                    type=int, default=22,  # 22
                    help='which subject to evaluate')
parser.add_argument('--eye',
                    default='left', choices=['left', 'right'],
                    help='Which eye to visualize, left or right')
parser.add_argument('--data_dir',
                    default=os.path.join(os.getcwd(), 'eye_data'),
                    help='absolute path to eye_data/, by default assumes same parent dir as this script')
parser.add_argument('--buffer',
                    type=int, default=2000,
                    help='How many events to store before displaying.')
opt = parser.parse_args()

'Types of data'
Event = namedtuple('Event', 'polarity row col timestamp label')
Frame = namedtuple('Frame', 'row col img timestamp')

'Color scheme for event polarity'
color = ['r', 'g']
color_plot = ['deeppink', 'seagreen']

'Event  346 x 260'
image_x_size = 346
image_y_size = 260
image_size = image_x_size * image_y_size
image_x_center = image_x_size / 2
image_y_center = image_y_size / 2

e_buf_dtype = np.dtype({'names': ('col', 'row', 'pol'), 'formats': (np.uint16, np.uint16, (np.str_, 1))})
e_buf_cal_dtype = np.dtype({'names': ('col', 'row', 'pol'), 'formats': (np.int32, np.int32, (np.str_, 1))})
COL, ROW, POL = int(0), int(1), int(2)
abort_fit_cnt_thsld = 10


class MsgC:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_color(pol_buf):
    return np.take(color, pol_buf)


def get_ellipse(center=[1, 1], width=1, height=.6, phi=3.14 / 5):
    t = np.linspace(0, 2 * pi, 100)
    ellipse_no_rot = np.array([width * np.cos(t), height * np.sin(t)])  # u,v removed to keep the same center location
    rotation_matrix = np.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]])  # 2-D rotation matrix

    ellipse = np.zeros((2, ellipse_no_rot.shape[1]))
    for i in range(ellipse_no_rot.shape[1]):
        ellipse[:, i] = np.dot(rotation_matrix, ellipse_no_rot[:, i])
    ellipse[0, :] = center[0] + ellipse[0, :]
    ellipse[1, :] = center[1] + ellipse[1, :]
    return ellipse


def get_rad_from_ell_c(center, width, height, angle, coord_buf):
    # The ellipse
    xy_c = coord_buf - center

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    xct = xy_c[:, COL] * cos_angle + xy_c[:, ROW] * sin_angle
    yct = xy_c[:, COL] * sin_angle - xy_c[:, ROW] * cos_angle

    rad_cc = (np.square(xct) / width ** 2) + (np.square(yct) / height ** 2)
    return rad_cc


def get_ell_roi_points(center, width, height, angle, e_buf, inner=True):
    rad_cc = get_rad_from_ell_c(center, width, height, angle, e_buf[:, 0:2])
    roi_e_buf_mask = rad_cc <= 1 if inner else rad_cc > 1
    roi_e_buf = np.compress(roi_e_buf_mask, e_buf, axis=0)
    return roi_e_buf


def get_ell_fit_score(center, width, height, angle, coord_buf):
    rad_cc = get_rad_from_ell_c(center, width, height, angle, coord_buf)
    return np.mean(np.abs(rad_cc-1))


def diff(current, previous):
    if current == previous:
        return 0
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return float('inf')


def diff_rad(current, previous):
    if current == previous:
        return 0
    try:
        return (abs(current - previous) / (2 * pi)) * 100.0
    except ZeroDivisionError:
        return float('inf')


def get_eye_offset(e_buf, pf=False):
    print_figure = pf
    disk_2 = disk(2)
    disk_5 = disk(5)

    # coordinate to image
    image = np.zeros((image_x_size, image_y_size), dtype=np.uint8)
    image[(e_buf[:, COL], e_buf[:, ROW])] = 1

    # morphing
    start_time = timeit.default_timer()
    im_out = cv.medianBlur(image, 3)
    im_out = cv.morphologyEx(im_out, cv.MORPH_CLOSE, disk_5)
    im_out = cv.medianBlur(im_out, 3)
    im_out = cv.medianBlur(im_out, 3)
    im_out = cv.dilate(im_out, disk_5)
    im_out = cv.erode(im_out, disk_2)
    im_out = cv.morphologyEx(im_out, cv.MORPH_CLOSE, disk_2)
    im_out = cv.dilate(im_out, disk_5)
    run_time = timeit.default_timer() - start_time
    print(f'*****  CPU time: {run_time}')

    # plot morphing result
    if print_figure:
        plt.subplot(2, 1, 1)
        plt.imshow(image.T, cmap='gray', origin='upper')
        plt.title("user_%s original image" % opt.subject)
        plt.subplot(2, 1, 2)
        plt.imshow(im_out.T, cmap='gray', origin='upper')
        plt.title("user_%s morphology image" % opt.subject)
        plt.tight_layout()
        plt.show()
        plt.pause(0.0001)

    # get bounding box and offset
    hist_bin_size = 5
    search_start, search_end = True, False
    cluster_size, cur_start, cur_end, row_min, row_max = 0, 0, 0, 0, 0
    indices = np.where(im_out != [0])
    col, row = indices[0], indices[1]
    hist_row, bin_edge_row = np.histogram(row, bins=np.arange(0, image_y_size, hist_bin_size))
    hist_row = np.append(hist_row, 0)

    for i in range(len(hist_row)):
        if search_start and (hist_row[i] > 0):
            cur_start = i
            search_start = False
            search_end = True

        elif search_end and (hist_row[i] == 0):
            cur_end = i - 1
            search_start = True
            search_end = False

            cur_cluster_size = cur_end - cur_start + 1
            if cur_cluster_size >= cluster_size:
                row_min, row_max, cluster_size = cur_start, cur_end, cur_cluster_size

    row_min, row_max = row_min * hist_bin_size, row_max * hist_bin_size
    filtered_col_buf = np.where(np.logical_and(row >= row_min, row <= row_max), col, 0)
    col_min = np.min(filtered_col_buf[np.nonzero(filtered_col_buf)])
    col_max = np.max(filtered_col_buf[np.nonzero(filtered_col_buf)])
    col_center, row_center = int((col_min + col_max) / 2), int((row_min + row_max) / 2)
    im_col_center, im_row_center = image_x_size / 2, image_y_size / 2
    col_offset, row_offset = int(col_center - im_col_center), int(row_center - im_row_center)
    eye_width, eye_height = int(col_max - col_min), int(row_max - row_min)

    # plot bounding box
    if print_figure:
        offset_e_buf = deepcopy(e_buf)
        offset_e_buf[:, :-1] -= np.array([col_offset, row_offset])
        offset_e_buf_col_mask = np.logical_and(offset_e_buf[:, COL] >= 0, offset_e_buf[:, COL] < image_x_size)
        offset_e_buf_row_mask = np.logical_and(offset_e_buf[:, ROW] >= 0, offset_e_buf[:, ROW] < image_y_size)
        offset_e_buf_mask = np.logical_and(offset_e_buf_col_mask, offset_e_buf_row_mask)
        offset_e_buf = np.compress(offset_e_buf_mask, offset_e_buf, axis=0)

        col_max, col_min = (col_max - col_offset), (col_min - col_offset)
        row_max, row_min = (row_max - row_offset), (row_min - row_offset)
        figure(figsize=(12, 9), dpi=240)
        plt.scatter(offset_e_buf[:, COL], offset_e_buf[:, ROW], color=get_color(offset_e_buf[:, POL]), alpha=0.3, s=1)
        plt.vlines(x=col_max, ymax=row_max, ymin=row_min, colors='mediumpurple', linestyles='-')
        plt.vlines(x=col_min, ymax=row_max, ymin=row_min, colors='mediumpurple', linestyles='-')
        plt.hlines(y=row_max, xmax=col_max, xmin=col_min, colors='mediumpurple', linestyles='-')
        plt.hlines(y=row_min, xmax=col_max, xmin=col_min, colors='mediumpurple', linestyles='-')
        plt.vlines(x=(col_max + col_min) / 2, ymax=row_max, ymin=row_min, colors='mediumpurple', linestyles='--')
        plt.hlines(y=(row_max + row_min) / 2, xmax=col_max, xmin=col_min, colors='mediumpurple', linestyles='--')
        plt.scatter(col_offset, row_offset, marker='x', s=50, zorder=10, color='purple')

        center = [173, 150]
        width, height, phi = 120, 70, 0
        roi_ellipse = get_ellipse(center, width, height, phi)
        plt.plot(roi_ellipse[0, :], roi_ellipse[1, :], 'blue')  # rotated ellipse
        plt.xlim(0, 346)
        plt.ylim(0, 260)
        plt.title("user_%s image OG BB" % opt.subject)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.pause(0.0001)
    else:
        plt.close()

    return col_offset, row_offset, eye_width, eye_height


def handle_sensor_issue(col_buf, row_buf, pol_buf):
    # remove sensor issue: multiple amount of the same point
    event_buf = list(zip(col_buf, row_buf, pol_buf))
    c = Counter(event_buf)
    most_common_item, most_common_item_cnt = c.most_common(1)[0]
    issue_exists = most_common_item_cnt > 15
    if issue_exists:
        event_buf = [event_buf_i for event_buf_i in event_buf if event_buf_i != most_common_item]
        if len(event_buf) == 0:
            col_buf, row_buf, pol_buf = [], [], []
        else:
            col_buf, row_buf, pol_buf = list(map(list, zip(*event_buf)))
    return issue_exists, col_buf, row_buf, pol_buf


def weighted_categorical_crossentropy(weights):

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss_wcc = y_true * K.log(y_pred) * weights
        loss_wcc = -K.sum(loss_wcc, -1)
        return loss_wcc

    return loss


def event_to_frame(event_set, col_offset, row_offset):
    img_x_size = 352
    img_y_size = 256
    image = np.zeros((img_x_size, img_y_size, 3), dtype=np.uint8)

    for i in range(event_set.shape[0]):
        if event_set[i][2] == 0:  # NEG polarity=0
            image[(event_set[i][0]+3+col_offset-1, event_set[i][1]-2+row_offset-2, 0)] += 10  # R
        else:
            image[(event_set[i][0]+3+col_offset-1, event_set[i][1]-2+row_offset-2, 1)] += 10  # G
        image[(event_set[i][0]+3+col_offset-1, event_set[i][1]-2+row_offset-2, 2)] += 10      # B
    image = np.clip(image, 0, 255)
    return image/255


def e_track(eye_dataset):
    col_buf, row_buf, pol_buf, ts_buf = [], [], [], []
    ell_col_buf, ell_row_buf, ell_pol_buf = [], [], []
    first_group, first_unet = True, True
    abort_unet, abort_fit = False, False
    do_unet = False
    print_figure = False
    abort_fit_cnt = 0
    buf_thsld = 4000

    group_cnt, good_ell_cnt = 0, 0
    init_width, init_height, init_phi = 120, 70, 0
    init_center = [173, 150]
    center, width, height, phi = init_center, init_width, init_height, init_phi

    frame_ts, frame_x, frame_y = [], [], []
    col_offset, row_offset = 0, 0
    unet_cnt, unet_ell_cnt = 0, 0
    roi_cnt, roi_ell_cnt = 0, 0
    num_ell = 500

    # Load U-Net
    custom_objects['loss'] = weighted_categorical_crossentropy(np.array([0.1, 0.9]))
    unet_model = tf.keras.models.load_model('trained_model/2023-01-24T00-11_42',
                                            custom_objects=custom_objects)
    unet_model.summary()

    for i, data in enumerate(eye_dataset):
        if type(data) is Frame:
            frame_ts += [data.timestamp]
            if group_cnt < num_ell:
                frame_x += [data.col]
                frame_y += [data.row]
                frame_img = data.img

        else:  # Event
            pre_do_unet = do_unet
            no_abort = not (abort_unet or abort_fit)
            if first_unet or abort_unet or (abort_fit and (pre_do_unet or (abort_fit_cnt >= abort_fit_cnt_thsld))):
                do_unet = True
                buf_thsld = 4000 if first_group else opt.buffer
            elif no_abort or abort_fit:
                do_unet = False
                buf_thsld = 256
            else:
                print(f"{MsgC.FAIL}[ABORT]" + " " + str(abort_unet) + " " + str(abort_fit) + f"{MsgC.ENDC}")

            # offset the eye to the center of image
            offset_col, offset_row = (data.col - col_offset), (data.row - row_offset)

            if not do_unet:
                'Event-Based RoI Mechanism'
                event_vec = np.array([[offset_col, offset_row]])
                event_vec = get_ell_roi_points(center, width + 4, height + 4, phi, event_vec)
                event_vec = get_ell_roi_points(center, width - 4, height - 4, phi, event_vec, inner=False)
                if event_vec.shape[0] == 0:
                    continue

            if (offset_col in range(0, image_x_size)) and (offset_row in range(0, image_y_size)):
                col_buf += [offset_col]
                row_buf += [offset_row]
                pol_buf += [data.polarity]
                ts_buf += [data.timestamp]

            if len(col_buf) == buf_thsld:
                # remove sensor issue: multiple amount of the same point
                issue_exists, col_buf, row_buf, pol_buf = handle_sensor_issue(col_buf, row_buf, pol_buf)
                if issue_exists:
                    continue
                e_buf = np.array([col_buf, row_buf, pol_buf], dtype=np.int32).T
                print('****** group_cnt: ' + str(group_cnt))

                # =====================================================================================================
                #    One-time Eye Region Offset Calculation
                # =====================================================================================================
                if first_group:
                    first_group = False
                    buf_thsld = opt.buffer
                    col_offset, row_offset, _, _ = get_eye_offset(e_buf, False)
                    # clear all buffers
                    for buf in [ell_col_buf, ell_row_buf, ell_pol_buf, col_buf, row_buf, pol_buf, ts_buf]:
                        buf.clear()
                    group_cnt += 1
                    print('****** col_offset: ' + str(col_offset) + ' row_offset:' + str(row_offset))
                    continue

                if data.label == [0, 0]:
                    # clear all buffers
                    for buf in [ell_col_buf, ell_row_buf, ell_pol_buf, col_buf, row_buf, pol_buf, ts_buf]:
                        buf.clear()
                    group_cnt += 1
                    continue

                # =====================================================================================================
                #    ROI Data Filtering
                # =====================================================================================================
                if do_unet:
                    unet_cnt += 1
                    first_unet = False
                    # reset ROI parameter
                    center, width, height, phi = init_center, init_width, init_height, init_phi

                    'Event to Frame Conversion'
                    image = event_to_frame(e_buf, 0, 0)  # col_offset, row_offset)
                    image = np.expand_dims(image, axis=0)

                    'Unet Prediction'
                    prediction = unet_model.predict(image)
                    ell_buf = tf.where(prediction[0, :, :, 1] > 0.9).numpy()
                    ell_buf += [-2, 4]

                    if ell_buf.shape[0] == 0:
                        abort_fit_cnt = 0
                        abort_unet = True
                        first_fit = False
                    else:
                        abort_unet = False
                        first_fit = True

                else:
                    'Event-Based RoI Mechanism'
                    roi_cnt += 1
                    first_fit = False
                    ell_buf = e_buf

                    if ell_buf.shape[0] == 0:
                        abort_fit_cnt += 1
                    print('****** ell_col_buf len: ' + str(ell_buf.shape[0]))

                # abort if number of events in ROI is not enough
                if ell_buf.shape[0] <= 30:  # or (0 not in ell_buf[:, POL]) or (1 not in ell_buf[:, POL]):
                    abort_fit = True
                    abort_fit_cnt += 1
                else:
                    abort_fit = False

                if abort_fit or abort_unet:
                    print(f"{MsgC.WARNING}[Warning] number of events in ROI is not enough !!!!!{MsgC.ENDC}")

                    # clear buffers
                    for buf in [ell_col_buf, ell_row_buf, ell_pol_buf, col_buf, row_buf, pol_buf, ts_buf]:
                        buf.clear()
                    group_cnt += 1
                    continue

                # =====================================================================================================
                #    Ellipse Fitting
                # =====================================================================================================
                # ellipse fitting
                noise = np.random.rand(ell_buf.shape[0], 2)
                measurements = ell_buf[:, 0:2] + (noise / 10)
                reg = LsqEllipse().fit(measurements)

                # save parameters for rolling back due to bad fit
                prvs_center, prvs_width, prvs_height, prvs_phi = center, width, height, phi
                ellipse_center, ellipse_width, ellipse_height, ellipse_phi = reg.as_parameters()
                center, width, height, phi = ellipse_center, ellipse_width, ellipse_height, ellipse_phi

                fit_score = get_ell_fit_score(center, width, height, phi, ell_buf[:, 0:2])
                diff_thsld, diff_width, diff_height = 17, diff(width, prvs_width), diff(height, prvs_height)

                if (fit_score > 0.19) or\
                        (first_fit and (width / height > 2) and (((center[0]-image_x_size/2) > 0) is not (phi > 0))) or\
                        ((not first_fit) and ((diff_width > diff_thsld) or (diff_height > diff_thsld) or (width > 40))):
                    abort_fit = True
                    abort_fit_cnt += 1
                    center, width, height, phi = prvs_center, prvs_width, prvs_height, prvs_phi

                    # clear buffers
                    for buf in [ell_col_buf, ell_row_buf, ell_pol_buf, col_buf, row_buf, pol_buf, ts_buf]:
                        buf.clear()
                    group_cnt += 1
                    continue
                else:
                    abort_fit = False
                    abort_fit_cnt = 0

                if print_figure:
                    figure(figsize=(7, 5))
                    plt.imshow(frame_img, cmap="gray", alpha=0.7)
                    plt.scatter(e_buf[:, COL]+col_offset, e_buf[:, ROW]+row_offset,
                                color=get_color(e_buf[:, POL]), alpha=0.5, s=10)
                    ellipse = get_ellipse(center, width, height, phi)
                    plt.plot(ellipse[0, :]+col_offset, ellipse[1, :]+row_offset, 'white', linewidth=2.5)
                    plt.plot(ellipse[0, :]+col_offset, ellipse[1, :]+row_offset, 'b', linewidth=2)
                    plt.xlim(0, 346)
                    plt.ylim(0, 260)

                    legend_elements = [mpl.lines.Line2D([0], [0], label='Positive Events', color='gainsboro',
                                                        marker='o', markerfacecolor='g'),
                                       mpl.lines.Line2D([0], [0], label='Negative Events', color='gainsboro',
                                                        marker='o', markerfacecolor='r'),  # ] #,
                                       mpl.lines.Line2D([0], [0], label='Fitted Ellipse', color='b')]
                    plt.legend(handles=legend_elements, loc=3)
                    plt.xlabel("x coordinate (pixel)", labelpad=10)
                    plt.ylabel("y coordinate (pixel)", labelpad=10)
                    plt.title(f'label={data.label}, position=({center[0]:.2f}, '
                              f'{center[1]:.2f}), width={width:.2f}, height={height:.2f}')
                    plt.gca().invert_yaxis()
                    plt.gca().set_box_aspect(260/346)
                    plt.tight_layout()
                    plt.pause(0.0001)
                    plt.show()

                group_cnt += 1
                good_ell_cnt += 1

                # clear part of buffers
                for buf in [ell_col_buf, ell_row_buf, ell_pol_buf, col_buf, row_buf, pol_buf, ts_buf]:
                    buf.clear()

                if do_unet:
                    unet_ell_cnt += 1
                else:
                    roi_ell_cnt += 1
    return None


def main():
    # Amit Kohli, Julien Martel, and Anastasios Angelopoulos
    # August 10, 2020
    eye_dataset = EyeDataset(opt.data_dir, opt.subject)
    if opt.eye == 'left':
        print('Showing the left eye of subject ' + str(opt.subject) + '\n')
        print('Loading Data from ' + opt.data_dir + '..... \n')
        eye_dataset.collect_data(0)
    else:
        print('Showing the right eye of subject ' + str(opt.subject) + '\n')
        print('Loading Data from ' + opt.data_dir + '..... \n')
        eye_dataset.collect_data(1)

    print('Total length of data (event + frame): ' + str(eye_dataset.__len__()))
    print('Parsing Data using a ' + str(opt.buffer) + ' event buffer...')

    target_event_sets, target_event_set_labels = e_track(eye_dataset)


if __name__ == '__main__':
    freeze_support()
    main()
