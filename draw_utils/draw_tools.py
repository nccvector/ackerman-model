from cv2 import cv2
import numpy as np

def draw_ackerman_model(image, center, object, mtp_ratio, pursuit_point=None):

    corners = object.get_axel_corners()
    corners = center + np.array(corners) * mtp_ratio
    corners = corners.reshape((-1,1,2))

    # Drawing axel frame
    cv2.polylines(image, [corners.astype(np.int32)], True, (255,50,255), 1)

    front_left_tyre = center + np.array(object.front_left_tyre) * mtp_ratio
    front_right_tyre = center + np.array(object.front_right_tyre) * mtp_ratio
    rear_left_tyre = center + np.array(object.rear_left_tyre) * mtp_ratio
    rear_right_tyre = center + np.array(object.rear_right_tyre) * mtp_ratio

    cv2.line(image, tuple(front_left_tyre[0].astype(np.int32)), 
        tuple(front_left_tyre[1].astype(np.int32)), (0,0,255), 3, cv2.LINE_AA)
    cv2.line(image, tuple(front_right_tyre[0].astype(np.int32)), 
        tuple(front_right_tyre[1].astype(np.int32)), (0,0,255), 3, cv2.LINE_AA)
    cv2.line(image, tuple(rear_left_tyre[0].astype(np.int32)), 
        tuple(rear_left_tyre[1].astype(np.int32)), (255,0,0), 3, cv2.LINE_AA)
    cv2.line(image, tuple(rear_right_tyre[0].astype(np.int32)), 
        tuple(rear_right_tyre[1].astype(np.int32)), (255,0,0), 3, cv2.LINE_AA)

    if not pursuit_point is None:
        cv2.circle(image, tuple(center + (pursuit_point * mtp_ratio).astype(np.int32)), 3, (0,100,255), -1)


def draw_np_point(image, center, array, mtp_ratio):
    array = center + array * mtp_ratio

    cv2.circle(image, tuple(array.astype(np.int32)), 3, (0,0,0), -1)


def draw_np_line(image, center, array1, array2, mtp_ratio):
    array1 = center + array1 * mtp_ratio
    array2 = center + array2 * mtp_ratio

    cv2.line(image, tuple(array1.astype(np.int32)), 
        tuple(array2.astype(np.int32)), (0,0,0), 2)