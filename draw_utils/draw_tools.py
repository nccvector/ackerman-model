from cv2 import cv2
import numpy as np

def draw_ackerman_model(image, center, object, mtp_ratio):

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