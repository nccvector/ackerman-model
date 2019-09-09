from cv2 import cv2
import numpy as np 
import math

from state_transition_model.AckermanModel import AckermanModel
from draw_utils.draw_tools import *


refPt = []
 
def select_points(event, x, y, flags, param):
    # grab references to the global variables
    global refPt

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)
        refPt.append((x,y))

        if len(refPt) > 1:
            for i in range(1, len(refPt)):
                cv2.line(display, refPt[i-1], refPt[i], (255,0,0), 1)

        for point in refPt:
            cv2.circle(display, point, 3, (0,255,0), -1)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None, None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


if __name__ == "__main__":
    
    display = np.ones((720,1280,3), dtype=np.uint8) * 255

    # Attaching mouse callback to display
    window_name = 'Display'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_points)

    center = np.array([display.shape[1]/2, display.shape[0]/2]).astype(np.int32) # Pixel space world center
    mtp_ratio = 10.0 # Meter to pixel scale factor

    # Selecting trajectory points
    print('Please click to select trajectory points to follow')
    print('Press Q to start the simulation')
    
    validated = False
    while not validated:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(5)
        if len(refPt) > 1 and key == ord('q'):
            validated = True

    # Resetting display
    display = np.ones((720,1280,3), dtype=np.uint8) * 255

    # Transforming points to image center and meter
    points_list = []
    for p in refPt:
        point = np.array([(p[0]-center[0])/mtp_ratio, (center[1]-p[1])/mtp_ratio], dtype=np.float)
        points_list.append(point)

    # Re drawing points for confirmation
    for i in range(1, len(points_list)):
        cv2.line(display, tuple((points_list[i-1] * mtp_ratio + center).astype(np.int32)), 
            tuple((points_list[i] * mtp_ratio + center).astype(np.int32)), (255,0,0), 1)

    for point in points_list:
        cv2.circle(display, tuple((point * mtp_ratio + center).astype(np.int32)), 3, (0,255,0), -1)

    # Time calculations
    delay_milliseconds = 20
    dt = delay_milliseconds/1000

    init_position = points_list[0].copy()
    sec_point = points_list[1].copy()
    diff = sec_point - init_position
    init_heading = diff/np.linalg.norm(diff)
    wheel_base = 4.76
    tread = 2.3
    max_steer_angle = 33.75
    min_velocity = -10.0 * dt
    max_velocity = 55.55 * dt

    # Creating Ackerman object and drawing
    car = AckermanModel(init_position, init_heading, wheel_base, tread, max_steer_angle, min_velocity, max_velocity)

    # PID parameters
    kp = 10
    ki = 0.01
    kd = 5

    total_error = 0
    prev_error = 0

    pursuit_point_distance = 5 # Meters
    pursuit_point = car.front_position + car.heading * pursuit_point_distance 
    
    new_display = display.copy()
    draw_ackerman_model(new_display, center, car, mtp_ratio, pursuit_point=pursuit_point)

    cv2.imshow(window_name, cv2.flip(new_display, 0))
    key = cv2.waitKey(0)
    
    # Initializers
    velocity = 25 * dt
    steer_angle = 0

    reached_goal = False
    while not reached_goal:
        
        min_dist_1 = math.inf # math inf
        
        # Finding two closest points
        for i in range(0, len(points_list)):
            distance = np.linalg.norm(points_list[i] - pursuit_point)

            if distance < min_dist_1:
                if i != len(points_list)-1:
                    min_dist_1 = distance
                    cp_1 = points_list[i]
                    cp_2 = points_list[i+1]
                else:
                    # There is no i+1 point left, so reached goal
                    reached_goal = True

        ### IMPLEMENT CONTROL
        f_vector = cp_2 - cp_1

        # Avoiding infinite slope
        if f_vector[1] == 0:
            m = 1000000
        else:
            m = f_vector[1]/f_vector[0]

        c = cp_1[1] - m * cp_1[0]

        y_1 = m * -1 + c
        y_2 = m * 1 + c

        # Avoiding infinite slope
        if m == 0:
            m_perp = 1000000
        else:
            m_perp = -1 / m

        c_perp = pursuit_point[1] - m_perp * pursuit_point[0]

        y_perp_1 = m_perp * -1 + c_perp
        y_perp_2 = m_perp * 1 + c_perp

        follow_line = [(-1, y_1), (1, y_2)]
        line_perp = [(-1, y_perp_1), (1, y_perp_2)]

        x_star, y_star = line_intersection(follow_line, line_perp)
        follow_point = np.array([x_star, y_star])

        side = (cp_2[0] - cp_1[0])*(pursuit_point[1] - cp_1[1]) - (cp_2[1] - cp_1[1])*(pursuit_point[0] - cp_1[0])

        if side > 0:
            error = np.linalg.norm(follow_point - pursuit_point)
        elif side < 0:
            error = -np.linalg.norm(follow_point - pursuit_point)
        else:
            error = 0

        # Steering angle is the error
        total_error += error
        steer_angle = kp * error + ki * total_error + kd * (error - prev_error) 

        print(steer_angle)
        car.update(steer_angle, velocity)
        # Update puruit point as well
        pursuit_point = car.front_position + car.heading * pursuit_point_distance 

        # Drawing car axel frame
        new_display = display.copy()
        draw_ackerman_model(new_display, center, car, mtp_ratio, pursuit_point=pursuit_point)

        # Drawing closest two points
        draw_np_point(new_display, center, cp_1, mtp_ratio) ###
        draw_np_point(new_display, center, cp_2, mtp_ratio) ###
        draw_np_point(new_display, center, follow_point, mtp_ratio)
        # draw_np_line(new_display, center, cp_1, cp_1+follow_line_vector, mtp_ratio)
        # draw_np_line(new_display, center, car.rear_position, car.rear_position+pursuit_vector, mtp_ratio)

        cv2.imshow(window_name, cv2.flip(new_display, 0))
        key = cv2.waitKey(delay_milliseconds)

        prev_error = error

    cv2.waitKey(0)