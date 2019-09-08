from cv2 import cv2
import numpy as np 

from state_transition_model.AckermanModel import AckermanModel
from draw_utils.draw_tools import draw_ackerman_model


if __name__ == "__main__":
    
    display = np.ones((720,1280,3), dtype=np.uint8) * 255
    center = np.array([display.shape[1]/2, display.shape[0]/2]).astype(np.int32) # Pixel space world center
    mtp_ratio = 10

    # Time calculations
    delay_milliseconds = 20
    dt = delay_milliseconds/1000

    init_position = np.array([0.0,0.0])
    init_heading = np.array([0.0,1.0])
    wheel_base = 4.76
    tread = 2.3
    max_steer_angle = 33.75
    min_velocity = -10.0 * dt
    max_velocity = 55.55 * dt

    # Creating Ackerman object and drawing
    car = AckermanModel(init_position, init_heading, wheel_base, tread, max_steer_angle, min_velocity, max_velocity)
    
    draw_ackerman_model(display, center, car, mtp_ratio)

    cv2.imshow('image', cv2.flip(display, 0))
    cv2.waitKey(0)
    
    velocity = 1
    step = 1
    steer_angle = 0
    for i in range(1, 1000):
 
        if i % 50 == 0:
            step = -step

        steer_angle += step
        car.update(steer_angle,velocity)

        # Drawing car axel frame
        display = np.ones((720,1280,3), dtype=np.uint8) * 255
        draw_ackerman_model(display, center, car, mtp_ratio)

        cv2.imshow('image', cv2.flip(display, 0))
        key = cv2.waitKey(delay_milliseconds)

    cv2.waitKey(0)