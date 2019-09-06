from cv2 import cv2
import numpy as np 
import math

class Ackerman:

    def __init__(self, position, heading, wheel_base, tread, max_steer, min_step, drive="rear", tyre_radius=10):

        """
        Ackerman model class
        Model position is taken from center of rear axel
        Front axel position lies at rear_position + heading * wheel_base

        In both front and rear axels, tires are placed at tread/2 * -(heading x z_vector) 
        and tread/2 * -(heading x z_vector)
        
        Parameters
        ----------
        position : numpy array ([x, y])
            Position of model in world

        heading : numpy array ([x, y])
            Heading of model (normalized vector)

        wheel_base : float
            Distance between front and back tires (meters)

        tread : float
            Axial distance between tires (meters)

        max_steer : float
            Maximum steering angle allowed (degrees)

        min_step : float
            Minimum step length (meters)

        """

        self.rear_position = position   # Center of rear axel
        self.heading = heading
        self.wheel_base = wheel_base
        self.tread = tread
        self.max_steer = max_steer
        self.min_step = min_step

        self.drive = drive
        self.tyre_radius = tyre_radius

        # Calculated attributes
        self.front_position = self.rear_position + self.heading * self.wheel_base   # Center of front axel
        self.normal = np.array([self.heading[1], -self.heading[0]])     # Right vector of car
        # Calculated corners of car
        self.front_left_position = self.front_position - self.normal * self.tread/2
        self.front_right_position = self.front_position + self.normal * self.tread/2
        self.rear_left_position = self.rear_position - self.normal * self.tread/2
        self.rear_right_position = self.rear_position + self.normal * self.tread/2

        # Calculating left and right tyres
        left_tyre_p1 = self.front_left_position + self.heading * self.tyre_radius
        left_tyre_p2 = self.front_left_position - self.heading * self.tyre_radius

        right_tyre_p1 = self.front_right_position + self.heading * self.tyre_radius
        right_tyre_p2 = self.front_right_position - self.heading * self.tyre_radius

        self.front_left_tyre = [left_tyre_p1, left_tyre_p2]
        self.front_right_tyre = [right_tyre_p1, right_tyre_p2]

        left_tyre_p1 = self.rear_left_position + self.heading * self.tyre_radius
        left_tyre_p2 = self.rear_left_position - self.heading * self.tyre_radius

        right_tyre_p1 = self.rear_right_position + self.heading * self.tyre_radius
        right_tyre_p2 = self.rear_right_position - self.heading * self.tyre_radius

        self.rear_left_tyre = [left_tyre_p1, left_tyre_p2]
        self.rear_right_tyre = [right_tyre_p1, right_tyre_p2]

    def _line_intersection(self, line1, line2):
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

    def update(self, steering_angle, step_size):
        # Limiting steering angle to max_steer
        if steering_angle < -self.max_steer:
            steering_angle = -self.max_steer
        elif steering_angle > self.max_steer:
            steering_angle = self.max_steer

        # if step_size < self.min_step:
        #     step_size = self.min_step

        # Drive types
        if self.drive == "rear":
            # Finding steering vector
            heading_angle = math.degrees(math.atan2(self.heading[1], self.heading[0])) # Absolute heading angle
            if heading_angle < 0:
                heading_angle += 360


            steering_angle_abs = heading_angle - steering_angle # Absolute steering angle
            if steering_angle_abs < 0:
                steering_angle_abs += 360

            heading_abs_vector = self.rear_position + np.array([self.heading[1], -self.heading[0]])
            steering_abs_vector = self.front_position + np.array([math.sin(math.radians(steering_angle_abs)), -math.cos(math.radians(steering_angle_abs))])

            x, y = self._line_intersection((self.rear_position, heading_abs_vector), (self.front_position,steering_abs_vector))

            if not x == None:
                arc_center = np.array([x, y])

                # Finding theta_s wrt to rear axel
                R = np.linalg.norm(arc_center - self.front_position)
                r = np.linalg.norm(arc_center - self.rear_position)
                theta_step = math.degrees(step_size/r)

                # theta_s is the step the front and back are both going to move along arc

                # Calculating arc angles
                rear_base_arc_vector = self.rear_position - arc_center
                rear_base_arc_angle = math.degrees(math.atan2(rear_base_arc_vector[1], rear_base_arc_vector[0]))
                # if rear_base_arc_angle < 0:
                #     rear_base_arc_angle += 360

                front_base_arc_vector = self.front_position - arc_center
                front_base_arc_angle = math.degrees(math.atan2(front_base_arc_vector[1], front_base_arc_vector[0]))
                # if front_base_arc_angle < 0:
                #     front_base_arc_angle += 360

                if steering_angle < 0:
                    rear_arc_angle = rear_base_arc_angle + theta_step
                    front_arc_angle = front_base_arc_angle + theta_step
                elif steering_angle > 0:
                    rear_arc_angle = rear_base_arc_angle - theta_step
                    front_arc_angle = front_base_arc_angle - theta_step
                
                if not steering_angle == None and steering_angle != 0:

                    # Calculating new positions
                    new_rear_x = arc_center[0] + r * math.cos(math.radians(rear_arc_angle))
                    new_rear_y = arc_center[1] + r * math.sin(math.radians(rear_arc_angle))

                    new_front_x = arc_center[0] + R * math.cos(math.radians(front_arc_angle))
                    new_front_y = arc_center[1] + R * math.sin(math.radians(front_arc_angle))

                    # Updating attributes
                    self.rear_position = np.array([new_rear_x, new_rear_y])
                    self.front_position = np.array([new_front_x, new_front_y])
                    diff = self.front_position - self.rear_position
                    self.heading = diff/np.linalg.norm(diff)
                    self.normal = np.array([self.heading[1], -self.heading[0]])

                    self.front_left_position = self.front_position - self.normal * self.tread/2
                    self.front_right_position = self.front_position + self.normal * self.tread/2
                    self.rear_left_position = self.rear_position - self.normal * self.tread/2
                    self.rear_right_position = self.rear_position + self.normal * self.tread/2

                    # Calculating left and right steering angles fro visualization
                    left_turn_vector = self.front_left_position - arc_center
                    left_turn_vector = np.array([left_turn_vector[1], -left_turn_vector[0]])
                    left_turn_vector = left_turn_vector/np.linalg.norm(left_turn_vector)
                    left_tyre_p1 = self.front_left_position + left_turn_vector * self.tyre_radius
                    left_tyre_p2 = self.front_left_position - left_turn_vector * self.tyre_radius

                    right_turn_vector = self.front_right_position - arc_center
                    right_turn_vector = np.array([right_turn_vector[1], -right_turn_vector[0]])
                    right_turn_vector = right_turn_vector/np.linalg.norm(right_turn_vector)
                    right_tyre_p1 = self.front_right_position + right_turn_vector * self.tyre_radius
                    right_tyre_p2 = self.front_right_position - right_turn_vector * self.tyre_radius

            if x == None or steering_angle == 0:
                # Updating attributes
                self.rear_position += self.heading * step_size
                self.front_position += self.heading * step_size
                self.front_left_position += self.heading * step_size
                self.front_right_position += self.heading * step_size
                self.rear_left_position += self.heading * step_size
                self.rear_right_position += self.heading * step_size

                # Calculating left and right steering angles for visualization
                left_tyre_p1 = self.front_left_position + self.heading * self.tyre_radius
                left_tyre_p2 = self.front_left_position - self.heading * self.tyre_radius

                right_tyre_p1 = self.front_right_position + self.heading * self.tyre_radius
                right_tyre_p2 = self.front_right_position - self.heading * self.tyre_radius

            self.front_left_tyre = [left_tyre_p1, left_tyre_p2]
            self.front_right_tyre = [right_tyre_p1, right_tyre_p2]

            left_tyre_p1 = self.rear_left_position + self.heading * self.tyre_radius
            left_tyre_p2 = self.rear_left_position - self.heading * self.tyre_radius

            right_tyre_p1 = self.rear_right_position + self.heading * self.tyre_radius
            right_tyre_p2 = self.rear_right_position - self.heading * self.tyre_radius

            self.rear_left_tyre = [left_tyre_p1, left_tyre_p2]
            self.rear_right_tyre = [right_tyre_p1, right_tyre_p2]


    def get_axel_corners(self):
        corners = [self.front_left_position, self.front_right_position, self.rear_right_position, self.rear_left_position]
        corners = np.array(corners, np.int32)
        corners = corners.reshape((-1,1,2))
        return [corners]


if __name__ == "__main__":
    
    display = np.ones((720,1280,3), dtype=np.uint8) * 255
    display2 = np.ones((720,1280,3), dtype=np.uint8) * 255

    # Drawing car axel frames
    car = Ackerman(np.array([640,300]), np.array([0,1]), 60, 25, 33.75, 1)
    cv2.polylines(display2, car.get_axel_corners(), True, (255,50,255), 1)
    # Drawing tyres
    cv2.line(display, tuple(car.front_left_tyre[0].astype(np.int32)), tuple(car.front_left_tyre[1].astype(np.int32)), (0,0,255), 1, cv2.LINE_AA)
    cv2.line(display, tuple(car.front_right_tyre[0].astype(np.int32)), tuple(car.front_right_tyre[1].astype(np.int32)), (0,0,255), 1, cv2.LINE_AA)
    cv2.line(display, tuple(car.rear_left_tyre[0].astype(np.int32)), tuple(car.rear_left_tyre[1].astype(np.int32)), (255,0,0), 1, cv2.LINE_AA)
    cv2.line(display, tuple(car.rear_right_tyre[0].astype(np.int32)), tuple(car.rear_right_tyre[1].astype(np.int32)), (255,0,0), 1, cv2.LINE_AA)

    display2 = cv2.addWeighted(display, 0.2, display2, 0.8, 0)
    cv2.imshow('image', cv2.flip(display2, 0))
    cv2.waitKey(0)
    
    step = 1
    steer_angle = -35
    for i in range(1, 1000):
 
        if i % 50 == 0:
            step = -step

        steer_angle += step
        car.update(steer_angle,10)

        # Drawing car axel frame
        display2 = np.ones((720,1280,3), dtype=np.uint8) * 255
        cv2.polylines(display2, car.get_axel_corners(), True, (255,50,255), 1)
        # Drawing tyres
        cv2.line(display, tuple(car.front_left_tyre[0].astype(np.int32)), tuple(car.front_left_tyre[1].astype(np.int32)), (0,0,255), 1, cv2.LINE_AA)
        cv2.line(display, tuple(car.front_right_tyre[0].astype(np.int32)), tuple(car.front_right_tyre[1].astype(np.int32)), (0,0,255), 1, cv2.LINE_AA)
        cv2.line(display, tuple(car.rear_left_tyre[0].astype(np.int32)), tuple(car.rear_left_tyre[1].astype(np.int32)), (255,0,0), 1, cv2.LINE_AA)
        cv2.line(display, tuple(car.rear_right_tyre[0].astype(np.int32)), tuple(car.rear_right_tyre[1].astype(np.int32)), (255,0,0), 1, cv2.LINE_AA)

        display2 = cv2.addWeighted(display, 0.2, display2, 0.8, 0)
        cv2.imshow('image', cv2.flip(display2, 0))
        cv2.waitKey(50)

    cv2.waitKey(0)