from cv2 import cv2
import numpy as np 
import math

class Ackerman:

    def __init__(self, position, heading, wheel_base, tread, max_steer, min_step, drive="rear"):

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

        # Calculated attributes
        self.front_position = self.rear_position + self.heading * self.wheel_base   # Center of front axel
        self.normal = np.array([self.heading[1], -self.heading[0]])     # Right vector of car
        # Calculated corners of car
        self.front_left_position = self.front_position - self.normal * self.tread/2
        self.front_right_position = self.front_position + self.normal * self.tread/2
        self.rear_left_position = self.rear_position - self.normal * self.tread/2
        self.rear_right_position = self.rear_position + self.normal * self.tread/2

    def _line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

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

            if steering_angle <= 0:
                rear_arc_angle = rear_base_arc_angle + theta_step
                front_arc_angle = front_base_arc_angle + theta_step
            else:
                rear_arc_angle = rear_base_arc_angle - theta_step
                front_arc_angle = front_base_arc_angle - theta_step

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

    def get_corners(self):
        corners = [self.front_left_position, self.front_right_position, self.rear_right_position, self.rear_left_position]
        corners = np.array(corners, np.int32)
        corners = corners.reshape((-1,1,2))
        return [corners]


if __name__ == "__main__":
    
    display = np.ones((720,1280,3), dtype=np.uint8) * 255

    car = Ackerman(np.array([640,360]), np.array([0,1]), 60, 25, 45, 1)
    cv2.polylines(display, car.get_corners(), True, (0,255,0), 1)
    
    for i in range(1, 25):
        car.update(30,10)
        cv2.polylines(display, car.get_corners(), True, (0,255,0), 1)

        cv2.imshow('image', cv2.flip(display, 0))
        cv2.waitKey(30)

    for i in range(0, 25):
        car.update(45,-6)
        cv2.polylines(display, car.get_corners(), True, (0,255,0), 1)

        cv2.imshow('image', cv2.flip(display, 0))
        cv2.waitKey(30)

    cv2.waitKey(0)