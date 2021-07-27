"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 5 - AR Markers
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
import enum

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

class Mode(enum.IntEnum):
    align = 0
    right_panic = 1
    left_panic = 2
    super_right_panic = 3
    super_left_panic = 4
    front_panic = 5
    wide = 6

class Mode_slalom(enum.IntEnum):
    red_align = 0  # Approaching a red cone to pass
    blue_align = 1  # Approaching a blue cone to pass
    red_pass = 2  # Passing a red cone (currently out of sight to our left)
    blue_pass = 3  # Passing a blue cone (currently out of sight to our right)
    red_find = 4  # Finding a red cone with which to align
    blue_find = 5  # Finding a blue cone with which to align
    red_reverse = 6  # Aligning with a red cone, but we are too close so must back up
    blue_reverse = 7  # Aligning with a blue cone, but we are too close so must back up
    no_cones = 8  # No cones in sight, inch forward until we find one


# >> Constants
# The maximum speed the car will travel
MAX_SPEED = 1.0

# When an object in front of the car is closer than this (in cm), start braking
BRAKE_DISTANCE = 150

# When a wall is within this distance (in cm), focus solely on not hitting that wall
SUPER_PANIC_DISTANCE = 21
PANIC_DISTANCE = 20

# When a wall is greater than this distance (in cm) away, exit panic mode
END_PANIC_DISTANCE = 25

WIDE_DISTANCE = 130

# Speed to travel in panic mode
PANIC_SPEED = 0.8
SUPER_PANIC_SPEED = 1.0

# The minimum and maximum angles to consider when measuring closest side distance
MIN_SIDE_ANGLE = 10
MAX_SIDE_ANGLE = 75

# The angles of the two distance measurements used to estimate the angle of the left
# and right walls
SIDE_FRONT_ANGLE = 70
SIDE_BACK_ANGLE = 110

# When the front and back distance measurements of a wall differ by more than this
# amount (in cm), assume that the hallway turns and begin turning
TURN_THRESHOLD = 30

# The angle of measurements to average when taking an average distance measurement
# WINDOW_ANGLE = 60

# >> Variables
cur_mode = Mode.align
pre_value = 0
panic_count = 0

#slalom
# Speeds
MAX_ALIGN_SPEED = 1
MIN_ALIGN_SPEED = 0.4
PASS_SPEED = 0.5
FIND_SPEED = 0.2
REVERSE_SPEED = -0.2
NO_CONES_SPEED = 0.4

# Times
REVERSE_BRAKE_TIME = 0.25
SHORT_PASS_TIME = 1.0
LONG_PASS_TIME = 1.2

# Cone finding parameters
MIN_CONTOUR_AREA = 100
MAX_DISTANCE = 250
REVERSE_DISTANCE = 50
STOP_REVERSE_DISTANCE = 60

CLOSE_DISTANCE = 30
FAR_DISTANCE = 120

cur_mode_slalom = Mode_slalom.no_cones
counter = 0
red_center = None
red_distance = 0
prev_red_distance = 0
blue_center = None
blue_distance = 0
prev_blue_distance = 0

class Stage(enum.IntEnum):
    none = 0
    marker_id = 1
    orientation = 2
    color = 3
    cone = 4
    lane = 5
    wall = 6


# >> Constants
RED = ((170, 50, 50), (10, 255, 255), "red")  # The HSV range for the color blue
BLUE = ((100, 150, 150), (120, 255, 255), "blue")  # The HSV range for the color blue
GREEN = ((40, 50, 50), (80, 255, 255), "green")  # The HSV range for the color green
ORANGE = ((10, 100, 100), (25, 255, 255), "orange")
PURPLE = ((125, 100, 100), (150, 255, 255), "purple")

colors = [RED, BLUE, GREEN, ORANGE, PURPLE]
colors_lane = [ORANGE, PURPLE]

# Wall follow constants
# Angle to the right side of the car
SIDE_ANGLE = 90

# Spread to look in each direction to the side of the car to estimate wall angle
SPREAD_ANGLE = 30

# The angle of measurements to average over for each distance measurement
WINDOW_ANGLE = 60

# Distance in (in cm) to try stay from the wall
WALL_DISTANCE = 36
IS_WALL_DISTANCE = 70
S_DISTANCE = 50
TURN_DISTANCE = 70
COLLISION_DISTANCE = 30

# The maximum difference between the desired value and the current value, which is
# scaled to a full left (-1) or a full right (1) turn
MAX_DIF = 10

# The amount we consider the distance from the wall compared to the angle of the wall
DISTANCE_COEFFICIENT = 0.5

# Line follow constants
# Minimum number of pixels to consider a valid contour
MIN_CONTOUR_AREA = 30
MIN_CONTOUR_AREA_LANE = 500

# Region to which to crop the image when looking for the colored line
CROP_WINDOW = ((300, 0), (rc.camera.get_height(), rc.camera.get_width()))
CROP_WINDOW_lane = ((rc.camera.get_height() * 2 // 3, 0),(rc.camera.get_height(), rc.camera.get_width()))
CROP_WINDOW_lane_right = ((280, int(rc.camera.get_width()/2)), (rc.camera.get_height() - 100, rc.camera.get_width()))
CROP_WINDOW_lane_left = ((280, 0), (rc.camera.get_height() - 100, int(rc.camera.get_width()/2)))
CROP_FLOOR = (
    (rc.camera.get_height() * 2 // 3, 0),
    (rc.camera.get_height(), rc.camera.get_width()),
)

# Speed constants
EXPLORE_SPEED = 0.4
WALL_FOLLOW_SPEED = 0.5
LINE_FOLLOW_SPEED = 0.7
LANE_FOLLOW_SPEED = 0.4
MAX_SPEED = 0.35
# Speed to use when the lane is the primary color
FAST_SPEED = 1.0
# Speed to use when the lane is the secondary color (indicating a sharp turn)
SLOW_SPEED = 0.5

# Amount to turn if we only see one lane
ONE_LANE_TURN_ANGLE = 0.6

# >> Variables
cur_stage = Stage.none

# The most recent marker detected (initialized to a non-existant marker)
cur_marker = rc_utils.ARMarker(-1, np.zeros((4, 2), dtype=np.int32))

# -1 if we are following the left wall, 1 if we are following the right wall
cur_direction = 0

# The color of the line we are currently following
cur_color = None

pre_center = None

# primary lane color (found on AR tag)
primary_color = ORANGE
# secondary lane color, indicating a sharp turns
secondary_color = PURPLE
########################################################################################
# Functions
########################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    global cur_stage
    global cur_marker
    global cur_direction
    global cur_color
    global cur_mode_slalom
    global counter

    # Have the car begin at a stop
    rc.drive.stop()

    # Print start message
    print(">> Lab 5 - AR Markers")

    # Initialize variables
    cur_stage = Stage.none
    cur_marker = rc_utils.ARMarker(-1, np.zeros((4, 2), dtype=np.int32))
    cur_direction = 1
    cur_color = None
    cur_mode_slalom = Mode_slalom.no_cones
    counter = 0


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global cur_stage
    global cur_marker
    global cur_direction
    global cur_color

    color_image = rc.camera.get_color_image_no_copy()
    markers = rc_utils.get_ar_markers(color_image)

    # If we see a new marker, change the stage
    if len(markers) > 0:
        length = markers[0].get_corners()[1][1] - markers[0].get_corners()[0][1]
        high = markers[0].get_corners()[3][0] - markers[0].get_corners()[0][0]
        if length < 10:
            if markers[0].get_corners()[1][0] - markers[0].get_corners()[0][0] > 0:
                length = - markers[0].get_corners()[3][1] + markers[0].get_corners()[0][1]
                high = markers[0].get_corners()[1][0] - markers[0].get_corners()[0][0]
            else:
                length =  markers[0].get_corners()[3][1] - markers[0].get_corners()[0][1]
                high = - markers[0].get_corners()[1][0] + markers[0].get_corners()[0][0]

        area = length * high
        # print("len = ", length)
        # print("high = ", high)
        print("ARmarker = ", area)
        
        if area > 1500 and markers[0].get_id() != cur_marker.get_id():
            change_stage(markers[0], color_image)
            print(markers[0])
        
    speed = 0
    angle = 0
    if cur_stage == Stage.none:
        # Until we see the first marker, gradually move forward
        angle, speed = wall_follow_lab4b()
    elif cur_stage == Stage.marker_id or cur_stage == Stage.orientation or cur_stage == Stage.wall:
        # After the first two markers, follow the wall indicated by the marker
        speed, angle = wall_follow2(cur_direction)
    elif cur_stage == Stage.color:
        # After the third marker, follow the color line indicated by the marker
        speed = LINE_FOLLOW_SPEED
        angle = line_follow(color_image)

        # If we cannot see the colored line yet, continue wall following
        if angle is None:
            _, angle = wall_follow2(cur_direction)
    elif cur_stage == Stage.cone:
        # Until we see the first marker, gradually move forward
        angle, speed = cone_slalom()

        if angle is None:
            _, angle = wall_follow2(cur_direction)
    elif cur_stage == Stage.lane:
        # Until we see the first marker, gradually move forward
        speed, angle = lane_follow_matthew()

        if angle is None:
            _, angle = wall_follow2(cur_direction)        

    rc.drive.set_speed_angle(speed, angle)
    print(cur_stage)

    # Print global variables when the X button is held down
    if rc.controller.is_down(rc.controller.Button.X):
        print(
            f"cur_stage: {cur_stage}, cur_direction: {cur_direction}, cur_color: {cur_color}"
        )


def change_stage(new_marker, color_image):
    """
    Moves to the next stage when a new marker is detected.
    """
    global cur_stage
    global cur_marker
    global cur_direction
    global cur_color
    global colors
    global primary_color
    global secondary_color

    #cur_stage += 1
    cur_marker = new_marker

    if cur_marker.get_id() == 1:
        cur_stage = Stage.lane
        cur_direction = 1
        if cur_color != primary_color[2]:
                temp = primary_color
                primary_color = secondary_color
                secondary_color = temp
                print(f"Primary color set to {primary_color[2]}")
    elif cur_marker.get_id() == 199:
        cur_stage = Stage.orientation
        cur_direction = (
            1 if cur_marker.get_orientation() == rc_utils.Orientation.RIGHT else -1
        )
    elif cur_marker.get_id() == 0:
        cur_stage = Stage.color
    elif cur_marker.get_id() == 2:
        cur_stage = Stage.cone
        cur_direction = 1
    elif cur_marker.get_id() == 3:
        cur_stage = Stage.wall
    
    cur_marker.detect_colors(color_image, colors)
    if new_marker.get_color() == "red":
        cur_color = RED
    elif new_marker.get_color() == "blue":
        cur_color = BLUE
    elif new_marker.get_color() == "green":
        cur_color = GREEN
    elif new_marker.get_color() == "orange":
        cur_color = ORANGE
    elif new_marker.get_color() == "purple":
        cur_color = PURPLE
    else:
        cur_color = PURPLE


def largest_contour(contours, MIN_CONTOUR_AREA):

    max_cont = None
    second_cont = None
    if len(contours) >= 1:
        max_cont = contours[0]
        if len(contours) >= 2:
            second_cont = contours[0]
        for contour in contours:
            if rc_utils.get_contour_area(max_cont) < rc_utils.get_contour_area(contour):
                max_cont = contour
            elif len(contours) >= 2 and rc_utils.get_contour_area(second_cont) < rc_utils.get_contour_area(contour):
                second_cont = contour
        if rc_utils.get_contour_area(max_cont) < MIN_CONTOUR_AREA:
            max_cont = None
        if rc_utils.get_contour_area(second_cont) < MIN_CONTOUR_AREA:
            second_cont = None
        
    return max_cont, second_cont

def lane_follow_matthew():
    speed = SLOW_SPEED
    angle = 0

    #check_ar()

    #if not driving:
        # No AR marker seen yet
    #    rc.drive.stop()
    #    return

    image = rc.camera.get_color_image()
    if image is None:
        print("No image")
        rc.drive.stop()
        return

    # Crop the image to the floor directly in front of the car
    image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])

    # Search for secondary (slow) color first
    contours = [
        contour
        for contour in rc_utils.find_contours(
            image, secondary_color[0], secondary_color[1]
        )
        if cv.contourArea(contour) > MIN_CONTOUR_AREA
    ]

    if len(contours) == 0:
        # Secondary color not found, search for primary (fast) color
        contours = [
            contour
            for contour in rc_utils.find_contours(
                image, primary_color[0], primary_color[1]
            )
            if cv.contourArea(contour) > MIN_CONTOUR_AREA
        ]
        if len(contours) == 0:
            # No contours of either color found, so proceed forward slowly
            print("No lanes found")
            #rc.drive.set_speed_angle(SLOW_SPEED, 0)
            return SLOW_SPEED, 0
        else:
            # We only see the primary color, so it is safe to go fast
            speed = FAST_SPEED

    # If we see at least two contours, aim for the midpoint of the centers of the two
    # largest contours (assumed to be the left and right lanes)
    if len(contours) >= 2:
        # Sort contours from largest to smallest
        contours.sort(key=cv.contourArea, reverse=True)

        # Calculate the midpoint of the two largest contours
        first_center = rc_utils.get_contour_center(contours[0])
        second_center = rc_utils.get_contour_center(contours[1])
        midpoint = (first_center[1] + second_center[1]) / 2

        # Use P-control to aim for the midpoint
        angle = rc_utils.remap_range(midpoint, 0, rc.camera.get_width(), -1, 1)

        # Draw the contours and centers onto the image (red one is larger)
        rc_utils.draw_contour(image, contours[0], rc_utils.ColorBGR.red.value)
        rc_utils.draw_circle(image, first_center, rc_utils.ColorBGR.red.value)
        rc_utils.draw_contour(image, contours[1], rc_utils.ColorBGR.blue.value)
        rc_utils.draw_circle(image, second_center, rc_utils.ColorBGR.blue.value)

    # If we see only one contour, turn back towards the "missing" line
    else:
        contour = contours[0]
        center = rc_utils.get_contour_center(contour)

        if center[1] > rc.camera.get_width() / 2:
            # We can only see the RIGHT lane, so turn LEFT
            angle = -ONE_LANE_TURN_ANGLE
        else:
            # We can only see the LEFT lane, so turn RIGHT
            angle = ONE_LANE_TURN_ANGLE

        # Draw the single contour and center onto the image
        rc_utils.draw_contour(image, contour)
        rc_utils.draw_circle(image, center)

    # Display the image to the screen
    rc.display.show_color_image(image)

    return speed, angle

def lane_follow(color_image):
    """
    Determines the angle of the wheels necessary to follow the colored line
    on the floor, with color priority specified by the colors parameter.

    Uses a similar strategy to Lab 2A.
    """
    global colors_lane
    global pre_center

    # Crop to the floor directly in front of the car
    cropped_image = rc_utils.crop(color_image, CROP_WINDOW_lane[0], CROP_WINDOW_lane[1])

    # Search for the colors in priority order
    for color in colors_lane:
        # Find the largest contour of the specified color
        contours = rc_utils.find_contours(cropped_image, color[0], color[1])
        contour, second_contour = largest_contour(contours, MIN_CONTOUR_AREA)

        # If the contour exists, steer toward the center of the contour
        if contour is not None and second_contour is not None:
            center = rc_utils.get_contour_center(contour)
            print(center)
            second_center = rc_utils.get_contour_center(second_contour)
            print(second_center)
            center = (int((center[0] + second_center[0]) / 2), int((center[1] + second_center[1]) / 2))
            #if pre_center == None:
            #    pre_center = center
            #if abs(center[0] - pre_center[0]) > 100 or abs(center[1] - pre_center[1]) > 200:
            #   center = pre_center
            rc_utils.draw_contour(cropped_image, contour)
            rc_utils.draw_contour(cropped_image, second_contour, (255, 0, 0))
            rc_utils.draw_circle(cropped_image, center)
            rc.display.show_color_image(cropped_image)
            return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)

    # If no color lines are found, return None so that we wall follow instead
    return None

def lane_follow_crop(color_image):
    """
    Determines the angle of the wheels necessary to follow the colored line
    on the floor, with color priority specified by the colors parameter.

    Uses a similar strategy to Lab 2A.
    """
    global colors_lane
    global pre_center

    # Crop to the floor directly in front of the car
    cropped_image_left = rc_utils.crop(color_image, CROP_WINDOW_lane_left[0], CROP_WINDOW_lane_left[1])
    cropped_image_right = rc_utils.crop(color_image, CROP_WINDOW_lane_right[0], CROP_WINDOW_lane_right[1])

    # Search for the colors in priority order
    for color in colors_lane:
        # Find the largest contour of the specified color
        contours = rc_utils.find_contours(cropped_image_right, color[0], color[1])
        contour_right = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        # If the contour exists, steer toward the center of the contour
        if contour_right is not None:
            center_right = rc_utils.get_contour_center(contour_right)
            rc_utils.draw_contour(cropped_image_right, contour_right)
            rc_utils.draw_circle(cropped_image_right, center_right)
            rc.display.show_color_image(cropped_image_right)
    
    for color in colors_lane:
        # Find the largest contour of the specified color
        contours = rc_utils.find_contours(cropped_image_left, color[0], color[1])
        contour_left = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        # If the contour exists, steer toward the center of the contour
        if contour_left is not None:
            center_left = rc_utils.get_contour_center(contour_left)
            rc_utils.draw_contour(cropped_image_left, contour_left)
            rc_utils.draw_circle(cropped_image_left, center_left)
            rc.display.show_color_image(cropped_image_left)

    if contour_right is not None and contour_left is not None:
        center = ((center_left[0] + rc.camera.get_width()/2 + center_right[0]) / 2, (center_left[1] + center_right[1]) / 2)
        print("both")
        return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)
    elif contour_left is not None:
        center = (center_left[0] + 200, center_left[1])
        print("left")
        return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)
    elif contour_right is not None:
        center = (rc.camera.get_width()/2 + center_right[0] - 200, center_right[1])
        print("right")
        return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)
    else:

        # If no color lines are found, return None so that we wall follow instead
        return None

def sab():
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
    second_contour = None
    if len(contours) >= 2:
        i = 0
        for cont in contours:
            if cont is not None or cont.all() == contour.all():
                largest_contour = i
            i += 1
        contours.pop(largest_contour)
        second_contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA_MIN)

def lane_follow_Ave(color_image):
    """
    Lane follow using all contours to get center value (Tottie)
    """
    global colors_lane
    global pre_center

    # Crop to the floor directly in front of the car
    cropped_image = rc_utils.crop(color_image, CROP_WINDOW_lane[0], CROP_WINDOW_lane[1])

    # Search for the colors in priority order
    for color in colors_lane:
        # Find the largest contour of the specified color
        contours = rc_utils.find_contours(cropped_image, color[0], color[1])
        center_sum = (0, 0)

        # If the contour exists, steer toward the center of the contour
        if len(contours) > 0:
            for cont in contours:
                center = rc_utils.get_contour_center(cont)
                if center is not None:
                    center_sum = (center[0] + center_sum[0], center[1] + center_sum[1])
                rc_utils.draw_contour(cropped_image, cont)

            center = (int(center_sum[0] / len(contours)), int(center_sum[1] / len(contours)))           
            rc_utils.draw_circle(cropped_image, center)
            rc.display.show_color_image(cropped_image)
            return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)

    # If no color lines are found, return None so that we wall follow instead
    return None

def find_cones():
    """
    Find the closest red and blue cones and update corresponding global variables.
    """
    global red_center
    global red_distance
    global prev_red_distance
    global blue_center
    global blue_distance
    global prev_blue_distance

    prev_red_distance = red_distance
    prev_blue_distance = blue_distance

    color_image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()

    if color_image is None or depth_image is None:
        red_center = None
        red_distance = 0
        blue_center = None
        blue_distance = 0
        print("No image found")
        return

    # Search for the red cone
    contours = rc_utils.find_contours(color_image, RED[0], RED[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

    if contour is not None:
        red_center = rc_utils.get_contour_center(contour)
        red_distance = rc_utils.get_pixel_average_distance(depth_image, red_center)

        # Only use count it if the cone is less than MAX_DISTANCE away
        if red_distance <= MAX_DISTANCE:
            rc_utils.draw_contour(color_image, contour, rc_utils.ColorBGR.green.value)
            rc_utils.draw_circle(color_image, red_center, rc_utils.ColorBGR.green.value)
        else:
            red_center = None
            red_distance = 0
    else:
        red_center = None
        red_distance = 0

    # Search for the blue cone
    contours = rc_utils.find_contours(color_image, BLUE[0], BLUE[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

    if contour is not None:
        blue_center = rc_utils.get_contour_center(contour)
        blue_distance = rc_utils.get_pixel_average_distance(depth_image, blue_center)

        # Only use count it if the cone is less than MAX_DISTANCE away
        if blue_distance <= MAX_DISTANCE:
            rc_utils.draw_contour(color_image, contour, rc_utils.ColorBGR.yellow.value)
            rc_utils.draw_circle(
                color_image, blue_center, rc_utils.ColorBGR.yellow.value
            )
        else:
            blue_center = None
            blue_distance = 0
    else:
        blue_center = None
        blue_distance = 0

def cone_slalom():
    global cur_mode_slalom
    global counter

    find_cones()
    #print(cur_mode_slalom)

    angle: float
    speed: float

    # Align ourselves to smoothly approach and pass the red cone while it is in view
    if cur_mode_slalom == Mode_slalom.red_align:
        # Once the red cone is out of view, enter Mode.red_pass
        if (
            red_center is None
            or red_distance == 0
            or red_distance - prev_red_distance > CLOSE_DISTANCE
        ):
            if 0 < prev_red_distance < FAR_DISTANCE:
                counter = max(SHORT_PASS_TIME, counter)
                cur_mode_slalom = Mode_slalom.red_pass
            else:
                cur_mode_slalom = Mode_slalom.no_cones

        # If it seems like we are not going to make the turn, enter Mode.red_reverse
        elif (
            red_distance < REVERSE_DISTANCE
            and red_center[1] > rc.camera.get_width() // 4
        ):
            counter = REVERSE_BRAKE_TIME
            cur_mode_slalom = Mode_slalom.red_reverse

        # Align with the cone so that it gets closer to the left side of the screen
        # as we get closer to it, and slow down as we approach
        else:
            goal_point = rc_utils.remap_range(
                red_distance,
                CLOSE_DISTANCE,
                FAR_DISTANCE,
                0,
                rc.camera.get_width() // 4,
                True,
            )

            angle = rc_utils.remap_range(
                red_center[1], goal_point, rc.camera.get_width() // 2, 0, 1
            )
            angle = rc_utils.clamp(angle, -1, 1)

            speed = rc_utils.remap_range(
                red_distance,
                CLOSE_DISTANCE,
                FAR_DISTANCE,
                MIN_ALIGN_SPEED,
                MAX_ALIGN_SPEED,
                True,
            )

    elif cur_mode_slalom == Mode_slalom.blue_align:
        if (
            blue_center is None
            or blue_distance == 0
            or blue_distance - prev_blue_distance > CLOSE_DISTANCE
        ):
            if 0 < prev_blue_distance < FAR_DISTANCE:
                counter = max(SHORT_PASS_TIME, counter)
                cur_mode_slalom = Mode_slalom.blue_pass
            else:
                cur_mode_slalom = Mode_slalom.no_cones
        elif (
            blue_distance < REVERSE_DISTANCE
            and blue_center[1] < rc.camera.get_width() * 3 // 4
        ):
            counter = REVERSE_BRAKE_TIME
            cur_mode_slalom = Mode_slalom.blue_reverse
        else:
            goal_point = rc_utils.remap_range(
                blue_distance,
                CLOSE_DISTANCE,
                FAR_DISTANCE,
                rc.camera.get_width(),
                rc.camera.get_width() * 3 // 4,
                True,
            )

            angle = rc_utils.remap_range(
                blue_center[1], goal_point, rc.camera.get_width() // 2, 0, -1
            )
            angle = rc_utils.clamp(angle, -1, 1)

            speed = rc_utils.remap_range(
                blue_distance,
                CLOSE_DISTANCE,
                FAR_DISTANCE,
                MIN_ALIGN_SPEED,
                MAX_ALIGN_SPEED,
                True,
            )

    # Curve around the cone at a fixed speed for a fixed time to pass it
    if cur_mode_slalom == Mode_slalom.red_pass:
        angle = rc_utils.remap_range(counter, 1, 0, 0, -0.5)
        speed = PASS_SPEED
        counter -= rc.get_delta_time()

        # After the counter expires, enter Mode.blue_align if we see the blue cone,
        # and Mode.blue_find if we do not
        if counter <= 0:
            cur_mode_slalom = Mode_slalom.blue_align if blue_distance > 0 else Mode_slalom.blue_find

    elif cur_mode_slalom == Mode_slalom.blue_pass:
        angle = rc_utils.remap_range(counter, 1, 0, 0, 0.5)
        speed = PASS_SPEED

        counter -= rc.get_delta_time()
        if counter <= 0:
            cur_mode_slalom = Mode_slalom.red_align if red_distance > 0 else Mode_slalom.red_find

    # If we know we are supposed to be aligning with a red cone but do not see one,
    # turn to the right until we find it
    elif cur_mode_slalom == Mode_slalom.red_find:
        angle = 1
        speed = FIND_SPEED
        if red_distance > 0:
            cur_mode_slalom = Mode_slalom.red_align

    elif cur_mode_slalom == Mode_slalom.blue_find:
        angle = -1
        speed = FIND_SPEED
        if blue_distance > 0:
            cur_mode_slalom = Mode_slalom.blue_align

    # If we are not going to make the turn, reverse while keeping the cone in view
    elif cur_mode_slalom == Mode_slalom.red_reverse:
        if counter >= 0:
            counter -= rc.get_delta_time()
            speed = -1
            angle = 1
        else:
            angle = -1
            speed = REVERSE_SPEED
            if (
                red_distance > STOP_REVERSE_DISTANCE
                or red_center[1] < rc.camera.get_width() // 10
            ):
                counter = LONG_PASS_TIME
                cur_mode_slalom = Mode_slalom.red_align

    elif cur_mode_slalom == Mode_slalom.blue_reverse:
        if counter >= 0:
            counter -= rc.get_delta_time()
            speed = -1
            angle = 1
        else:
            angle = 1
            speed = REVERSE_SPEED
            if (
                blue_distance > STOP_REVERSE_DISTANCE
                or blue_center[1] > rc.camera.get_width() * 9 / 10
            ):
                counter = LONG_PASS_TIME
                cur_mode_slalom = Mode_slalom.blue_align

    # If no cones are seen, drive forward until we see either a red or blue cone
    elif cur_mode_slalom == Mode_slalom.no_cones:
        angle = 0
        speed = NO_CONES_SPEED

        if red_distance > 0 and blue_distance == 0:
            cur_mode_slalom = Mode_slalom.red_align
        elif blue_distance > 0 and red_distance == 0:
            cur_mode_slalom = Mode_slalom.blue_align
        elif blue_distance > 0 and red_distance > 0:
            cur_mode_slalom = (
                Mode_slalom.red_align if red_distance < blue_distance else Mode_slalom.blue_align
            )

    print(
        f"Mode: {cur_mode_slalom.name}, red_distance: {red_distance:.2f} cm, blue_distance: {blue_distance:.2f} cm, speed: {speed:.2f}, angle: {angle:2f}"
    )

    return angle, speed


def wall_follow_lab4b():
    global cur_mode
    global pre_value
    global panic_count

    scan = rc.lidar.get_samples()
    scan_ave = np.zeros(360*2)
    speed = 0
    angle = 0
    for angle in range(360*2):
        scan_ave[angle] = rc_utils.get_lidar_average_distance(scan, angle, WINDOW_ANGLE)

    # Find the minimum distance to the front, side, and rear of the car
    front_angle, front_dist = rc_utils.get_lidar_closest_point(
        scan, (-MIN_SIDE_ANGLE, MIN_SIDE_ANGLE)
    )
    left_angle, left_dist = rc_utils.get_lidar_closest_point(
        scan, (-MAX_SIDE_ANGLE, -MIN_SIDE_ANGLE)
    )
    right_angle, right_dist = rc_utils.get_lidar_closest_point(
        scan, (MIN_SIDE_ANGLE, MAX_SIDE_ANGLE)
    )

    # Estimate the left wall angle relative to the car by comparing the distance
    # to the left-front and left-back
    left_front_dist = rc_utils.get_lidar_average_distance(
        scan, -SIDE_FRONT_ANGLE, WINDOW_ANGLE
    )
    left_back_dist = rc_utils.get_lidar_average_distance(
        scan, -SIDE_BACK_ANGLE, WINDOW_ANGLE
    )
    left_dif = left_front_dist - left_back_dist

    # Use the same process for the right wall angle
    right_front_dist = rc_utils.get_lidar_average_distance(
        scan, SIDE_FRONT_ANGLE, WINDOW_ANGLE
    )
    right_back_dist = rc_utils.get_lidar_average_distance(
        scan, SIDE_BACK_ANGLE, WINDOW_ANGLE
    )
    right_dif = right_front_dist - right_back_dist

    if front_dist < PANIC_DISTANCE:
        cur_mode = Mode.front_panic
        panic_count = 0
    elif left_dist < SUPER_PANIC_DISTANCE or right_dist < SUPER_PANIC_DISTANCE:
        cur_mode = Mode.super_left_panic if left_dist < right_dist else Mode.super_right_panic
    # If we are within PANIC_DISTANCE of either wall, enter panic mode
    elif (left_dist < PANIC_DISTANCE or right_dist < PANIC_DISTANCE) and (cur_mode != Mode.super_left_panic or cur_mode != Mode.super_right_panic):
        cur_mode = Mode.left_panic if left_dist < right_dist else Mode.right_panic
        panic_count = 0
    elif left_dist > WIDE_DISTANCE and right_dist > WIDE_DISTANCE:
        cur_mode = Mode.wide
        panic_count = 0

    

    # If there are no visible walls to follow, stop the car
    if left_front_dist == 0.0 and right_front_dist == 0.0:
        speed = 0
        angle = 0

    elif cur_mode == Mode.front_panic:
        angle = 0
        speed = -SUPER_PANIC_SPEED

        if front_dist > END_PANIC_DISTANCE:
            cur_mode = Mode.align

    # LEFT PANIC: We are close to hitting a wall to the left, so turn hard right
    elif cur_mode == Mode.left_panic:
        angle = 1
        speed = PANIC_SPEED

        if left_dist > END_PANIC_DISTANCE:
            cur_mode = Mode.align

    # RIGHT PANIC: We are close to hitting a wall to the right, so turn hard left
    elif cur_mode == Mode.right_panic:
        angle = -1
        speed = PANIC_SPEED

        if right_dist > END_PANIC_DISTANCE:
            cur_mode = Mode.align
    
    elif cur_mode == Mode.super_left_panic:
        if panic_count < 5:
            angle = 0
        else:
            angle = -0.8
        speed = -SUPER_PANIC_SPEED
        print("panic = ", panic_count)
        panic_count += 1

        if left_dist > END_PANIC_DISTANCE:
            cur_mode = Mode.align
            panic_count = 0

    elif cur_mode == Mode.super_right_panic:
        if panic_count < 5:
            angle = 0
        else:
            angle = 0.8
        speed = -SUPER_PANIC_SPEED
        print("panic = ", panic_count)
        panic_count += 1

        if right_dist > END_PANIC_DISTANCE:
            cur_mode = Mode.align
            panic_count = 0
    
    # ALIGN: Try to align straight and equidistant between the left and right walls
    else:
        value = (right_dif - left_dif) + (right_dist - left_dist)
        #value = (pre_value + value)/2
        panic_count = 0

        # Choose speed based on the distance of the object in front of the car
        speed = rc_utils.remap_range(front_dist, 0, BRAKE_DISTANCE, 0, MAX_SPEED, True)

        # If left_dif is very large, the hallway likely turns to the left
        if left_dif > TURN_THRESHOLD:
            angle = -1
            speed *= 0.8
            print("Turn left")

        # Similarly, if right_dif is very large, the hallway likely turns to the right
        elif right_dif > TURN_THRESHOLD:
            angle = 1
            speed *= 0.8
            print("Turn right")

        # Otherwise, determine angle by taking into account both the relative angles and
        # distances of the left and right walls
        else:
            if cur_mode == Mode.wide:
                angle = 0
            else:
                angle = rc_utils.remap_range(
                    value, -TURN_THRESHOLD, TURN_THRESHOLD, -1, 1, True
                )
    
    return angle, speed

def wall_follow2(direction):
    """
    Determines the angle of the wheels necessary to follow the wall on the side
    specified by the direction parameter.

    Uses a similar strategy to Lab 5B.
    """
    lidar_scan = rc.lidar.get_samples()

    # Measure 3 points along the wall we are trying to follow
    side_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, SIDE_ANGLE * direction, WINDOW_ANGLE
    )
    side_front_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, (SIDE_ANGLE - SPREAD_ANGLE) * direction, WINDOW_ANGLE
    )
    side_back_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, (SIDE_ANGLE + SPREAD_ANGLE) * direction, WINDOW_ANGLE
    )
    front_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, 0, WINDOW_ANGLE
    )
     # Estimate the left wall angle relative to the car by comparing the distance
    # to the left-front and left-back
    left_front_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, -SIDE_FRONT_ANGLE, WINDOW_ANGLE
    )
    left_back_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, -SIDE_BACK_ANGLE, WINDOW_ANGLE
    )
    left_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, -SIDE_ANGLE, WINDOW_ANGLE
    )
    right_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, SIDE_ANGLE, WINDOW_ANGLE
    )

    left_dif = left_front_dist - left_back_dist

    # Use the same process for the right wall angle
    right_front_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, SIDE_FRONT_ANGLE, WINDOW_ANGLE
    )
    right_back_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, SIDE_BACK_ANGLE, WINDOW_ANGLE
    )
    right_dif = right_front_dist - right_back_dist
    # print("front_dist = ", front_dist)

    # Determine a goal angle based on how we are aligned with the wall
    dif_component = rc_utils.remap_range(
        side_front_dist - side_back_dist, -MAX_DIF, MAX_DIF, -1, 1, True
    )

    # Determine a goal angle based on how far we are from the wall
    distance_component = rc_utils.remap_range(
        WALL_DISTANCE - side_dist, -MAX_DIF, MAX_DIF, 1, -1, True
    )
    angle = dif_component + distance_component * DISTANCE_COEFFICIENT
    speed = WALL_FOLLOW_SPEED
    if 0 < min(front_dist, right_front_dist, left_front_dist) < COLLISION_DISTANCE:
        if left_front_dist > right_front_dist:
            speed = -1 * PANIC_SPEED
            angle = 1 * direction
            print("collision -> left")
        else:
            speed = -1 * PANIC_SPEED
            angle = -1 * direction
            print("collision -> right")
    # If left_dif is very large, the hallway likely turns to the left
    elif 0 < front_dist < TURN_DISTANCE:
        if (left_dif > TURN_THRESHOLD or left_dist > IS_WALL_DISTANCE) and right_front_dist < IS_WALL_DISTANCE:
            angle = -1
            speed = PANIC_SPEED
            print("Turn left")
    
        # Similarly, if right_dif is very large, the hallway likely turns to the right
        elif (right_dif > TURN_THRESHOLD or right_dist > IS_WALL_DISTANCE) and left_front_dist < IS_WALL_DISTANCE:
            angle = 1
            speed = PANIC_SPEED
            print("Turn right")
    # elif 0 < front_dist < TURN_DISTANCE:
    #     angle = -1
    #     speed = PANIC_SPEED
    #     print("turn")
    # elif left_front_dist > S_DISTANCE and right_back_dist > S_DISTANCE:
    #     angle = -1
    #     speed = PANIC_SPEED
    #     print("S shape left")
    # elif right_front_dist > S_DISTANCE and left_back_dist > S_DISTANCE:
    #     angle = 1
    #     speed = PANIC_SPEED
    #     print("S shape right")
    elif left_dif > TURN_THRESHOLD and right_front_dist < WALL_DISTANCE:
        dif_component = rc_utils.remap_range(
        right_dif, -MAX_DIF, MAX_DIF, -1, 1, True
        )
        distance_component = rc_utils.remap_range(
        WALL_DISTANCE - right_front_dist, -MAX_DIF, MAX_DIF, 1, -1, True
        )
        angle = dif_component + distance_component * DISTANCE_COEFFICIENT
        speed = WALL_FOLLOW_SPEED
        print("follow right")
    elif right_dif > TURN_THRESHOLD and left_front_dist < WALL_DISTANCE:
        dif_component = rc_utils.remap_range(
        left_dif, -MAX_DIF, MAX_DIF, -1, 1, True
        )
        distance_component = rc_utils.remap_range(
        WALL_DISTANCE - left_front_dist, -MAX_DIF, MAX_DIF, 1, -1, True
        )
        angle = dif_component + distance_component * DISTANCE_COEFFICIENT
        speed = WALL_FOLLOW_SPEED
        print("follow left")  
        
        # print("angle = ", angle)
    

    return speed, direction * rc_utils.clamp(angle, -1, 1)

# def wall_follow(direction):
#     """
#     Determines the angle of the wheels necessary to follow the wall on the side
#     specified by the direction parameter.

#     Uses a similar strategy to Lab 5B.
#     """
#     lidar_scan = rc.lidar.get_samples()

#     # Measure 3 points along the wall we are trying to follow
#     side_dist = rc_utils.get_lidar_average_distance(
#         lidar_scan, SIDE_ANGLE * direction, WINDOW_ANGLE
#     )
#     side_front_dist = rc_utils.get_lidar_average_distance(
#         lidar_scan, (SIDE_ANGLE - SPREAD_ANGLE) * direction, WINDOW_ANGLE
#     )
#     side_back_dist = rc_utils.get_lidar_average_distance(
#         lidar_scan, (SIDE_ANGLE + SPREAD_ANGLE) * direction, WINDOW_ANGLE
#     )
#     front_dist = rc_utils.get_lidar_average_distance(
#         lidar_scan, -WINDOW_ANGLE/2, WINDOW_ANGLE
#     )
#      # Estimate the left wall angle relative to the car by comparing the distance
#     # to the left-front and left-back
#     left_front_dist = rc_utils.get_lidar_average_distance(
#         lidar_scan, -SIDE_FRONT_ANGLE, WINDOW_ANGLE
#     )
#     left_back_dist = rc_utils.get_lidar_average_distance(
#         lidar_scan, -SIDE_BACK_ANGLE, WINDOW_ANGLE
#     )
#     left_dif = left_front_dist - left_back_dist

#     # Use the same process for the right wall angle
#     right_front_dist = rc_utils.get_lidar_average_distance(
#         lidar_scan, SIDE_FRONT_ANGLE, WINDOW_ANGLE
#     )
#     right_back_dist = rc_utils.get_lidar_average_distance(
#         lidar_scan, SIDE_BACK_ANGLE, WINDOW_ANGLE
#     )
#     right_dif = right_front_dist - right_back_dist
#     print("front_dist = ", front_dist)

#     # Determine a goal angle based on how we are aligned with the wall
#     dif_component = rc_utils.remap_range(
#         side_front_dist - side_back_dist, -MAX_DIF, MAX_DIF, -1, 1, True
#     )

#     # Determine a goal angle based on how far we are from the wall
#     distance_component = rc_utils.remap_range(
#         WALL_DISTANCE - side_dist, -MAX_DIF, MAX_DIF, 1, -1, True
#     )
#     if 0 < front_dist < COLLISION_DISTANCE:
#         speed = -1 * PANIC_SPEED
#         angle = 1
#         print("collision")
#     # If left_dif is very large, the hallway likely turns to the left
#     elif 0 < front_dist < TURN_DISTANCE and left_dif > TURN_THRESHOLD and side_dist <= WALL_DISTANCE:
#         angle = -1
#         speed = PANIC_SPEED
#         print("Turn left")

#     # Similarly, if right_dif is very large, the hallway likely turns to the right
#     elif 0 < front_dist < TURN_DISTANCE and right_dif > TURN_THRESHOLD:
#         angle = 1
#         speed = PANIC_SPEED
#         print("Turn right")
#     # elif 0 < front_dist < TURN_DISTANCE:
#     #     angle = -1
#     #     speed = PANIC_SPEED
#     #     print("turn")
#     else:    
#         # Take a linear combination of the two goal angles
#         angle = dif_component + distance_component * DISTANCE_COEFFICIENT
#         print("angle = ", angle)
#         speed = WALL_FOLLOW_SPEED
#     return speed, direction * rc_utils.clamp(angle, -1, 1)


def line_follow(color_image):
    """
    Determines the angle of the wheels necessary to follow the colored line
    on the floor, with color priority specified by the colors parameter.

    Uses a similar strategy to Lab 2A.
    """
    global cur_color

    # Crop to the floor directly in front of the car
    cropped_image = rc_utils.crop(color_image, CROP_WINDOW[0], CROP_WINDOW[1])

    # Search for the colors in priority order
    #for color in colors:
    # Find the largest contour of the specified color
    contours = rc_utils.find_contours(cropped_image, cur_color[0], cur_color[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

    # If the contour exists, steer toward the center of the contour
    if contour is not None:
        center = rc_utils.get_contour_center(contour)
        rc_utils.draw_contour(cropped_image, contour)
        rc_utils.draw_circle(cropped_image, center)

        # Display the image to the screen
        rc.display.show_color_image(cropped_image)
        return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)

    # If no color lines are found, return None so that we wall follow instead
    return None


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()