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
import time

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

class Mode(enum.IntEnum):
    align = 0
    right_close = 1
    left_close = 2
    super_right_close = 3
    super_left_close = 4
    front_close = 5
    wide = 6

class Mode_color(enum.IntEnum):
    red_align = 0
    blue_align = 1
    red_pass = 2
    blue_pass = 3
    red_find = 4
    blue_find = 5
    red_reverse = 6
    blue_reverse = 7
    no_cones = 8

speed = 0
stop_counter = 0
MAX_SPEED = 1.0
BRAKE_DISTANCE = 150
SUPER_close_DISTANCE = 21
close_DISTANCE = 20
END_close_DISTANCE = 25
WIDE_DISTANCE = 130
close_SPEED = 0.3
SUPER_close_SPEED = 1.0
MIN_SIDE_ANGLE = 10
MAX_SIDE_ANGLE = 75
SIDE_FRONT_ANGLE = 70
SIDE_BACK_ANGLE = 110
TURN_THRESHOLD = 30
cur_mode = Mode.align
pre_value = 0
close_count = 0
MAX_ALIGN_SPEED = 0.8
MIN_ALIGN_SPEED = 0.4
PASS_SPEED = 0.5
FIND_SPEED = 0.2
REVERSE_SPEED = -0.2
NO_CONES_SPEED = 0.4
REVERSE_BRAKE_TIME = 0.25
SHORT_PASS_TIME = 1.0
LONG_PASS_TIME = 1.2
MIN_CONTOUR_AREA = 100
MAX_DISTANCE = 250
REVERSE_DISTANCE = 50
STOP_REVERSE_DISTANCE = 60
CLOSE_DISTANCE = 30
FAR_DISTANCE = 120
cur_mode_color = Mode_color.no_cones
counter = 0
red_center = None
red_distance = 0
prev_red_distance = 0
blue_center = None
blue_distance = 0
prev_blue_distance = 0

class course(enum.IntEnum):
    none = 0
    marker_id = 1
    orientation = 2
    color = 3
    cone = 4
    lane = 5
    wall = 6

RED = ((170, 50, 50), (10, 255, 255), "red")
BLUE = ((100, 150, 150), (120, 255, 255), "blue")
GREEN = ((40, 50, 50), (80, 255, 255), "green")
ORANGE = ((10, 100, 100), (25, 255, 255), "orange")
PURPLE = ((125, 100, 100), (150, 255, 255), "purple")
colors = [RED, BLUE, GREEN, ORANGE, PURPLE]
colors_lane = [ORANGE, PURPLE]
SIDE_ANGLE = 90
SPREAD_ANGLE = 30
WINDOW_ANGLE = 60
WALL_DISTANCE = 36
IS_WALL_DISTANCE = 70
S_DISTANCE = 50
TURN_DISTANCE = 70
COLLISION_DISTANCE = 30
MAX_DIF = 10
DISTANCE_COEFFICIENT = 0.5
MIN_CONTOUR_AREA = 30
MIN_CONTOUR_AREA_LANE = 500
CROP_WINDOW = ((300, 0), (rc.camera.get_height(), rc.camera.get_width()))
CROP_WINDOW_lane = ((rc.camera.get_height() * 2 // 3, 0),(rc.camera.get_height(), rc.camera.get_width()))
CROP_WINDOW_lane_right = ((280, int(rc.camera.get_width()/2)), (rc.camera.get_height() - 100, rc.camera.get_width()))
CROP_WINDOW_lane_left = ((280, 0), (rc.camera.get_height() - 100, int(rc.camera.get_width()/2)))
CROP_FLOOR = (
    (rc.camera.get_height() * 2 // 3, 0),
    (rc.camera.get_height(), rc.camera.get_width()),
)

EXPLORE_SPEED = 0.4
WALL_FOLLOW_SPEED = 0.5
LINE_FOLLOW_SPEED = 0.7
LANE_FOLLOW_SPEED = 0.4
MAX_SPEED = 0.35
FAST_SPEED = 1.0
SLOW_SPEED = 0.7
ONE_LANE_TURN_ANGLE = 0.6
cur_course = course.none
cur_marker = rc_utils.ARMarker(-1, np.zeros((4, 2), dtype=np.int32))
cur_direction = 0
cur_color = None
pre_center = None
primary_color = ORANGE
secondary_color = PURPLE

########################################################################################
# Functions
########################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    global cur_course
    global cur_marker
    global cur_direction
    global cur_color
    global cur_mode_color
    global counter
    global speed
    global stop_counter

    rc.drive.stop()
    cur_course = course.none
    cur_marker = rc_utils.ARMarker(-1, np.zeros((4, 2), dtype=np.int32))
    cur_direction = 1
    cur_color = None
    cur_mode_color = Mode_color.no_cones
    counter = 0
    speed = 0
    stop_counter = 0


def course_shift(new_marker, color_image):
    global cur_course
    global cur_marker
    global cur_direction
    global cur_color
    global colors
    global primary_color
    global secondary_color


    cur_marker = new_marker

    if cur_marker.get_id() == 1:
        cur_course = course.lane
        cur_direction = 1
        if cur_color != primary_color[2]:
                temp = primary_color
                primary_color = secondary_color
                secondary_color = temp

    elif cur_marker.get_id() == 199:
        cur_course = course.orientation
        cur_direction = (
            1 if cur_marker.get_orientation() == rc_utils.Orientation.RIGHT else -1
        )

    elif cur_marker.get_id() == 0:
        cur_course = course.color

    elif cur_marker.get_id() == 2:
        cur_course = course.cone
        cur_direction = 1

    elif cur_marker.get_id() == 3:
        cur_course = course.wall

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

def lane_follow1():
    speed = SLOW_SPEED
    angle = 0

    image = rc.camera.get_color_image()
    if image is None:
        rc.drive.stop()
        return


    image = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])


    contours = [
        contour
        for contour in rc_utils.find_contours(
            image, secondary_color[0], secondary_color[1]
        )
        if cv.contourArea(contour) > MIN_CONTOUR_AREA
    ]

    if len(contours) == 0:

        contours = [
            contour
            for contour in rc_utils.find_contours(
                image, primary_color[0], primary_color[1]
            )
            if cv.contourArea(contour) > MIN_CONTOUR_AREA
        ]
        if len(contours) == 0:
            return SLOW_SPEED, 0

        else:
            speed = FAST_SPEED

    if len(contours) >= 2:

        contours.sort(key=cv.contourArea, reverse=True)


        first_center = rc_utils.get_contour_center(contours[0])
        second_center = rc_utils.get_contour_center(contours[1])
        midpoint = (first_center[1] + second_center[1]) / 2


        angle = rc_utils.remap_range(midpoint, 0, rc.camera.get_width(), -1, 1)


        rc_utils.draw_contour(image, contours[0], rc_utils.ColorBGR.red.value)
        rc_utils.draw_circle(image, first_center, rc_utils.ColorBGR.red.value)
        rc_utils.draw_contour(image, contours[1], rc_utils.ColorBGR.blue.value)
        rc_utils.draw_circle(image, second_center, rc_utils.ColorBGR.blue.value)


    else:
        contour = contours[0]
        center = rc_utils.get_contour_center(contour)

        if center[1] > rc.camera.get_width() / 2:
            angle = -ONE_LANE_TURN_ANGLE

        else:
            angle = ONE_LANE_TURN_ANGLE

        rc_utils.draw_contour(image, contour)
        rc_utils.draw_circle(image, center)


    rc.display.show_color_image(image)

    return speed, angle

def lane_follow(color_image):
    global colors_lane
    global pre_center

    cropped_image = rc_utils.crop(color_image, CROP_WINDOW_lane[0], CROP_WINDOW_lane[1])

    for color in colors_lane:
        contours = rc_utils.find_contours(cropped_image, color[0], color[1])
        contour, second_contour = largest_contour(contours, MIN_CONTOUR_AREA)


        if contour is not None and second_contour is not None:
            center = rc_utils.get_contour_center(contour)
            second_center = rc_utils.get_contour_center(second_contour)
            center = (int((center[0] + second_center[0]) / 2), int((center[1] + second_center[1]) / 2))
            rc_utils.draw_contour(cropped_image, contour)
            rc_utils.draw_contour(cropped_image, second_contour, (255, 0, 0))
            rc_utils.draw_circle(cropped_image, center)
            rc.display.show_color_image(cropped_image)
            return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)


    return None

def lane_follow2(color_image):
    global colors_lane
    global pre_center

    cropped_image_left = rc_utils.crop(color_image, CROP_WINDOW_lane_left[0], CROP_WINDOW_lane_left[1])
    cropped_image_right = rc_utils.crop(color_image, CROP_WINDOW_lane_right[0], CROP_WINDOW_lane_right[1])

    for color in colors_lane:
        contours = rc_utils.find_contours(cropped_image_right, color[0], color[1])
        contour_right = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        if contour_right is not None:
            center_right = rc_utils.get_contour_center(contour_right)
            rc_utils.draw_contour(cropped_image_right, contour_right)
            rc_utils.draw_circle(cropped_image_right, center_right)
            rc.display.show_color_image(cropped_image_right)

    for color in colors_lane:
        contours = rc_utils.find_contours(cropped_image_left, color[0], color[1])
        contour_left = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        if contour_left is not None:
            center_left = rc_utils.get_contour_center(contour_left)
            rc_utils.draw_contour(cropped_image_left, contour_left)
            rc_utils.draw_circle(cropped_image_left, center_left)
            rc.display.show_color_image(cropped_image_left)

    if contour_right is not None and contour_left is not None:
        center = ((center_left[0] + rc.camera.get_width()/2 + center_right[0]) / 2, (center_left[1] + center_right[1]) / 2)
        return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)

    elif contour_left is not None:
        center = (center_left[0] + 200, center_left[1])
        return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)

    elif contour_right is not None:
        center = (rc.camera.get_width()/2 + center_right[0] - 200, center_right[1])
        return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)

    else:
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

def lane_follow_Average(color_image):
    global colors_lane
    global pre_center

    cropped_image = rc_utils.crop(color_image, CROP_WINDOW_lane[0], CROP_WINDOW_lane[1])

    for color in colors_lane:
        contours = rc_utils.find_contours(cropped_image, color[0], color[1])
        center_sum = (0, 0)

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

    return None

def find_cones():
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
        return

    contours = rc_utils.find_contours(color_image, RED[0], RED[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

    if contour is not None:
        red_center = rc_utils.get_contour_center(contour)
        red_distance = rc_utils.get_pixel_average_distance(depth_image, red_center)

        if red_distance <= MAX_DISTANCE:
            rc_utils.draw_contour(color_image, contour, rc_utils.ColorBGR.green.value)
            rc_utils.draw_circle(color_image, red_center, rc_utils.ColorBGR.green.value)
        else:
            red_center = None
            red_distance = 0
    else:
        red_center = None
        red_distance = 0

    contours = rc_utils.find_contours(color_image, BLUE[0], BLUE[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

    if contour is not None:
        blue_center = rc_utils.get_contour_center(contour)
        blue_distance = rc_utils.get_pixel_average_distance(depth_image, blue_center)


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

def cone_color():
    global cur_mode_color
    global counter

    find_cones()

    angle: float
    speed: float


    if cur_mode_color == Mode_color.red_align:

        if (
            red_center is None
            or red_distance == 0
            or red_distance - prev_red_distance > CLOSE_DISTANCE
        ):
            if 0 < prev_red_distance < FAR_DISTANCE:
                counter = max(SHORT_PASS_TIME, counter)
                cur_mode_color = Mode_color.red_pass
            else:
                cur_mode_color = Mode_color.no_cones


        elif (
            red_distance < REVERSE_DISTANCE
            and red_center[1] > rc.camera.get_width() // 4
        ):
            counter = REVERSE_BRAKE_TIME
            cur_mode_color = Mode_color.red_reverse


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

    elif cur_mode_color == Mode_color.blue_align:
        if (
            blue_center is None
            or blue_distance == 0
            or blue_distance - prev_blue_distance > CLOSE_DISTANCE
        ):
            if 0 < prev_blue_distance < FAR_DISTANCE:
                counter = max(SHORT_PASS_TIME, counter)
                cur_mode_color = Mode_color.blue_pass
            else:
                cur_mode_color = Mode_color.no_cones
        elif (
            blue_distance < REVERSE_DISTANCE
            and blue_center[1] < rc.camera.get_width() * 3 // 4
        ):
            counter = REVERSE_BRAKE_TIME
            cur_mode_color = Mode_color.blue_reverse
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
    if cur_mode_color == Mode_color.red_pass:
        angle = rc_utils.remap_range(counter, 1, 0, 0, -0.5)
        speed = PASS_SPEED
        counter -= rc.get_delta_time()

        if counter <= 0:
            cur_mode_color = Mode_color.blue_align if blue_distance > 0 else Mode_color.blue_find

    elif cur_mode_color == Mode_color.blue_pass:
        angle = rc_utils.remap_range(counter, 1, 0, 0, 0.5)
        speed = PASS_SPEED

        counter -= rc.get_delta_time()
        if counter <= 0:
            cur_mode_color = Mode_color.red_align if red_distance > 0 else Mode_color.red_find


    elif cur_mode_color == Mode_color.red_find:
        angle = 1
        speed = FIND_SPEED
        if red_distance > 0:
            cur_mode_color = Mode_color.red_align

    elif cur_mode_color == Mode_color.blue_find:
        angle = -1
        speed = FIND_SPEED
        if blue_distance > 0:
            cur_mode_color = Mode_color.blue_align

    elif cur_mode_color == Mode_color.red_reverse:
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
                cur_mode_color = Mode_color.red_align

    elif cur_mode_color == Mode_color.blue_reverse:
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
                cur_mode_color = Mode_color.blue_align

    elif cur_mode_color == Mode_color.no_cones:
        angle = 0
        speed = NO_CONES_SPEED

        if red_distance > 0 and blue_distance == 0:
            cur_mode_color = Mode_color.red_align
        elif blue_distance > 0 and red_distance == 0:
            cur_mode_color = Mode_color.blue_align
        elif blue_distance > 0 and red_distance > 0:
            cur_mode_color = (
                Mode_color.red_align if red_distance < blue_distance else Mode_color.blue_align
            )

    return angle, speed


def wall_follow1():
    global cur_mode
    global pre_value
    global close_count

    scan = rc.lidar.get_samples()
    scan_ave = np.zeros(360*2)
    speed = 0
    angle = 0
    for angle in range(360*2):
        scan_ave[angle] = rc_utils.get_lidar_average_distance(scan, angle, WINDOW_ANGLE)

    front_angle, front_dist = rc_utils.get_lidar_closest_point(
        scan, (-MIN_SIDE_ANGLE, MIN_SIDE_ANGLE)
    )
    left_angle, left_dist = rc_utils.get_lidar_closest_point(
        scan, (-MAX_SIDE_ANGLE, -MIN_SIDE_ANGLE)
    )
    right_angle, right_dist = rc_utils.get_lidar_closest_point(
        scan, (MIN_SIDE_ANGLE, MAX_SIDE_ANGLE)
    )

    left_front_dist = rc_utils.get_lidar_average_distance(
        scan, -SIDE_FRONT_ANGLE, WINDOW_ANGLE
    )
    left_back_dist = rc_utils.get_lidar_average_distance(
        scan, -SIDE_BACK_ANGLE, WINDOW_ANGLE
    )
    left_dif = left_front_dist - left_back_dist

    right_front_dist = rc_utils.get_lidar_average_distance(
        scan, SIDE_FRONT_ANGLE, WINDOW_ANGLE
    )
    right_back_dist = rc_utils.get_lidar_average_distance(
        scan, SIDE_BACK_ANGLE, WINDOW_ANGLE
    )
    right_dif = right_front_dist - right_back_dist

    if front_dist < close_DISTANCE:
        cur_mode = Mode.front_close
        close_count = 0

    elif left_dist < SUPER_close_DISTANCE or right_dist < SUPER_close_DISTANCE:
        cur_mode = Mode.super_left_close if left_dist < right_dist else Mode.super_right_close

    elif (left_dist < close_DISTANCE or right_dist < close_DISTANCE) and (cur_mode != Mode.super_left_close or cur_mode != Mode.super_right_close):
        cur_mode = Mode.left_close if left_dist < right_dist else Mode.right_close
        close_count = 0

    elif left_dist > WIDE_DISTANCE and right_dist > WIDE_DISTANCE:
        cur_mode = Mode.wide
        close_count = 0

    if left_front_dist == 0.0 and right_front_dist == 0.0:
        speed = 0
        angle = 0

    elif cur_mode == Mode.front_close:
        angle = 0
        speed = -SUPER_close_SPEED

        if front_dist > END_close_DISTANCE:
            cur_mode = Mode.align

    elif cur_mode == Mode.left_close:
        angle = 1
        speed = close_SPEED

        if left_dist > END_close_DISTANCE:
            cur_mode = Mode.align

    elif cur_mode == Mode.right_close:
        angle = -1
        speed = close_SPEED

        if right_dist > END_close_DISTANCE:
            cur_mode = Mode.align

    elif cur_mode == Mode.super_left_close:
        if close_count < 5:
            angle = 0

        else:
            angle = -0.8
        speed = -SUPER_close_SPEED
        close_count += 1

        if left_dist > END_close_DISTANCE:
            cur_mode = Mode.align
            close_count = 0

    elif cur_mode == Mode.super_right_close:
        if close_count < 5:
            angle = 0

        else:
            angle = 0.8
        speed = -SUPER_close_SPEED
        close_count += 1

        if right_dist > END_close_DISTANCE:
            cur_mode = Mode.align
            close_count = 0

    else:
        value = (right_dif - left_dif) + (right_dist - left_dist)
        close_count = 0
        speed = rc_utils.remap_range(front_dist, 0, BRAKE_DISTANCE, 0, MAX_SPEED, True)
        if left_dif > TURN_THRESHOLD:
            angle = -1
            speed *= 0.8

        elif right_dif > TURN_THRESHOLD:
            angle = 1
            speed *= 0.8

        else:
            if cur_mode == Mode.wide:
                angle = 0

            else:
                angle = rc_utils.remap_range(
                    value, -TURN_THRESHOLD, TURN_THRESHOLD, -1, 1, True
                )

    return angle, speed

def wall_follow2(direction):
    lidar_scan = rc.lidar.get_samples()
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

    right_front_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, SIDE_FRONT_ANGLE, WINDOW_ANGLE
    )
    right_back_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, SIDE_BACK_ANGLE, WINDOW_ANGLE
    )
    right_dif = right_front_dist - right_back_dist

    dif_component = rc_utils.remap_range(
        side_front_dist - side_back_dist, -MAX_DIF, MAX_DIF, -1, 1, True
    )


    distance_component = rc_utils.remap_range(
        WALL_DISTANCE - side_dist, -MAX_DIF, MAX_DIF, 1, -1, True
    )
    angle = dif_component + distance_component * DISTANCE_COEFFICIENT
    speed = WALL_FOLLOW_SPEED
    if 0 < min(front_dist, right_front_dist, left_front_dist) < COLLISION_DISTANCE:
        if left_front_dist > right_front_dist:
            speed = -1 * close_SPEED
            angle = 1 * direction

        else:
            speed = -1 * close_SPEED
            angle = -1 * direction

    elif 0 < front_dist < TURN_DISTANCE:
        if (left_dif > TURN_THRESHOLD or left_dist > IS_WALL_DISTANCE) and right_front_dist < IS_WALL_DISTANCE:
            angle = -1
            speed = close_SPEED

        elif (right_dif > TURN_THRESHOLD or right_dist > IS_WALL_DISTANCE) and left_front_dist < IS_WALL_DISTANCE:
            angle = 1
            speed = close_SPEED

        elif left_dif > TURN_THRESHOLD and right_front_dist < WALL_DISTANCE:
            dif_component = rc_utils.remap_range(
            right_dif, -MAX_DIF, MAX_DIF, -1, 1, True
            )
        distance_component = rc_utils.remap_range(
        WALL_DISTANCE - right_front_dist, -MAX_DIF, MAX_DIF, 1, -1, True
        )
        angle = dif_component + distance_component * DISTANCE_COEFFICIENT
        speed = WALL_FOLLOW_SPEED

    elif right_dif > TURN_THRESHOLD and left_front_dist < WALL_DISTANCE:
        dif_component = rc_utils.remap_range(
        left_dif, -MAX_DIF, MAX_DIF, -1, 1, True
        )
        distance_component = rc_utils.remap_range(
        WALL_DISTANCE - left_front_dist, -MAX_DIF, MAX_DIF, 1, -1, True
        )
        angle = dif_component + distance_component * DISTANCE_COEFFICIENT
        speed = WALL_FOLLOW_SPEED

    return speed, direction * rc_utils.clamp(angle, -1, 1)

def line_follow(color_image):
    global cur_color

    cropped_image = rc_utils.crop(color_image, CROP_WINDOW[0], CROP_WINDOW[1])
    contours = rc_utils.find_contours(cropped_image, cur_color[0], cur_color[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
    if contour is not None:
        center = rc_utils.get_contour_center(contour)
        rc_utils.draw_contour(cropped_image, contour)
        rc_utils.draw_circle(cropped_image, center)
        rc.display.show_color_image(cropped_image)
        return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)
    return None

def update():

    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global cur_course
    global cur_marker
    global cur_direction
    global cur_color
    global speed
    global stop_counter
    global time

    passed_time = time.perf_counter()
    stop_counter = 0
    color_image = rc.camera.get_color_image_no_copy()
    markers = rc_utils.get_ar_markers(color_image)

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
        if area > 1500 and markers[0].get_id() != cur_marker.get_id():
            course_shift(markers[0], color_image)

    speed = 0
    angle = 0
    if cur_course == course.none:

        angle, speed =wall_follow1()
    elif cur_course == course.marker_id or cur_course == course.orientation or cur_course == course.wall:

        speed, angle = wall_follow2(cur_direction)
    elif cur_course == course.color:

        speed = LINE_FOLLOW_SPEED
        angle = line_follow(color_image)


        if angle is None:
            _, angle = wall_follow2(cur_direction)
    elif cur_course == course.cone:

        angle, speed = cone_color()

        if angle is None:
            _, angle = wall_follow2(cur_direction)
    elif cur_course == course.lane:

        speed, angle = lane_follow1()

        if angle is None:
            _, angle = wall_follow2(cur_direction)

    if 120 < passed_time < 124:
        rc.drive.set_speed_angle(-1, 0)


    if -0.1 < speed < 0.1:
        stop_counter += 1
        if stop_counter > 8:
            rc.drive.stop()
            rc.drive.set_speed_angle(1, -1) * 7

    rc.drive.set_speed_angle(speed, angle)



########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
