import sys
import cv2 as cv
import numpy as np
import enum

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar()

class Mode(enum.IntEnum):
    red_align = 0  # Approaching a red cone to pass
    blue_align = 1  # Approaching a blue cone to pass
    red_pass = 2  # Passing a red cone (currently out of sight to our left)
    blue_pass = 3  # Passing a blue cone (currently out of sight to our right)
    red_find = 4  # Finding a red cone with which to align
    blue_find = 5  # Finding a blue cone with which to align
    red_reverse = 6  # Aligning with a red cone, but we are too close so must back up
    blue_reverse = 7  # Aligning with a blue cone, but we are too close so must back up
    no_cones = 8  # No cones in sight, inch forward until we find one

class Stage(enum.IntEnum):
    none = 0
    marker_id = 1
    orientation = 2
    color = 3
    cone = 4
    lane = 5
    wall = 6

RED = ((170, 50, 50), (10, 255, 255), "red")  # The HSV range for the color blue
BLUE = ((100, 150, 150), (120, 255, 255), "blue")  # The HSV range for the color blue
GREEN = ((40, 50, 50), (80, 255, 255), "green")  # The HSV range for the color green
ORANGE = ((10, 100, 100), (25, 255, 255), "orange")
PURPLE = ((125, 100, 100), (150, 255, 255), "purple")

colors = [RED, BLUE, GREEN, ORANGE, PURPLE]
colors_lane = [ORANGE, PURPLE]
primary_color = ORANGE
secondary_color = PURPLE
cur_stage = Stage.none

cur_mode = Mode.no_cones
counter = 0
red_center = None
red_distance = 0
prev_red_distance = 0
blue_center = None
blue_distance = 0
prev_blue_distance = 0

x = True
timer = 0



def start():
    global cur_stage
    global cur_marker
    global cur_direction
    global cur_color
    global cur_mode_slalom
    global counter
    global timer

    rc.drive.stop()
    cur_stage = Stage.none
    cur_mode = Mode.no_cones
    cur_marker = rc_utils.ARMarker(-1, np.zeros((4, 2), dtype=np.int32))
    cur_direction = 1
    cur_color = None
    counter = 0
    timer = 0


    print(">> Final Challenge - Grand Prix")


def update():
    global cur_stage

    color_image = rc.camera.get_color_image()
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
            change_stage(markers[0], color_image)

    speed = 0
    angle = 0
    if cur_stage == Stage.none:
        speed, angle = none_mode()

    elif cur_stage == Stage.lane:
        speed, angle = lane_mode()

    elif cur_stage == Stage.wall:
        speed, angle = wall_mode()

    elif cur_stage == Stage.cone:
        speed, angle = cone_mode()

    elif cur_stage == Stage.color:
        speed = 1
        angle = color_mode()

    elif cur_stage ==Stage.orientation:
        speed, angle = orientation_mode(cur_direction)

    speed = rc_utils.clamp(speed, -1, 1)
    angle = rc_utils.clamp(angle, -1, 1)
    print(round(speed, 1), round(angle, 1), )
    rc.drive.set_speed_angle(speed,angle)

    if rc.controller.is_down(rc.controller.Button.X):
        a = b


def none_mode():
    scan = rc.lidar.get_samples()
    front_angle, front_dis = rc_utils.get_lidar_closest_point(scan, (-10, 10))
    left_dist = rc_utils.get_lidar_average_distance(scan, -50, 60)
    right_dist = rc_utils.get_lidar_average_distance(scan, 50, 60)
    
    speed = rc_utils.remap_range(front_dis, 40, 200, 0, 1)
    dif = right_dist - left_dist
    angle = rc_utils.remap_range(dif, -50, 50, 0, 1)

    speed = rc_utils.clamp(speed, -1, 1)
    angle = rc_utils.clamp(angle, -1, 1)
    return speed, angle


def lane_mode():
    speed = 1
    angle = 0

    image = rc.camera.get_color_image()
    if image is None:
        print("No image")
        rc.drive.stop()
        return

    image = rc_utils.crop(image, (rc.camera.get_height() * 2 // 3, 0), (rc.camera.get_height(), rc.camera.get_width()))

    contours = [
        contour
        for contour in rc_utils.find_contours(
            image, primary_color[0], primary_color[1]
        )
        if cv.contourArea(contour) > 1000
    ]

    if len(contours) == 0:
        contours = [
            contour
            for contour in rc_utils.find_contours(
                image, secondary_color[0], secondary_color[1]
            )
            if cv.contourArea(contour) > 1000
        ]
        if len(contours) == 0:
            return 0.5, 0
        else:
            speed = 1

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
            angle = -0.8
        else:
            angle = 0.8

        rc_utils.draw_contour(image, contour)
        rc_utils.draw_circle(image, center)

    return speed, angle

def wall_mode():
    scan = rc.lidar.get_samples()
    front_angle, front_dis = rc_utils.get_lidar_closest_point(scan, (-10, 10))
    left_dist = rc_utils.get_lidar_average_distance(scan, -40, 50)
    right_dist = rc_utils.get_lidar_average_distance(scan, 40, 50)
    _, right_front_dist = rc_utils.get_lidar_closest_point(scan, (10, 70))
    _, left_front_dist = rc_utils.get_lidar_closest_point(scan, (-10, -70))

    print(round(left_dist, 1), round(right_dist, 1))
    
    speed = rc_utils.remap_range(front_dis, 40, 150, 0, 1)
    dif = min(right_dist - left_dist, right_front_dist - left_front_dist)
    angle = rc_utils.remap_range(dif, -50, 50, -1, 1)

    if left_dist > 300:
        angle = 0.3

    speed = rc_utils.clamp(speed, -1, 1)
    angle = rc_utils.clamp(angle, -1, 1)
    return speed, angle

def cone_mode():
    global cur_mode
    global counter

    find_cones()

    angle: float
    speed: float

    if cur_mode == Mode.red_align:
        if (
            red_center is None
            or red_distance == 0
            or red_distance - prev_red_distance > 30
        ):
            if 0 < prev_red_distance < 120:
                counter = max(1, counter)
                cur_mode = Mode.red_pass
            else:
                cur_mode = Mode.no_cones

        elif (
            red_distance < 50
            and red_center[1] > rc.camera.get_width() // 4
        ):
            counter = 0.25
            cur_mode = Mode.red_reverse

        else:
            goal_point = rc_utils.remap_range(
                red_distance,
                30,
                120,
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
                30,
                120,
                0.4,
                0.8,
                True,
            )

    elif cur_mode == Mode.blue_align:
        if (
            blue_center is None
            or blue_distance == 0
            or blue_distance - prev_blue_distance > 30
        ):
            if 0 < prev_blue_distance < 120:
                counter = max(1, counter)
                cur_mode = Mode.blue_pass
            else:
                cur_mode = Mode.no_cones
        elif (
            blue_distance < 50
            and blue_center[1] < rc.camera.get_width() * 3 // 4
        ):
            counter = 0.25
            cur_mode = Mode.blue_reverse
        else:
            goal_point = rc_utils.remap_range(
                blue_distance,
                30,
                120,
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
                30,
                120,
                0.4,
                0.8,
                True,
            )

    if cur_mode == Mode.red_pass:
        angle = rc_utils.remap_range(counter, 1, 0, 0, -0.5)
        speed = 0.5
        counter -= rc.get_delta_time()

        if counter <= 0:
            cur_mode = Mode.blue_align if blue_distance > 0 else Mode.blue_find

    elif cur_mode == Mode.blue_pass:
        angle = rc_utils.remap_range(counter, 1, 0, 0, 0.5)
        speed = 0.5

        counter -= rc.get_delta_time()
        if counter <= 0:
            cur_mode = Mode.red_align if red_distance > 0 else Mode.red_find

    elif cur_mode == Mode.red_find:
        angle = 1
        speed = 0.2
        if red_distance > 0:
            cur_mode = Mode.red_align

    elif cur_mode == Mode.blue_find:
        angle = -1
        speed = 0.2
        if blue_distance > 0:
            cur_mode = Mode.blue_align

    elif cur_mode == Mode.red_reverse:
        if counter >= 0:
            counter -= rc.get_delta_time()
            speed = -1
            angle = 1
        else:
            angle = -1
            speed = -0.2
            if (
                red_distance > 60
                or red_center[1] < rc.camera.get_width() // 10
            ):
                counter = 1.2
                cur_mode = Mode.red_align

    elif cur_mode == Mode.blue_reverse:
        if counter >= 0:
            counter -= rc.get_delta_time()
            speed = -1
            angle = 1
        else:
            angle = 1
            speed = -0.2
            if (
                blue_distance > 60
                or blue_center[1] > rc.camera.get_width() * 9 / 10
            ):
                counter = 1.2
                cur_mode = Mode.blue_align

    elif cur_mode == Mode.no_cones:
        angle = 0
        speed = 0.4

        if red_distance > 0 and blue_distance == 0:
            cur_mode = Mode.red_align
        elif blue_distance > 0 and red_distance == 0:
            cur_mode = Mode.blue_align
        elif blue_distance > 0 and red_distance > 0:
            cur_mode = (
                Mode.red_align if red_distance < blue_distance else Mode.blue_align
            )
    
    return speed, angle


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
        print("No image found")
        return

    contours = rc_utils.find_contours(color_image, RED[0], RED[1])
    contour = rc_utils.get_largest_contour(contours, 100)

    if contour is not None:
        red_center = rc_utils.get_contour_center(contour)
        red_distance = rc_utils.get_pixel_average_distance(depth_image, red_center)

        if red_distance <= 250:
            rc_utils.draw_contour(color_image, contour, rc_utils.ColorBGR.green.value)
            rc_utils.draw_circle(color_image, red_center, rc_utils.ColorBGR.green.value)
        else:
            red_center = None
            red_distance = 0
    else:
        red_center = None
        red_distance = 0

    contours = rc_utils.find_contours(color_image, BLUE[0], BLUE[1])
    contour = rc_utils.get_largest_contour(contours, 100)

    if contour is not None:
        blue_center = rc_utils.get_contour_center(contour)
        blue_distance = rc_utils.get_pixel_average_distance(depth_image, blue_center)

        if blue_distance <= 250:
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


def color_mode():
    global cur_color
    global find_angle

    image = rc.camera.get_color_image()
    cropped_image = rc_utils.crop(image, (300, 0), (rc.camera.get_height(), rc.camera.get_width()))
    contours = rc_utils.find_contours(cropped_image, cur_color[0], cur_color[1])
    contour = rc_utils.get_largest_contour(contours, 40)

    if contour is not None:
        center = rc_utils.get_contour_center(contour)
        rc_utils.draw_contour(cropped_image, contour)
        rc_utils.draw_circle(cropped_image, center)

        if center[1] < rc.camera.get_width() // 2:
            find_angle = -1
        else:
            find_angle = 1

        rc.display.show_color_image(cropped_image)
        return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)

    return find_angle

def orientation_mode(direction):
    global speed
    global angle
    global counter
    global cur_stage
    global timer
    global x

    scan = rc.lidar.get_samples()
    counter += rc.get_delta_time()

    if counter < 2:
        angle = direction * 0.45
    elif counter < 3.2:
        angle = direction * -0.73
    elif rc_utils.get_lidar_average_distance(scan, -50 * direction, 6) < 100 and x:
        if direction == 1:
            left_angle, left_dis = rc_utils.get_lidar_closest_point(scan, (240, 300))
            print(left_angle, left_dis)
            if abs((left_angle - 270) / 30) > abs((-left_dis + 30) / 30):
                angle = (left_angle - 270) / 30
            else:
                angle = (-left_dis + 30) / 30
            rc.display.show_lidar(scan, highlighted_samples =[(left_angle, left_dis), ] )
        if direction == -1:
            right_angle, right_dis = rc_utils.get_lidar_closest_point(scan, (60, 120))
            print(right_angle, right_dis)
            if abs((right_angle - 90) / 30) > abs((right_dis - 30) / 30):
                angle = (right_angle - 90) / 30
            else:
                angle = (right_dis - 30) / 30
            rc.display.show_lidar(scan, highlighted_samples =[(right_angle, right_dis)] )

    else:
        timer += rc.get_delta_time()
        x = False
        if timer < 1:
            angle = direction * -1
        elif timer < 2:
            angle = direction * 1
        else:
            front_angle, front_dis = rc_utils.get_lidar_closest_point(scan, (-10, 10))
            left_dist = rc_utils.get_lidar_average_distance(scan, -40, 50)
            right_dist = rc_utils.get_lidar_average_distance(scan, 40, 50)
            _, right_front_dist = rc_utils.get_lidar_closest_point(scan, (10, 70))
            _, left_front_dist = rc_utils.get_lidar_closest_point(scan, (-10, -70))

            print(round(left_dist, 1), round(right_dist, 1))
            
            speed = rc_utils.remap_range(front_dis, 40, 150, 0, 1)
            dif = min(right_dist - left_dist, right_front_dist - left_front_dist)
            angle = rc_utils.remap_range(dif, -100, 100, -1, 1)

            speed = rc_utils.clamp(speed, -1, 1)
            angle = rc_utils.clamp(angle, -1, 1)
            return speed, angle

    speed = 1
    angle = rc_utils.clamp(angle, -1, 1)

    print("orientation_mode")
    return speed, angle

def change_stage(new_marker, color_image):
    global cur_stage
    global cur_marker
    global cur_direction
    global cur_color
    global colors
    global primary_color
    global secondary_color
    global counter

    cur_marker = new_marker

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

    if cur_marker.get_id() == 1:
        cur_stage = Stage.lane
        cur_direction = 1
        print(cur_marker.get_color())
        if cur_marker.get_color() == "purple":
            primary_color = ORANGE
            secondary_color = PURPLE
        if cur_marker.get_color() == "orange":
            primary_color = PURPLE
            secondary_color = ORANGE
        print(f"Primary:{primary_color[2]}  secondary:{secondary_color[2]}")
    elif cur_marker.get_id() == 199:
        cur_stage = Stage.orientation
        counter = 0
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
    


if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
