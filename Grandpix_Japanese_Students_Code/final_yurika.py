"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 5B - LIDAR Wall Following
"""

########################################################################################
# Imports
########################################################################################

#from lab2b import ORANGE
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
    red_align = 0  #赤に近づく
    blue_align = 1  # 青に近づく
    red_pass = 2  #赤を通り抜ける
    blue_pass = 3  # 青を通り抜ける
    red_find = 4  # 赤を探す
    blue_find = 5  # 青を探す
    red_reverse = 6  #赤に近すぎたとき離れる
    blue_reverse = 7  # 青から離れる
    no_cones = 8  # コーンがないとき前に行く


# >> Constants
# 速さの最大値
MAX_SPEED = 1.0

#ブレーキをかける距離
BRAKE_DISTANCE = 150

#壁にこれ以上近づいたら離れる
SUPER_PANIC_DISTANCE = 21
PANIC_DISTANCE = 20

#この距離よりも遠いとパニックモードから抜ける
END_PANIC_DISTANCE = 25

WIDE_DISTANCE = 130

#パニックモードのスピード
PANIC_SPEED = 0.3
SUPER_PANIC_SPEED = 1.0

#横の距離を計測するときの角度
MIN_SIDE_ANGLE = 10
MAX_SIDE_ANGLE = 75

# 左の壁と右の壁の角度を推定するために使用される2つの距離
SIDE_FRONT_ANGLE = 70
SIDE_BACK_ANGLE = 110

# 壁の前後の距離の測定値が、この距離以上異なる場合、廊下が曲がっていると仮定する
TURN_THRESHOLD = 30

# >> Variables
cur_mode = Mode.align
pre_value = 0
panic_count = 0

#slalom
# スピード
MAX_ALIGN_SPEED = 0.8
MIN_ALIGN_SPEED = 0.4
PASS_SPEED = 0.5
FIND_SPEED = 0.2
REVERSE_SPEED = -0.2
NO_CONES_SPEED = 0.4

# 時間
REVERSE_BRAKE_TIME = 0.25
SHORT_PASS_TIME = 1.0
LONG_PASS_TIME = 1.2

# コーンを見つけるとき使う
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
#それぞれの色のHSV（（最小）、（最大））
RED = ((170, 50, 50), (10, 255, 255), "red") 
BLUE = ((100, 150, 150), (120, 255, 255), "blue")
GREEN = ((40, 50, 50), (80, 255, 255), "green") 
ORANGE = ((10, 100, 100), (25, 255, 255), "orange")
PURPLE = ((125, 100, 100), (150, 255, 255), "purple")

colors = [RED, BLUE, GREEN, ORANGE, PURPLE]
colors_lane = [ORANGE, PURPLE]

# 壁に沿って進む時使う
# 車の右の角度
SIDE_ANGLE = 90

# 壁の角度を推定するために、車の側面をそれぞれの方向で見る
SPREAD_ANGLE = 30

# 各距離測定のために平均化する測定値の角度
WINDOW_ANGLE = 60

#壁から離れようとする距離
WALL_DISTANCE = 36
IS_WALL_DISTANCE = 70
S_DISTANCE = 50
TURN_DISTANCE = 70
COLLISION_DISTANCE = 30

# The maximum difference between the desired value and the current value, which is
# scaled to a full left (-1) or a full right (1) turn
MAX_DIF = 10

# 壁の角度に比べて壁からの距離を考慮している？
DISTANCE_COEFFICIENT = 0.5

# ラインフォローのとき使う
# 有用な値として認識する最小値
MIN_CONTOUR_AREA = 30
MIN_CONTOUR_AREA_LANE = 500

# 色のついた線を探すときの画像の領域
CROP_WINDOW = ((300, 0), (rc.camera.get_height(), rc.camera.get_width()))
CROP_WINDOW_lane = ((rc.camera.get_height() * 2 // 3, 0),(rc.camera.get_height(), rc.camera.get_width()))
CROP_WINDOW_lane_right = ((280, int(rc.camera.get_width()/2)), (rc.camera.get_height() - 100, rc.camera.get_width()))
CROP_WINDOW_lane_left = ((280, 0), (rc.camera.get_height() - 100, int(rc.camera.get_width()/2)))
CROP_FLOOR = (
    (rc.camera.get_height() * 2 // 3, 0),
    (rc.camera.get_height(), rc.camera.get_width()),
)

# スピード
EXPLORE_SPEED = 0.4
WALL_FOLLOW_SPEED = 0.5
LINE_FOLLOW_SPEED = 0.7
LANE_FOLLOW_SPEED = 0.4
MAX_SPEED = 0.35
# レーンが一番目の色の場合の使用速度
FAST_SPEED = 1.0
# レーンが二番目の色の場合に使用する速度
SLOW_SPEED = 0.5

# 1車線しか見えない場合の曲がり方の量
ONE_LANE_TURN_ANGLE = 0.6

# >> Variables
cur_stage = Stage.none

# 検出された最新のマーカー
cur_marker = rc_utils.ARMarker(-1, np.zeros((4, 2), dtype=np.int32))

# 左の壁に沿っている場合は-1、右の壁に沿っている場合は1
cur_direction = 0

# 現在フォローしている色
cur_color = None

pre_center = None

# 一番優先する色 
primary_color = ORANGE
# 二番目に優先する色
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

    # 新しいマーカーを見たときステージを変える
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
        print("ARmarker = ", area)
        
        if area > 1500 and markers[0].get_id() != cur_marker.get_id():
            change_stage(markers[0], color_image)
            print(markers[0])
        
    speed = 0
    angle = 0
    if cur_stage == Stage.none:
        #一つ目を見つけるまで前に進む
        angle, speed = wall_follow_lab4b()
    elif cur_stage == Stage.marker_id or cur_stage == Stage.orientation or cur_stage == Stage.wall:
        #最初の2つのマーカーの後は、マーカーが示す壁に沿って進む
        speed, angle = wall_follow2(cur_direction)
    elif cur_stage == Stage.color:
        speed = LINE_FOLLOW_SPEED
        angle = line_follow(color_image)

        # カラーラインが見えなかったら壁に沿って進む
        if angle is None:
            _, angle = wall_follow2(cur_direction)
   
    elif cur_stage == Stage.lane:
        # マーカーが見えたら徐々に前に進む
        speed, angle = lane_follow_matthew()

        if angle is None:
            _, angle = wall_follow2(cur_direction)        

    rc.drive.set_speed_angle(speed, angle)
    print(cur_stage)



def lane_follow_matthew():
    speed = SLOW_SPEED
    angle = 0
    image = rc.camera.get_color_image()
    if image is None:
        print("No image")
        rc.drive.stop()
        return

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
    #最大輪郭を見つける

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



def lane_follow(color_image):
    
    """
    Determines the angle of the wheels necessary to follow the colored line
    on the floor, with color priority specified by the colors parameter.

    Uses a similar strategy to Lab 2A.
    """
    global colors_lane
    global pre_center

    # クロップする
    cropped_image = rc_utils.crop(color_image, CROP_WINDOW_lane[0], CROP_WINDOW_lane[1])

    # 一番優先する色を見つける
    for color in colors_lane:
        # 一番大きい輪郭を見つける
        contours = rc_utils.find_contours(cropped_image, color[0], color[1])
        contour, second_contour = largest_contour(contours, MIN_CONTOUR_AREA)

        # 輪郭がある場合は、輪郭の中心に向かう
        if contour is not None and second_contour is not None:
            center = rc_utils.get_contour_center(contour)
            print(center)
            second_center = rc_utils.get_contour_center(second_contour)
            print(second_center)
            center = (int((center[0] + second_center[0]) / 2), int((center[1] + second_center[1]) / 2))
            #輪郭と円を書く
            rc_utils.draw_contour(cropped_image, contour)
            rc_utils.draw_contour(cropped_image, second_contour, (255, 0, 0))
            rc_utils.draw_circle(cropped_image, center)
            rc.display.show_color_image(cropped_image)
            return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)

    # カラーラインが見つからない場合は、Noneを返し、壁に沿って進む
    return None 

def lane_follow_crop(color_image):
    """
   Determines the angle of the wheels necessary to follow the colored line
    on the floor, with color priority specified by the colors parameter.

    Uses a similar strategy to Lab 2A. 
    """
    global colors_lane
    global pre_center

    #クロップする
    cropped_image_left = rc_utils.crop(color_image, CROP_WINDOW_lane_left[0], CROP_WINDOW_lane_left[1])
    cropped_image_right = rc_utils.crop(color_image, CROP_WINDOW_lane_right[0], CROP_WINDOW_lane_right[1])

    # 一番優先する色を探す
    for color in colors_lane:
        #一番大きい輪郭を探す（右側）
        contours = rc_utils.find_contours(cropped_image_right, color[0], color[1])
        contour_right = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        # 見つけたらそこに向かって進む
        if contour_right is not None:
            center_right = rc_utils.get_contour_center(contour_right)
            rc_utils.draw_contour(cropped_image_right, contour_right)
            rc_utils.draw_circle(cropped_image_right, center_right)
            rc.display.show_color_image(cropped_image_right)
    
    for color in colors_lane:
        # 一番大きい輪郭を探す
        contours = rc_utils.find_contours(cropped_image_left, color[0], color[1])
        contour_left = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        # 見つけたらそこに向かって進む（左側）
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

        # カラーラインが見つからない場合は、Noneを返し、壁に沿って進む
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

""" def lane_follow_Ave(color_image):
   
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
 """
def find_cones():
    """
    赤と青のコーンを探して値を新しくする
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

    #赤いコーンを探す
    contours = rc_utils.find_contours(color_image, RED[0], RED[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

    if contour is not None:
        red_center = rc_utils.get_contour_center(contour)
        red_distance = rc_utils.get_pixel_average_distance(depth_image, red_center)

        # コーンの距離がMAX_DISTANCE以下の場合にのみ、countを使用する（赤）
        if red_distance <= MAX_DISTANCE:
            rc_utils.draw_contour(color_image, contour, rc_utils.ColorBGR.green.value)
            rc_utils.draw_circle(color_image, red_center, rc_utils.ColorBGR.green.value)
        else:
            red_center = None
            red_distance = 0
    else:
        red_center = None
        red_distance = 0

    #青いコーンを探す
    contours = rc_utils.find_contours(color_image, BLUE[0], BLUE[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

    if contour is not None:
        blue_center = rc_utils.get_contour_center(contour)
        blue_distance = rc_utils.get_pixel_average_distance(depth_image, blue_center)

        # コーンの距離がMAX_DISTANCE以下の場合にのみ、countを使用する（青）
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

    # 前方、側方、後方への最小距離を求める
    front_angle, front_dist = rc_utils.get_lidar_closest_point(
        scan, (-MIN_SIDE_ANGLE, MIN_SIDE_ANGLE)
    )
    left_angle, left_dist = rc_utils.get_lidar_closest_point(
        scan, (-MAX_SIDE_ANGLE, -MIN_SIDE_ANGLE)
    )
    right_angle, right_dist = rc_utils.get_lidar_closest_point(
        scan, (MIN_SIDE_ANGLE, MAX_SIDE_ANGLE)
    )


    #左の壁の角度を求める
    left_front_dist = rc_utils.get_lidar_average_distance(
        scan, -SIDE_FRONT_ANGLE, WINDOW_ANGLE
    )
    left_back_dist = rc_utils.get_lidar_average_distance(
        scan, -SIDE_BACK_ANGLE, WINDOW_ANGLE
    )
    left_dif = left_front_dist - left_back_dist

    # 右も同様に
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
    elif (left_dist < PANIC_DISTANCE or right_dist < PANIC_DISTANCE) and (cur_mode != Mode.super_left_panic or cur_mode != Mode.super_right_panic):
        cur_mode = Mode.left_panic if left_dist < right_dist else Mode.right_panic
        panic_count = 0
    elif left_dist > WIDE_DISTANCE and right_dist > WIDE_DISTANCE:
        cur_mode = Mode.wide
        panic_count = 0

    

    #壁がなかったら車を止める
    if left_front_dist == 0.0 and right_front_dist == 0.0:
        speed = 0
        angle = 0

    elif cur_mode == Mode.front_panic:
        angle = 0
        speed = -SUPER_PANIC_SPEED

        if front_dist > END_PANIC_DISTANCE:
            cur_mode = Mode.align

    # 左のパニックモード
    elif cur_mode == Mode.left_panic:
        angle = 1
        speed = PANIC_SPEED

        if left_dist > END_PANIC_DISTANCE:
            cur_mode = Mode.align

    # 右のパニックモード
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
    
    # ALIGNモード　左右の壁の間にまっすぐ進む
    else:
        value = (right_dif - left_dif) + (right_dist - left_dist)
        panic_count = 0
        speed = rc_utils.remap_range(front_dist, 0, BRAKE_DISTANCE, 0, MAX_SPEED, True)

        # left_difが非常に大きい場合は、左に曲がる
        if left_dif > TURN_THRESHOLD:
            angle = -1
            speed *= 0.8
            print("Turn left")

        # 右も同様
        elif right_dif > TURN_THRESHOLD:
            angle = 1
            speed *= 0.8
            print("Turn right")
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

    # ３つの点を図る
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
    
    #左側
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

    #右も同様
    right_front_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, SIDE_FRONT_ANGLE, WINDOW_ANGLE
    )
    right_back_dist = rc_utils.get_lidar_average_distance(
        lidar_scan, SIDE_BACK_ANGLE, WINDOW_ANGLE
    )
    right_dif = right_front_dist - right_back_dist

    # 壁との位置関係からゴールの角度を決める
    dif_component = rc_utils.remap_range(
        side_front_dist - side_back_dist, -MAX_DIF, MAX_DIF, -1, 1, True
    )

    # D 壁との位置関係からゴールの角度を決める
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
    # left_difが非常に大きい場合は、廊下が左に曲がっている可能性が高い
    elif 0 < front_dist < TURN_DISTANCE:
        if (left_dif > TURN_THRESHOLD or left_dist > IS_WALL_DISTANCE) and right_front_dist < IS_WALL_DISTANCE:
            angle = -1
            speed = PANIC_SPEED
            print("Turn left")
    
        # 右も同じ
        elif (right_dif > TURN_THRESHOLD or right_dist > IS_WALL_DISTANCE) and left_front_dist < IS_WALL_DISTANCE:
            angle = 1
            speed = PANIC_SPEED
            print("Turn right")
   
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
        
     
    

    return speed, direction * rc_utils.clamp(angle, -1, 1)



def line_follow(color_image):
    """
    Determines the angle of the wheels necessary to follow the colored line
    on the floor, with color priority specified by the colors parameter.

    Uses a similar strategy to Lab 2A.
    """
    global cur_color

    # クロップする
    cropped_image = rc_utils.crop(color_image, CROP_WINDOW[0], CROP_WINDOW[1])


    contours = rc_utils.find_contours(cropped_image, cur_color[0], cur_color[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

    if contour is not None:
        center = rc_utils.get_contour_center(contour)
        rc_utils.draw_contour(cropped_image, contour)
        rc_utils.draw_circle(cropped_image, center)

        
        rc.display.show_color_image(cropped_image)
        return rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1)

    # ラインが見えなかったらnoneを返す
    return None


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
 