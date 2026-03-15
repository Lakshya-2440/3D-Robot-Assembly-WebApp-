
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import time
import os
import urllib.request
import ssl
from collections import deque

CAM_W, CAM_H = 640, 480
SIM_W, SIM_H = 1920, 1080
TOTAL_W, TOTAL_H = SIM_W, SIM_H

ROBOT_SPEED = 0.15
WORLD_RADIUS = 8.0
GESTURE_STABLE_FRAMES = 3

# ------------------ MEDIAPIPE SETUP ------------------

ssl._create_default_https_context = ssl._create_unverified_context

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded!")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    min_hand_presence_confidence=0.4
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

robot_x = 0.0
robot_z = 0.0
robot_angle = 0.0       # current facing angle (degrees, CCW, 0=south)
target_angle = 0.0      # desired facing angle
TURN_SPEED = 10.0       # degrees per frame
robot_state = "IDLE"
current_gesture = "NONE"
stable_gesture = "NONE"
gesture_history = deque(maxlen=GESTURE_STABLE_FRAMES)

# --- Action hand (second hand) state ---
action_gesture = "NONE"
stable_action_gesture = "NONE"
action_gesture_history = deque(maxlen=GESTURE_STABLE_FRAMES)
robot_frozen = False
prev_action_fist = False

# ------------------ PICK/DROP STATE ------------------
held_object = None
held_object_index = None
PICKUP_RANGE = 1.5

# ------------------ TOOLBOX STATE ------------------

toolbox_categories = ["Chassis", "Wheels (WASD)", "Lidar", "Camera (View)", "Arm (SPACE)"]
toolbox_parts = {
    "Chassis":       ["None", "Standard", "Tank", "Humanoid", "Spider"],
    "Wheels (WASD)": ["None", "Standard", "Treads", "Legs"],
    "Lidar":         ["Off", "On"],
    "Camera (View)": ["None", "Standard", "Wide-Angle", "Thermal"],
    "Arm (SPACE)":   ["None", "Retracted", "Extended"],
}

selected_parts = {cat: 0 for cat in toolbox_categories}

toolbox_colors = [
    (100, 140, 50),   # Chassis (Teal BGR)
    (50, 110, 220),   # Wheels (Orange BGR)
    (50, 50, 200),    # Lidar (Red BGR)
    (230, 110, 60),   # Camera (Blue BGR)
    (220, 50, 150),   # Arm (Purple BGR)
]

last_click_time = 0.0
CLICK_COOLDOWN = 0.5
last_hovered_cat = None  # tracks which box the cursor is inside, for toggle-on-enter

# ------------------ WORLD OBJECTS ------------------
import random

random.seed(42)

# Static objects scattered in the playground (world coords, color, shape, size)
static_objects = []
shape_types = ["circle", "square", "triangle"]
obj_colors = [
    (0, 200, 0), (0, 120, 255), (255, 100, 100), (200, 0, 200),
    (0, 255, 255), (50, 200, 255), (255, 200, 0), (100, 255, 100),
    (180, 50, 255), (255, 150, 50),
]
for _ in range(50):
    wx = random.uniform(-WORLD_RADIUS * 0.95, WORLD_RADIUS * 0.95)
    wz = random.uniform(-WORLD_RADIUS * 0.95, WORLD_RADIUS * 0.95)
    static_objects.append({
        "x": wx, "z": wz,
        "color": random.choice(obj_colors),
        "shape": random.choice(shape_types),
        "size": random.randint(6, 14),
    })

# Wandering stick figure humans
wandering_humans = []
for _ in range(12):
    wx = random.uniform(-WORLD_RADIUS * 0.9, WORLD_RADIUS * 0.9)
    wz = random.uniform(-WORLD_RADIUS * 0.9, WORLD_RADIUS * 0.9)
    wandering_humans.append({
        "x": wx, "z": wz,
        "vx": random.uniform(-0.02, 0.02),
        "vz": random.uniform(-0.02, 0.02),
        "color": random.choice([(200, 200, 200), (180, 160, 255), (150, 220, 255), (200, 255, 180)]),
    })

# ------------------ UTIL FUNCTIONS ------------------

def fingers_up(landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    out = []
    for tip, pip in zip(tips, pips):
        out.append(landmarks[tip].y < landmarks[pip].y)
    return out

def thumb_up(landmarks, handedness="Right"):
    if handedness == "Right":
        return landmarks[4].x < landmarks[3].x
    else:
        return landmarks[4].x > landmarks[3].x

def get_pinch_distance(landmarks_list):
    if not landmarks_list:
        return 1.0
    lm = landmarks_list[0]
    dx = lm[8].x - lm[4].x
    dy = lm[8].y - lm[4].y
    return math.hypot(dx, dy)

def classify_gesture(landmarks, handedness="Right"):
    index, middle, ring, pinky = fingers_up(landmarks)
    thumb = thumb_up(landmarks, handedness)
    up_count = sum([index, middle, ring, pinky])

    if up_count == 0:
        return "LAND"
    if index and middle and ring and pinky and thumb:
        return "HOVER"
    if index and middle and ring and pinky and not thumb:
        return "MOVE_LEFT"
    if index and not middle and not ring and not pinky:
        return "MOVE_FORWARD"
    if index and middle and not ring and not pinky:
        return "MOVE_BACKWARD"
    if index and middle and ring and not pinky:
        return "MOVE_RIGHT"
    return "NONE"

def update_stable_gesture(new_gesture):
    global stable_gesture
    gesture_history.append(new_gesture)
    if len(gesture_history) == GESTURE_STABLE_FRAMES:
        unique = set(gesture_history)
        if len(unique) == 1:
            stable_gesture = list(unique)[0]

def update_stable_action_gesture(new_gesture):
    global stable_action_gesture
    action_gesture_history.append(new_gesture)
    if len(action_gesture_history) == GESTURE_STABLE_FRAMES:
        unique = set(action_gesture_history)
        if len(unique) == 1:
            stable_action_gesture = list(unique)[0]

def update_robot():
    global robot_x, robot_z, robot_state, robot_angle, target_angle

    # Freeze overrides everything — action hand open palm
    if robot_frozen:
        robot_state = "FROZEN"
        return

    # No wheels/treads/legs = no movement
    wheels = toolbox_parts["Wheels (WASD)"][selected_parts["Wheels (WASD)"]]
    if wheels == "None":
        robot_state = "IDLE"
        return

    if stable_gesture == "LAND":
        # Drive back to center with wheels, not fly/slide
        dist_to_center = math.sqrt(robot_x**2 + robot_z**2)
        if dist_to_center < 0.3:
            robot_x, robot_z = 0.0, 0.0
            robot_state = "IDLE"
            return

        # Calculate angle to face the center
        target_angle = math.degrees(math.atan2(-robot_x, -robot_z)) % 360

        # Smooth rotation toward center
        diff = (target_angle - robot_angle + 180) % 360 - 180
        if abs(diff) <= TURN_SPEED:
            robot_angle = target_angle
        else:
            robot_angle += TURN_SPEED if diff > 0 else -TURN_SPEED
        robot_angle = robot_angle % 360

        # Drive toward center once facing close enough
        if abs(diff) < 30:
            robot_state = "WALKING"
            robot_x += (-robot_x / dist_to_center) * ROBOT_SPEED
            robot_z += (-robot_z / dist_to_center) * ROBOT_SPEED
        else:
            robot_state = "TURNING"
        return

    if stable_gesture == "HOVER":
        robot_state = "STANDING"
        return

    # Determine target angle based on gesture
    # 0=south(down), 90=east(right), 180=north(up), 270=west(left)
    moving = False
    if stable_gesture == "MOVE_FORWARD":
        target_angle = 180
        moving = True
    elif stable_gesture == "MOVE_BACKWARD":
        target_angle = 0
        moving = True
    elif stable_gesture == "MOVE_LEFT":
        target_angle = 270
        moving = True
    elif stable_gesture == "MOVE_RIGHT":
        target_angle = 90
        moving = True

    # Smooth rotation toward target (shortest path)
    diff = (target_angle - robot_angle + 180) % 360 - 180
    if abs(diff) <= TURN_SPEED:
        robot_angle = target_angle
    else:
        robot_angle += TURN_SPEED if diff > 0 else -TURN_SPEED
    robot_angle = robot_angle % 360

    # Only move when facing close enough to the target direction
    if moving and abs(diff) < 25:
        robot_state = "WALKING"
        if stable_gesture == "MOVE_FORWARD":
            robot_z -= ROBOT_SPEED
        elif stable_gesture == "MOVE_BACKWARD":
            robot_z += ROBOT_SPEED
        elif stable_gesture == "MOVE_LEFT":
            robot_x -= ROBOT_SPEED
        elif stable_gesture == "MOVE_RIGHT":
            robot_x += ROBOT_SPEED
    elif moving:
        robot_state = "TURNING"
    else:
        robot_state = "STANDING"

    r = math.sqrt(robot_x**2 + robot_z**2)
    if r > WORLD_RADIUS:
        scale = WORLD_RADIUS / r
        robot_x *= scale
        robot_z *= scale


def try_pickup():
    """Attempt to pick up the nearest static_object within PICKUP_RANGE."""
    global held_object, held_object_index
    if held_object is not None:
        return
    arm = toolbox_parts["Arm (SPACE)"][selected_parts["Arm (SPACE)"]]
    if arm != "Extended":
        return
    best_dist = PICKUP_RANGE
    best_idx = None
    for i, obj in enumerate(static_objects):
        dx = obj["x"] - robot_x
        dz = obj["z"] - robot_z
        dist = math.sqrt(dx * dx + dz * dz)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    if best_idx is not None:
        held_object = static_objects[best_idx]
        held_object_index = best_idx


def try_drop():
    """Release the held object at the robot's current position."""
    global held_object, held_object_index
    if held_object is None:
        return
    held_object["x"] = robot_x
    held_object["z"] = robot_z
    held_object = None
    held_object_index = None


# ------------------ DRAWING ------------------

def draw_robot_model(sim, cx, cy, preview_scale=1.0):
    """Draw a top-down robot platform, rotated to face robot_angle."""
    t = time.time()

    chassis = toolbox_parts["Chassis"][selected_parts["Chassis"]]
    wheels = toolbox_parts["Wheels (WASD)"][selected_parts["Wheels (WASD)"]]
    lidar = toolbox_parts["Lidar"][selected_parts["Lidar"]]
    camera = toolbox_parts["Camera (View)"][selected_parts["Camera (View)"]]
    arm = toolbox_parts["Arm (SPACE)"][selected_parts["Arm (SPACE)"]]

    if chassis == "None" and wheels == "None" and lidar == "Off" and camera == "None" and arm == "None":
        return

    s = preview_scale

    # Draw on a temp canvas centered at (tc, tc), then rotate and overlay
    temp_sz = 250
    tc = temp_sz // 2
    temp = np.zeros((temp_sz, temp_sz, 3), dtype=np.uint8)
    glow = np.zeros_like(temp)

    # Colors matching the reference image
    green_body = (80, 180, 80)
    green_border = (60, 220, 60)
    black_wheel = (30, 30, 30)
    wheel_border = (80, 80, 80)
    red_lidar = (60, 60, 240)
    blue_cam = (240, 160, 40)
    purple_arm = (200, 60, 200)
    purple_dark = (150, 40, 150)

    # --- Chassis ---
    if chassis != "None":
        if chassis == "Standard":
            bw, bh = int(60 * s), int(45 * s)
        elif chassis == "Tank":
            bw, bh = int(70 * s), int(50 * s)
        elif chassis == "Humanoid":
            bw, bh = int(50 * s), int(55 * s)
        elif chassis == "Spider":
            bw, bh = int(55 * s), int(55 * s)
        else:
            bw, bh = int(60 * s), int(45 * s)

        cv2.rectangle(temp, (tc - bw // 2, tc - bh // 2),
                      (tc + bw // 2, tc + bh // 2), green_body, -1)
        cv2.rectangle(temp, (tc - bw // 2, tc - bh // 2),
                      (tc + bw // 2, tc + bh // 2), green_border, 2)
        cv2.line(temp, (tc - bw // 2 + 6, tc), (tc + bw // 2 - 6, tc), (70, 150, 70), 1)
        if chassis == "Tank":
            cv2.line(temp, (tc, tc - bh // 2 + 4), (tc, tc + bh // 2 - 4), (70, 150, 70), 1)
        elif chassis == "Spider":
            cv2.line(temp, (tc - bw//3, tc - bh//3), (tc + bw//3, tc + bh//3), (70, 150, 70), 1)
            cv2.line(temp, (tc + bw//3, tc - bh//3), (tc - bw//3, tc + bh//3), (70, 150, 70), 1)

    # --- Wheels ---
    is_moving = robot_state in ("WALKING", "TURNING")
    if wheels != "None" and chassis != "None":
        wheel_r = int(14 * s)  # big circular wheel radius

        if wheels == "Treads":
            ww, wh = int(14 * s), int(bh * 0.95)
            for side in [-1, 1]:
                tx = tc + side * (bw // 2 + ww // 2 + int(2 * s))
                cv2.rectangle(temp, (tx - ww // 2, tc - wh // 2),
                              (tx + ww // 2, tc + wh // 2), black_wheel, -1)
                cv2.rectangle(temp, (tx - ww // 2, tc - wh // 2),
                              (tx + ww // 2, tc + wh // 2), wheel_border, 2)
                # Scrolling tread pattern
                spacing = max(1, wh // 6)
                tread_offset = int((t * 80) % spacing) if is_moving else 0
                for j in range(8):
                    ly = int(tc - wh // 2 + j * spacing + tread_offset)
                    if tc - wh // 2 <= ly <= tc + wh // 2:
                        cv2.line(temp, (tx - ww//2 + 2, ly), (tx + ww//2 - 2, ly), wheel_border, 1)
        elif wheels == "Legs":
            corners = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
            for idx, (sx_dir, sz_dir) in enumerate(corners):
                if is_moving:
                    phase = math.sin(t * 8 + idx * math.pi / 2)
                    leg_ext = int(6 * s + phase * 4 * s)
                else:
                    leg_ext = int(6 * s)
                lx = tc + sx_dir * (bw // 2 + leg_ext)
                ly = tc + sz_dir * (bh // 2 - int(4 * s))
                cv2.circle(temp, (lx, ly), int(7 * s), black_wheel, -1)
                cv2.circle(temp, (lx, ly), int(7 * s), wheel_border, 2)
                cv2.line(temp, (tc + sx_dir * (bw // 2), ly), (lx, ly), (100, 100, 100), int(2 * s))
        else:
            # Standard: 4 big circular wheels at corners
            corners = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
            for idx, (sx_dir, sz_dir) in enumerate(corners):
                wx_t = tc + sx_dir * (bw // 2 + int(4 * s))
                wy_t = tc + sz_dir * (bh // 2 - int(2 * s))
                # Tire (outer dark circle)
                cv2.circle(temp, (wx_t, wy_t), wheel_r, black_wheel, -1)
                cv2.circle(temp, (wx_t, wy_t), wheel_r, wheel_border, 2)
                # Hub (inner lighter circle)
                hub_r = int(5 * s)
                cv2.circle(temp, (wx_t, wy_t), hub_r, (60, 60, 60), -1)
                cv2.circle(temp, (wx_t, wy_t), hub_r, wheel_border, 1)
                # Spinning spokes
                spoke_r = wheel_r - 2
                if is_moving:
                    for spoke in range(3):
                        angle = t * 10 + spoke * math.pi * 2 / 3 + idx * 0.3
                        sdx = int(spoke_r * math.cos(angle))
                        sdy = int(spoke_r * math.sin(angle))
                        cv2.line(temp, (wx_t, wy_t),
                                 (wx_t + sdx, wy_t + sdy), wheel_border, 2)
                else:
                    for spoke in range(3):
                        angle = spoke * math.pi * 2 / 3
                        sdx = int(spoke_r * math.cos(angle))
                        sdy = int(spoke_r * math.sin(angle))
                        cv2.line(temp, (wx_t, wy_t),
                                 (wx_t + sdx, wy_t + sdy), wheel_border, 1)

    # --- Lidar ---
    if lidar == "On" and chassis != "None":
        lidar_px = tc
        lidar_py = tc - int(8 * s)
        cv2.circle(temp, (lidar_px, lidar_py), int(8 * s), red_lidar, -1)
        cv2.circle(temp, (lidar_px, lidar_py), int(8 * s), (80, 80, 255), 2)
        pulse_lidar = abs(math.sin(t * 8))
        scan_r = int(12 * s + pulse_lidar * 6)
        cv2.circle(temp, (lidar_px, lidar_py), scan_r, (80, 80, 255), 1)
        cv2.circle(glow, (lidar_px, lidar_py), int(10 * s), red_lidar, -1)

    # --- Camera ---
    if camera != "None" and chassis != "None":
        cam_px = tc
        cam_py = tc + int(10 * s)
        cam_r = int(5 * s) if camera == "Standard" else int(7 * s) if camera == "Wide-Angle" else int(6 * s)
        if camera == "Thermal":
            cv2.circle(temp, (cam_px, cam_py), cam_r, (0, 100, 255), -1)
            cv2.circle(temp, (cam_px, cam_py), cam_r, (0, 140, 255), 2)
        else:
            cv2.circle(temp, (cam_px, cam_py), cam_r, blue_cam, -1)
            cv2.circle(temp, (cam_px, cam_py), cam_r, (255, 200, 80), 2)
        cv2.circle(temp, (cam_px - int(1*s), cam_py - int(1*s)), int(1 * s), (255, 255, 255), -1)

    # --- Arm ---
    if arm != "None" and chassis != "None":
        arm_base_y = tc - bh // 2

        if arm == "Extended":
            arm_top_y = arm_base_y - int(30 * s)
            cv2.line(temp, (tc, arm_base_y), (tc, arm_top_y), purple_arm, int(5 * s))
            cv2.circle(temp, (tc, arm_top_y), int(4 * s), purple_dark, -1)
            cv2.circle(temp, (tc, arm_top_y), int(4 * s), purple_arm, 2)
            forearm_top_y = arm_top_y - int(20 * s)
            cv2.line(temp, (tc, arm_top_y), (tc, forearm_top_y), purple_arm, int(4 * s))
            grip_len = int(12 * s)
            if held_object is not None:
                g_spread = int(4 * s)
            else:
                g_spread = grip_len
            cv2.line(temp, (tc, forearm_top_y),
                     (tc - g_spread, forearm_top_y - grip_len), purple_arm, int(3 * s))
            cv2.line(temp, (tc, forearm_top_y),
                     (tc + g_spread, forearm_top_y - grip_len), purple_arm, int(3 * s))
            cv2.circle(temp, (tc - g_spread, forearm_top_y - grip_len), int(3 * s), purple_dark, -1)
            cv2.circle(temp, (tc + g_spread, forearm_top_y - grip_len), int(3 * s), purple_dark, -1)
            cv2.circle(glow, (tc, forearm_top_y), 10, purple_arm, -1)

            # Draw held object at gripper tip
            if held_object is not None:
                obj = held_object
                grip_tip_y = forearm_top_y - int(grip_len * 0.7)
                obj_sz = int(obj["size"] * s * 0.8)
                col = obj["color"]
                if obj["shape"] == "circle":
                    cv2.circle(temp, (tc, grip_tip_y), obj_sz, col, -1)
                    cv2.circle(temp, (tc, grip_tip_y), obj_sz, (255, 255, 255), 1)
                elif obj["shape"] == "square":
                    cv2.rectangle(temp, (tc - obj_sz, grip_tip_y - obj_sz),
                                  (tc + obj_sz, grip_tip_y + obj_sz), col, -1)
                    cv2.rectangle(temp, (tc - obj_sz, grip_tip_y - obj_sz),
                                  (tc + obj_sz, grip_tip_y + obj_sz), (255, 255, 255), 1)
                elif obj["shape"] == "triangle":
                    pts = np.array([
                        [tc, grip_tip_y - obj_sz],
                        [tc - obj_sz, grip_tip_y + obj_sz],
                        [tc + obj_sz, grip_tip_y + obj_sz],
                    ], np.int32)
                    cv2.fillPoly(temp, [pts], col)
                    cv2.polylines(temp, [pts], True, (255, 255, 255), 1)

        elif arm == "Retracted":
            arm_top_y = arm_base_y - int(15 * s)
            cv2.line(temp, (tc, arm_base_y), (tc, arm_top_y), purple_arm, int(5 * s))
            cv2.circle(temp, (tc, arm_top_y), int(4 * s), purple_dark, -1)
            cv2.circle(temp, (tc, arm_top_y), int(4 * s), purple_arm, 2)

    # Apply glow on temp canvas
    glow = cv2.GaussianBlur(glow, (21, 21), 0)
    temp = cv2.addWeighted(temp, 1.0, glow, 0.3, 0)

    # Rotate the temp canvas by robot_angle (CCW positive, 0=south)
    # Add 180 so that arm/camera (drawn at top of canvas) point in the facing direction
    M = cv2.getRotationMatrix2D((tc, tc), robot_angle + 180, 1.0)
    rotated = cv2.warpAffine(temp, M, (temp_sz, temp_sz))

    # Overlay non-black pixels onto sim
    sim_h, sim_w = sim.shape[:2]
    x1, y1 = cx - tc, cy - tc
    sx1, sy1 = max(0, -x1), max(0, -y1)
    sx2 = min(temp_sz, sim_w - x1)
    sy2 = min(temp_sz, sim_h - y1)
    dx1, dy1 = max(0, x1), max(0, y1)
    dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)

    if sx2 > sx1 and sy2 > sy1:
        roi = rotated[sy1:sy2, sx1:sx2]
        mask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) > 5
        sim[dy1:dy2, dx1:dx2][mask] = roi[mask]

def update_toolbox_logic(cx, cy, pinching):
    """Handle hover-to-toggle and pinch-to-cycle logic for toolbox (no drawing)."""
    global selected_parts, last_click_time, last_hovered_cat

    box_sz = 100
    padding = 15
    total_w = len(toolbox_categories) * box_sz + (len(toolbox_categories) - 1) * padding
    start_x = (SIM_W - total_w) // 2
    start_y = 65

    cursor_in_ui = False

    if start_x <= cx <= start_x + total_w and start_y <= cy <= start_y + box_sz:
        cursor_in_ui = True

    chassis_equipped = toolbox_parts["Chassis"][selected_parts["Chassis"]] != "None"

    # If chassis is removed, reset all other parts
    if not chassis_equipped:
        for cat in toolbox_categories:
            if cat != "Chassis":
                selected_parts[cat] = 0

    now = time.time()
    current_hover = None
    for i, cat in enumerate(toolbox_categories):
        bx = start_x + i * (box_sz + padding)
        by = start_y

        is_hover = (bx <= cx <= bx + box_sz) and (by <= cy <= by + box_sz)

        if is_hover:
            current_hover = cat

            # Skip non-chassis parts if chassis not equipped
            if cat != "Chassis" and not chassis_equipped:
                continue

            # Toggle on enter: only when cursor first enters this box
            if last_hovered_cat != cat:
                if selected_parts[cat] == 0 and len(toolbox_parts[cat]) > 1:
                    # Part is off → add it (set to first real option)
                    selected_parts[cat] = 1
                else:
                    # Part is on → remove it (set back to None/Off)
                    selected_parts[cat] = 0

            # Pinch to cycle through variants
            if pinching and (now - last_click_time > CLICK_COOLDOWN):
                selected_parts[cat] = (selected_parts[cat] + 1) % len(toolbox_parts[cat])
                last_click_time = now

    last_hovered_cat = current_hover
    return cursor_in_ui


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if thickness == -1:
        # Filled
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        # Border only
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_toolbox(sim, cursor_x, cursor_y):
    """Draw the toolbox UI overlay on top of the simulation."""
    box_sz = 100
    padding = 15
    total_w = len(toolbox_categories) * box_sz + (len(toolbox_categories) - 1) * padding
    start_x = (SIM_W - total_w) // 2
    start_y = 65
    corner_r = 12

    title_sz = cv2.getTextSize("Toolbox", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(sim, "Toolbox", ((SIM_W - title_sz[0]) // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    chassis_equipped = toolbox_parts["Chassis"][selected_parts["Chassis"]] != "None"

    for i, cat in enumerate(toolbox_categories):
        bx = start_x + i * (box_sz + padding)
        by = start_y

        is_hover = (bx <= cursor_x <= bx + box_sz) and (by <= cursor_y <= by + box_sz)
        locked = cat != "Chassis" and not chassis_equipped

        if locked:
            color = (40, 40, 40)
        else:
            color = toolbox_colors[i]

        draw_rounded_rect(sim, (bx, by), (bx + box_sz, by + box_sz), color, -1, corner_r)
        if is_hover and not locked:
            draw_rounded_rect(sim, (bx - 2, by - 2), (bx + box_sz + 2, by + box_sz + 2), (255, 255, 255), 2, corner_r)

        active_part = toolbox_parts[cat][selected_parts[cat]]
        short_name = cat.split(" (")[0]
        if locked:
            label = "--"
            text_color = (80, 80, 80)
        else:
            is_on = active_part not in ("None", "Off")
            label = "On" if is_on else "Off"
            text_color = (255, 255, 255)

        # Name centered
        name_sz = cv2.getTextSize(short_name, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.putText(sim, short_name, (bx + (box_sz - name_sz[0]) // 2, by + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
        # On/Off centered below
        label_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.putText(sim, label, (bx + (box_sz - label_sz[0]) // 2, by + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

def draw_landmarks_on_image(image, hand_landmarks_list, image_width, image_height):
    t = time.time()
    overlay = image.copy()
    glow_layer = np.zeros_like(image)
    
    for hand_landmarks in hand_landmarks_list:
        points = []
        for landmark in hand_landmarks:
            point = (int(landmark.x * image_width), int(landmark.y * image_height))
            points.append(point)
        
        palm_indices = [0, 5, 9, 13, 17]
        palm_x = int(np.mean([points[i][0] for i in palm_indices]))
        palm_y = int(np.mean([points[i][1] for i in palm_indices]))
        palm_center = (palm_x, palm_y)
        
        pulse = abs(math.sin(t * 4)) * 0.5 + 0.5
        pulse2 = abs(math.sin(t * 6 + 1)) * 0.5 + 0.5
        
        primary_color = (255, 200, 0)
        secondary_color = (255, 100, 0)
        accent_color = (255, 255, 100)
        glow_color = (255, 150, 0)
        
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = points[start_idx]
            end_point = points[end_idx]
            cv2.line(glow_layer, start_point, end_point, glow_color, 8)
            cv2.line(glow_layer, start_point, end_point, primary_color, 4)
            cv2.line(overlay, start_point, end_point, accent_color, 2)
        
        for i, point in enumerate(points):
            radius = int(12 + pulse * 4)
            cv2.circle(glow_layer, point, radius, glow_color, 2)
            cv2.circle(overlay, point, 8, primary_color, 2)
            cv2.circle(overlay, point, 4, accent_color, -1)
            if i in [4, 8, 12, 16, 20]:
                cv2.circle(overlay, point, int(6 + pulse2 * 3), secondary_color, 2)
                cv2.circle(glow_layer, point, 18, glow_color, 1)
        
        for i in range(3):
            radius = int(40 + i * 20 + pulse * 8)
            thickness = 2 if i == 0 else 1
            alpha = 1.0 - (i * 0.3)
            color = tuple(int(c * alpha) for c in primary_color)
            cv2.circle(overlay, palm_center, radius, color, thickness)
        
        arc_radius = 70
        for i in range(6):
            angle_offset = (t * 2 + i * 60) % 360
            start_angle = int(angle_offset)
            end_angle = int(angle_offset + 30)
            cv2.ellipse(overlay, palm_center, (arc_radius, arc_radius), 
                       0, start_angle, end_angle, primary_color, 2)
        
        arc_radius2 = 50
        for i in range(4):
            angle_offset = (-t * 3 + i * 90) % 360
            start_angle = int(angle_offset)
            end_angle = int(angle_offset + 45)
            cv2.ellipse(overlay, palm_center, (arc_radius2, arc_radius2), 
                       0, start_angle, end_angle, secondary_color, 1)
        
        fingertips = [4, 8, 12, 16, 20]
        for tip_idx in fingertips:
            tip = points[tip_idx]
            dist = np.sqrt((tip[0] - palm_center[0])**2 + (tip[1] - palm_center[1])**2)
            if dist > 0:
                num_dashes = int(dist / 10)
                for j in range(num_dashes):
                    if j % 2 == 0:
                        t1 = j / num_dashes
                        t2 = (j + 1) / num_dashes
                        p1 = (int(palm_center[0] + t1 * (tip[0] - palm_center[0])),
                              int(palm_center[1] + t1 * (tip[1] - palm_center[1])))
                        p2 = (int(palm_center[0] + t2 * (tip[0] - palm_center[0])),
                              int(palm_center[1] + t2 * (tip[1] - palm_center[1])))
                        cv2.line(overlay, p1, p2, (200, 100, 0), 1)
        
        cv2.putText(overlay, f"HAND LOCK", 
                   (palm_x - 40, palm_y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, accent_color, 1)
        cv2.rectangle(overlay, (palm_x - 45, palm_y - 90), (palm_x + 45, palm_y - 70), primary_color, 1)
        cv2.putText(overlay, f"X:{palm_x:03d} Y:{palm_y:03d}", 
                   (palm_x - 35, palm_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.35, primary_color, 1)
    
    glow_layer = cv2.GaussianBlur(glow_layer, (21, 21), 0)
    result = cv2.addWeighted(image, 0.7, glow_layer, 0.5, 0)
    result = cv2.addWeighted(result, 1.0, overlay, 0.8, 0)
    
    return result

def world_to_screen(wx, wz, cx, horizon_y, ground_y, span_x):
    """Convert world coordinates to perspective screen coordinates."""
    sx = int(cx + (wx / WORLD_RADIUS) * span_x)
    z_norm = (wz + WORLD_RADIUS) / (2 * WORLD_RADIUS)
    sy = int(horizon_y + z_norm * (ground_y - horizon_y))
    # Scale objects smaller when further away (closer to horizon)
    scale = 0.3 + 0.7 * z_norm
    return sx, sy, scale


def update_wandering_humans():
    """Move wandering humans around randomly."""
    for h in wandering_humans:
        h["x"] += h["vx"]
        h["z"] += h["vz"]
        # Bounce off world edges
        if abs(h["x"]) > WORLD_RADIUS * 0.7:
            h["vx"] *= -1
            h["x"] = max(-WORLD_RADIUS * 0.7, min(WORLD_RADIUS * 0.7, h["x"]))
        if abs(h["z"]) > WORLD_RADIUS * 0.7:
            h["vz"] *= -1
            h["z"] = max(-WORLD_RADIUS * 0.7, min(WORLD_RADIUS * 0.7, h["z"]))
        # Occasionally change direction
        if random.random() < 0.005:
            h["vx"] = random.uniform(-0.02, 0.02)
            h["vz"] = random.uniform(-0.02, 0.02)


def draw_stick_figure(sim, px, py, scale, color):
    """Draw a small stick figure human at screen position."""
    s = max(0.4, scale)
    head_r = int(4 * s)
    body_len = int(14 * s)
    leg_len = int(10 * s)
    arm_len = int(8 * s)

    head_y = py - body_len - head_r
    # Head
    cv2.circle(sim, (px, head_y), head_r, color, -1)
    cv2.circle(sim, (px, head_y), head_r, (255, 255, 255), 1)
    # Body
    cv2.line(sim, (px, head_y + head_r), (px, py), color, max(1, int(2 * s)))
    # Arms
    cv2.line(sim, (px, head_y + head_r + int(4*s)),
             (px - arm_len, head_y + head_r + int(8*s)), color, max(1, int(2 * s)))
    cv2.line(sim, (px, head_y + head_r + int(4*s)),
             (px + arm_len, head_y + head_r + int(8*s)), color, max(1, int(2 * s)))
    # Legs
    cv2.line(sim, (px, py), (px - int(5*s), py + leg_len), color, max(1, int(2 * s)))
    cv2.line(sim, (px, py), (px + int(5*s), py + leg_len), color, max(1, int(2 * s)))


def get_lidar_detections(max_range=4.0):
    """Return list of objects/humans within lidar range of the robot."""
    detections = []
    for i, obj in enumerate(static_objects):
        if i == held_object_index:
            continue
        dx = obj["x"] - robot_x
        dz = obj["z"] - robot_z
        dist = math.sqrt(dx*dx + dz*dz)
        if dist <= max_range:
            detections.append({"type": obj["shape"].capitalize(), "dist": dist})
    for h in wandering_humans:
        dx = h["x"] - robot_x
        dz = h["z"] - robot_z
        dist = math.sqrt(dx*dx + dz*dz)
        if dist <= max_range:
            detections.append({"type": "Human", "dist": dist})
    detections.sort(key=lambda d: d["dist"])
    return detections


def render_robot_camera_view(view_w, view_h):
    """Render a first-person view from the robot's camera."""
    cam_type = toolbox_parts["Camera (View)"][selected_parts["Camera (View)"]]
    view = np.zeros((view_h, view_w, 3), dtype=np.uint8)

    if cam_type == "None":
        cv2.putText(view, "NO CAMERA", (view_w // 2 - 45, view_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
        return view

    # Camera properties by type
    fov_map = {"Standard": 60, "Wide-Angle": 100, "Thermal": 60}
    range_map = {"Standard": 5.0, "Wide-Angle": 5.0, "Thermal": 7.0}
    fov_deg = fov_map.get(cam_type, 60)
    max_range = range_map.get(cam_type, 5.0)
    half_fov = math.radians(fov_deg / 2)
    is_thermal = cam_type == "Thermal"

    # Sky / ground
    horizon = view_h // 3
    if is_thermal:
        view[:horizon] = (40, 20, 10)
        view[horizon:] = (50, 35, 15)
    else:
        view[:horizon] = (50, 40, 20)
        view[horizon:] = (40, 55, 40)

    # Ground lines for depth
    robot_rad = math.radians(robot_angle)
    for i in range(1, 8):
        gy = horizon + int((view_h - horizon) * i / 8)
        col = (60, 70, 50) if not is_thermal else (60, 40, 20)
        cv2.line(view, (0, gy), (view_w, gy), col, 1)

    # Gather visible objects in the robot's FOV
    visible = []
    for i, obj in enumerate(static_objects):
        if i == held_object_index:
            continue
        dx = obj["x"] - robot_x
        dz = obj["z"] - robot_z
        dist = math.sqrt(dx * dx + dz * dz)
        if dist > max_range or dist < 0.3:
            continue
        angle_to_obj = math.atan2(dx, dz)
        rel_angle = angle_to_obj - robot_rad
        # Normalize to [-pi, pi]
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
        if abs(rel_angle) < half_fov:
            visible.append({"type": "obj", "data": obj, "dist": dist, "angle": rel_angle})

    for h in wandering_humans:
        dx = h["x"] - robot_x
        dz = h["z"] - robot_z
        dist = math.sqrt(dx * dx + dz * dz)
        if dist > max_range or dist < 0.3:
            continue
        angle_to_obj = math.atan2(dx, dz)
        rel_angle = angle_to_obj - robot_rad
        rel_angle = (rel_angle + math.pi) % (2 * math.pi) - math.pi
        if abs(rel_angle) < half_fov:
            visible.append({"type": "human", "data": h, "dist": dist, "angle": rel_angle})

    # Sort back-to-front (far objects drawn first)
    visible.sort(key=lambda v: -v["dist"])

    # Draw each visible thing
    for v in visible:
        # Screen X from relative angle (negated so left/right are correct)
        sx = int(view_w / 2 - (v["angle"] / half_fov) * (view_w / 2))
        # Screen Y and size from distance
        depth_frac = 1.0 - (v["dist"] / max_range)
        sy = horizon + int((view_h - horizon) * (0.2 + 0.7 * depth_frac))
        scale = 0.3 + 0.7 * depth_frac

        if v["type"] == "obj":
            obj = v["data"]
            sz = int(obj["size"] * scale * 1.5)
            if is_thermal:
                col = (0, int(80 + 100 * depth_frac), int(150 + 100 * depth_frac))
            else:
                col = obj["color"]
            if obj["shape"] == "circle":
                cv2.circle(view, (sx, sy), sz, col, -1)
                cv2.circle(view, (sx, sy), sz, (255, 255, 255), 1)
            elif obj["shape"] == "square":
                cv2.rectangle(view, (sx - sz, sy - sz), (sx + sz, sy + sz), col, -1)
                cv2.rectangle(view, (sx - sz, sy - sz), (sx + sz, sy + sz), (255, 255, 255), 1)
            elif obj["shape"] == "triangle":
                pts = np.array([[sx, sy - sz], [sx - sz, sy + sz], [sx + sz, sy + sz]], np.int32)
                cv2.fillPoly(view, [pts], col)
                cv2.polylines(view, [pts], True, (255, 255, 255), 1)
        else:
            # Draw stick figure
            h_scale = scale * 0.8
            head_r = max(2, int(4 * h_scale))
            body_len = max(3, int(14 * h_scale))
            leg_len = max(2, int(10 * h_scale))
            arm_len = max(2, int(8 * h_scale))
            if is_thermal:
                col = (0, int(100 + 155 * depth_frac), int(200 + 55 * depth_frac))
            else:
                col = v["data"]["color"]
            head_y = sy - body_len - head_r
            cv2.circle(view, (sx, head_y), head_r, col, -1)
            lw = max(1, int(2 * h_scale))
            cv2.line(view, (sx, head_y + head_r), (sx, sy), col, lw)
            cv2.line(view, (sx, head_y + head_r + int(4 * h_scale)),
                     (sx - arm_len, head_y + head_r + int(8 * h_scale)), col, lw)
            cv2.line(view, (sx, head_y + head_r + int(4 * h_scale)),
                     (sx + arm_len, head_y + head_r + int(8 * h_scale)), col, lw)
            cv2.line(view, (sx, sy), (sx - int(5 * h_scale), sy + leg_len), col, lw)
            cv2.line(view, (sx, sy), (sx + int(5 * h_scale), sy + leg_len), col, lw)

    # Crosshair
    chx, chy = view_w // 2, view_h // 2
    ch_col = (0, 200, 200)
    cv2.line(view, (chx - 8, chy), (chx - 3, chy), ch_col, 1)
    cv2.line(view, (chx + 3, chy), (chx + 8, chy), ch_col, 1)
    cv2.line(view, (chx, chy - 8), (chx, chy - 3), ch_col, 1)
    cv2.line(view, (chx, chy + 3), (chx, chy + 8), ch_col, 1)

    return view


def draw_status_panel(sim):
    """Draw a small info box at the bottom-left showing equipped parts and lidar detections."""
    h, w, _ = sim.shape
    panel_w = 600
    panel_h = 200
    px, py = 10, h - panel_h - 10

    # Semi-transparent background
    overlay = sim.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, sim, 0.3, 0, sim)
    cv2.rectangle(sim, (px, py), (px + panel_w, py + panel_h), (100, 100, 100), 1)

    # --- Left side: Equipped Parts ---
    cv2.putText(sim, "EQUIPPED", (px + 10, py + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.line(sim, (px + 10, py + 30), (px + 120, py + 30), (80, 80, 80), 1)

    row = 0
    part_labels = [
        ("Chassis", toolbox_colors[0]),
        ("Wheels (WASD)", toolbox_colors[1]),
        ("Lidar", toolbox_colors[2]),
        ("Camera (View)", toolbox_colors[3]),
        ("Arm (SPACE)", toolbox_colors[4]),
    ]
    for cat, color in part_labels:
        val = toolbox_parts[cat][selected_parts[cat]]
        if val in ("None", "Off"):
            continue
        short_cat = cat.split(" (")[0]
        txt = f"{short_cat}: {val}"
        ty = py + 50 + row * 22
        cv2.circle(sim, (px + 16, ty - 5), 5, color, -1)
        cv2.putText(sim, txt, (px + 28, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        row += 1

    if row == 0:
        cv2.putText(sim, "No parts equipped", (px + 16, py + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    if held_object is not None:
        ty = py + 50 + row * 22
        cv2.circle(sim, (px + 16, ty - 5), 5, held_object["color"], -1)
        cv2.putText(sim, f"Holding: {held_object['shape'].capitalize()}", (px + 28, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    # --- Right side: Lidar Detections ---
    lidar_x = px + 280
    cv2.line(sim, (lidar_x - 5, py + 8), (lidar_x - 5, py + panel_h - 8), (60, 60, 60), 1)
    cv2.putText(sim, "LIDAR SCAN", (lidar_x + 6, py + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 240), 1)
    cv2.line(sim, (lidar_x + 6, py + 30), (lidar_x + 140, py + 30), (80, 80, 80), 1)

    lidar = toolbox_parts["Lidar"][selected_parts["Lidar"]]
    if lidar != "On":
        cv2.putText(sim, "OFFLINE", (lidar_x + 16, py + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    else:
        detections = get_lidar_detections()
        if not detections:
            cv2.putText(sim, "No contacts", (lidar_x + 16, py + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 200, 80), 1)
        else:
            for i, det in enumerate(detections[:8]):
                dy = py + 50 + i * 20
                dist_str = f"{det['dist']:.1f}m"
                label = f"{det['type']} - {dist_str}"
                if det["dist"] < 1.5:
                    col = (50, 50, 255)
                elif det["dist"] < 3.0:
                    col = (50, 200, 255)
                else:
                    col = (80, 200, 80)
                cv2.circle(sim, (lidar_x + 12, dy - 4), 4, col, -1)
                cv2.putText(sim, label, (lidar_x + 24, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)


def draw_sim(sim):
    sim[:] = 0
    h, w, _ = sim.shape
    cx, ground_y = w // 2, int(h * 0.75)
    horizon_y = int(h * 0.25)
    t = time.time()
    span_x = w * 0.4

    # Draw perspective grid
    num_lines = 60
    for i in range(-num_lines, num_lines + 1):
        x_bottom = cx + i * 30
        x_top = cx + int(i * 10)
        cv2.line(sim, (x_bottom, ground_y), (x_top, horizon_y), (30, 50, 50), 1)

    for j in range(1, 10):
        tj = j / 10.0
        y = int(horizon_y + tj * (ground_y - horizon_y))
        cv2.line(sim, (0, y), (w, y), (25, 45, 45), 1)

    # --- Draw static objects ---
    for i, obj in enumerate(static_objects):
        if i == held_object_index:
            continue
        ox, oy, oscale = world_to_screen(obj["x"], obj["z"], cx, horizon_y, ground_y, span_x)
        sz = int(obj["size"] * oscale)
        col = obj["color"]
        if obj["shape"] == "circle":
            cv2.circle(sim, (ox, oy), sz, col, -1)
            cv2.circle(sim, (ox, oy), sz, (255, 255, 255), 1)
        elif obj["shape"] == "square":
            cv2.rectangle(sim, (ox - sz, oy - sz), (ox + sz, oy + sz), col, -1)
            cv2.rectangle(sim, (ox - sz, oy - sz), (ox + sz, oy + sz), (255, 255, 255), 1)
        elif obj["shape"] == "triangle":
            pts = np.array([
                [ox, oy - sz],
                [ox - sz, oy + sz],
                [ox + sz, oy + sz],
            ], np.int32)
            cv2.fillPoly(sim, [pts], col)
            cv2.polylines(sim, [pts], True, (255, 255, 255), 1)

    # --- Draw wandering humans ---
    update_wandering_humans()
    for human in wandering_humans:
        hx, hy, hscale = world_to_screen(human["x"], human["z"], cx, horizon_y, ground_y, span_x)
        draw_stick_figure(sim, hx, hy, hscale, human["color"])

    # --- Draw robot ---
    sx, sy, _ = world_to_screen(robot_x, robot_z, cx, horizon_y, ground_y, span_x)
    draw_robot_model(sim, sx, sy, preview_scale=1.0)

    # --- Highlight nearest pickable block when arm is Extended ---
    arm = toolbox_parts["Arm (SPACE)"][selected_parts["Arm (SPACE)"]]
    if arm == "Extended" and held_object is None:
        for obj in static_objects:
            dx = obj["x"] - robot_x
            dz = obj["z"] - robot_z
            dist = math.sqrt(dx * dx + dz * dz)
            if dist < PICKUP_RANGE:
                ox, oy, oscale = world_to_screen(obj["x"], obj["z"], cx, horizon_y, ground_y, span_x)
                r = int(obj["size"] * oscale) + 6
                pulse = abs(math.sin(time.time() * 5)) * 0.5 + 0.5
                col = (0, int(255 * pulse), int(255 * pulse))
                cv2.circle(sim, (ox, oy), r, col, 2)
                break  # highlight only the closest

    # Status indicator lights
    accent_color = (255, 200, 0)
    for i in range(3):
        light_x = sx - 10 + i * 10
        light_pulse = abs(math.sin(t * 3 + i * 0.5)) * 0.8 + 0.2
        if robot_state == "IDLE":
            light_color = (0, 0, int(255 * light_pulse))
        elif robot_state == "STANDING":
            light_color = (0, int(255 * light_pulse), int(255 * light_pulse))
        elif robot_state == "TURNING":
            light_color = (0, int(200 * light_pulse), int(255 * light_pulse))
        elif robot_state == "FROZEN":
            light_color = (int(255 * light_pulse), 0, int(128 * light_pulse))
        else:
            light_color = (0, int(255 * light_pulse), 0)
        cv2.circle(sim, (light_x, sy - 14), 2, light_color, -1)


def draw_hud_frame(frame, current_gesture, stable_gesture):
    h, w = frame.shape[:2]
    t = time.time()
    cyan = (255, 200, 0)
    bright = (255, 255, 100)
    dim = (180, 100, 0)
    orange = (0, 140, 255)

    pulse = abs(math.sin(t * 3)) * 0.4 + 0.6

    bl = 60
    m = 15
    cv2.line(frame, (m, m), (m + bl, m), cyan, 2)
    cv2.line(frame, (m, m), (m, m + bl), cyan, 2)
    cv2.line(frame, (w - m, m), (w - m - bl, m), cyan, 2)
    cv2.line(frame, (w - m, m), (w - m, m + bl), cyan, 2)
    cv2.line(frame, (m, h - m), (m + bl, h - m), cyan, 2)
    cv2.line(frame, (m, h - m), (m, h - m - bl), cyan, 2)
    cv2.line(frame, (w - m, h - m), (w - m - bl, h - m), cyan, 2)
    cv2.line(frame, (w - m, h - m), (w - m, h - m - bl), cyan, 2)

    scan_y = int((t * 100) % h)
    cv2.line(frame, (m + 5, scan_y), (w - m - 5, scan_y), dim, 1)

    cv2.rectangle(frame, (m + bl + 10, m - 5), (w - m - bl - 10, m + 25), (0, 0, 0), -1)
    cv2.rectangle(frame, (m + bl + 10, m - 5), (w - m - bl - 10, m + 25), cyan, 1)
    cv2.putText(frame, "SYSTEM ONLINE", (m + bl + 20, m + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bright, 1)

    panel_y = h // 2 - 60
    cv2.rectangle(frame, (m, panel_y), (m + 120, panel_y + 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (m, panel_y), (m + 120, panel_y + 120), cyan, 1)
    cv2.putText(frame, "STATUS", (m + 25, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cyan, 1)
    cv2.line(frame, (m + 5, panel_y + 28), (m + 115, panel_y + 28), dim, 1)
    cv2.putText(frame, f"DETECT:", (m + 8, panel_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.35, dim, 1)
    cv2.putText(frame, current_gesture[:8], (m + 8, panel_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, orange if current_gesture != "NONE" else dim, 1)
    cv2.putText(frame, f"LOCKED:", (m + 8, panel_y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.35, dim, 1)
    cv2.putText(frame, stable_gesture[:8], (m + 8, panel_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bright if stable_gesture != "NONE" else dim, 1)

    indicator_x = w - m - 30
    indicator_y = h // 2
    indicator_r = int(15 + pulse * 5)
    cv2.circle(frame, (indicator_x, indicator_y), indicator_r, cyan, 2)
    cv2.circle(frame, (indicator_x, indicator_y), 8, bright, -1)

    cv2.rectangle(frame, (m + bl + 10, h - m - 30), (w - m - bl - 10, h - m + 5), (0, 0, 0), -1)
    cv2.rectangle(frame, (m + bl + 10, h - m - 30), (w - m - bl - 10, h - m + 5), cyan, 1)
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(frame, f"TIME: {timestamp}  |  FPS: 30  |  LINK: ACTIVE  |  MODE: UNIFIED", 
               (m + bl + 20, h - m - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, cyan, 1)

    for i in range(5):
        bar_x = w - m - 25
        bar_y = panel_y + 10 + i * 18
        bar_w = int(20 * (0.3 + 0.7 * abs(math.sin(t * 2 + i))))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), cyan, -1)

    return frame


# ------------------ MAIN LOOP ------------------

def main():
    global current_gesture, action_gesture, robot_frozen, prev_action_fist

    print("\n" + "=" * 60)
    print("  J.A.R.V.I.S. GESTURE ROBOT ASSEMBLER + TOOLBOX")
    print("=" * 60)
    print("\n  Controls:")
    print("  • Fist   = STOP")
    print("  • Palm   = STAND")
    print("  • 1 finger  = FORWARD")
    print("  • 2 fingers = BACKWARD")
    print("  • 3 fingers = RIGHT")
    print("  • 4 fingers = LEFT")
    print("  • HOVER over toolbox to add parts to your robot!")
    print("  • PINCH over toolbox to cycle through variants")
    print("\n  Press 'Q' or ESC to quit")
    print("=" * 60 + "\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return

    window_name = "3D Gesture Robot Assembler"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        # Process mediapipe on standard camera resolution for speed
        frame_small = cv2.resize(frame, (CAM_W, CAM_H))
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = hand_landmarker.detect(mp_image)

        current_gesture = "NONE"
        action_gesture = "NONE"
        pinching = False
        cursor_x, cursor_y = -100, -100

        # We will draw the simulator on the full SIM_W x SIM_H canvas
        sim = np.zeros((SIM_H, SIM_W, 3), dtype=np.uint8)

        if results.hand_landmarks:
            movement_lm = None
            action_lm = None
            mv_handedness = "Right"

            # Assign hands by handedness
            for i, hand_lm in enumerate(results.hand_landmarks):
                label = results.handedness[i][0].category_name
                if label == "Right":
                    movement_lm = hand_lm
                    mv_handedness = "Right"
                elif label == "Left":
                    action_lm = hand_lm

            # Single-hand fallback: if only one hand, use it for movement
            if movement_lm is None and action_lm is not None:
                movement_lm = action_lm
                mv_handedness = "Left"
                action_lm = None

            # Process movement hand
            if movement_lm is not None:
                cursor_x = int(movement_lm[8].x * SIM_W)
                cursor_y = int(movement_lm[8].y * SIM_H)
                dx = movement_lm[8].x - movement_lm[4].x
                dy = movement_lm[8].y - movement_lm[4].y
                pinching = math.hypot(dx, dy) < 0.05
                current_gesture = classify_gesture(movement_lm, mv_handedness)

            # Process action hand
            if action_lm is not None:
                action_gesture = classify_gesture(action_lm, "Left")

        # Stabilize both gesture streams
        update_stable_gesture(current_gesture)
        update_stable_action_gesture(action_gesture)

        # Derive freeze from action hand open palm
        robot_frozen = (stable_action_gesture == "HOVER")

        # Derive pick/drop from action hand fist (edge-triggered)
        current_action_fist = (stable_action_gesture == "LAND")
        if current_action_fist and not prev_action_fist:
            if held_object is None:
                try_pickup()
            else:
                try_drop()
        prev_action_fist = current_action_fist

        # 1. Update toolbox selection logic (hover-to-add + pinch-to-cycle)
        cursor_in_ui = update_toolbox_logic(cursor_x, cursor_y, pinching)

        if not cursor_in_ui:
            update_robot()

        # Auto-drop if arm is no longer Extended while holding
        if held_object is not None:
            arm = toolbox_parts["Arm (SPACE)"][selected_parts["Arm (SPACE)"]]
            if arm != "Extended":
                try_drop()

        # 2. Draw the background and robot simulation onto `sim`
        draw_sim(sim)

        if robot_frozen:
            text_size = cv2.getTextSize("FROZEN", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            tx = (SIM_W - text_size[0]) // 2
            cv2.putText(sim, "FROZEN", (tx, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)

        # 3. Draw the toolbox UI overlay on top of the simulation
        draw_toolbox(sim, cursor_x, cursor_y)

        # 4. Draw status panel (equipped parts + lidar detections)
        draw_status_panel(sim)

        # 5. Draw Picture-in-Picture camera feed in the bottom right corner
        pip_size = 380
        # Crop frame to square (center crop) to avoid stretching
        fh, fw = frame.shape[:2]
        crop_dim = min(fw, fh)
        cx, cy = fw // 2, fh // 2
        half = crop_dim // 2
        frame_cropped = frame[cy - half : cy + half, cx - half : cx + half]
        pip_frame = cv2.resize(frame_cropped, (pip_size, pip_size))

        if results.hand_landmarks:
            # --- Draw virtual cursor ---
            color = (0, 255, 0) if pinching else (0, 200, 255)
            cv2.circle(sim, (cursor_x, cursor_y), 10, color, -1)
            cv2.circle(sim, (cursor_x, cursor_y), 14, (255, 255, 255), 2)

        pip_x_start = SIM_W - pip_size - 20
        pip_y_start = SIM_H - pip_size - 20
        sim[pip_y_start : pip_y_start + pip_size, pip_x_start : pip_x_start + pip_size] = pip_frame

        # PIP Border
        cv2.rectangle(sim, (pip_x_start, pip_y_start), (pip_x_start + pip_size, pip_y_start + pip_size), (255, 200, 0), 2)

        # 6. Robot camera view to the left of user camera (only if camera equipped)
        if toolbox_parts["Camera (View)"][selected_parts["Camera (View)"]] != "None":
            rv_w, rv_h = 320, 240
            robot_view = render_robot_camera_view(rv_w, rv_h)
            rv_x = pip_x_start - rv_w - 6
            rv_y = pip_y_start + pip_size - rv_h
            sim[rv_y : rv_y + rv_h, rv_x : rv_x + rv_w] = robot_view
            cv2.rectangle(sim, (rv_x, rv_y), (rv_x + rv_w, rv_y + rv_h), (255, 200, 0), 1)
            cv2.circle(sim, (rv_x + 10, rv_y + 12), 4, (0, 0, 220), -1)
            cv2.putText(sim, "Live", (rv_x + 18, rv_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        cv2.imshow(window_name, sim)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == 32:  # SPACE - pick up or drop
            if held_object is None:
                try_pickup()
            else:
                try_drop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
