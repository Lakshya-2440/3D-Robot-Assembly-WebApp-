import re

with open('main.py', 'r') as f:
    content = f.read()

# Replace APP MODE and BUILDER STATE
content = re.sub(
    r"# ------------------ APP MODE ------------------.*?last_nav_time = 0\.0  # cooldown timer for builder navigation",
    """# ------------------ TOOLBOX STATE ------------------

toolbox_categories = ["Chassis", "Wheels (WASD)", "Lidar", "Camera (View)", "Arm (SPACE)"]
toolbox_parts = {
    "Chassis":       ["Standard", "Racing", "Heavy-Lift", "Stealth"],
    "Wheels (WASD)": ["Standard", "Off-Road", "Racing", "None"],
    "Lidar":         ["Off", "On"],
    "Camera (View)": ["Standard", "Wide-Angle", "Thermal", "None"],
    "Arm (SPACE)":   ["Retracted", "Extended"],
}

selected_parts = {cat: 0 for cat in toolbox_categories}

toolbox_colors = [
    (110, 150, 60),   # Chassis (Teal BGR)
    (50, 110, 220),   # Wheels (Orange BGR)
    (50, 50, 200),    # Lidar (Red BGR)
    (230, 110, 60),   # Camera (Blue BGR)
    (220, 50, 150),   # Arm (Purple BGR)
]

last_click_time = 0.0
CLICK_COOLDOWN = 0.5""",
    content,
    flags=re.DOTALL
)

# Replace BUILDER LOGIC
content = re.sub(
    r"# ------------------ BUILDER LOGIC ------------------.*?if acted:\n        last_nav_time = now",
    """# ------------------ CURSOR LOGIC ------------------

def get_pinch_distance(landmarks_list):
    if not landmarks_list:
        return 1.0
    lm = landmarks_list[0]
    dx = lm[8].x - lm[4].x
    dy = lm[8].y - lm[4].y
    return (dx**2 + dy**2)**0.5""",
    content,
    flags=re.DOTALL
)

# Replace _get_drone_colors and draw_drone_preview start
content = re.sub(
    r"def _get_drone_colors\(\):.*?camera = builder_parts\[\"CAMERA\"\]\[selected_parts\[\"CAMERA\"\]\]",
    """def draw_drone_preview(sim, cx, cy, preview_scale=1.0):
    \"\"\"Draw a 2D drone preview that reflects selected parts.\"\"\"
    t = time.time()
    body_col = (80, 80, 80)
    accent_col = (255, 200, 0)
    
    chassis = toolbox_parts["Chassis"][selected_parts["Chassis"]]
    wheels = toolbox_parts["Wheels (WASD)"][selected_parts["Wheels (WASD)"]]
    lidar = toolbox_parts["Lidar"][selected_parts["Lidar"]]
    camera = toolbox_parts["Camera (View)"][selected_parts["Camera (View)"]]
    arm = toolbox_parts["Arm (SPACE)"][selected_parts["Arm (SPACE)"]]""",
    content,
    flags=re.DOTALL
)

# Replace Wheels to Central LED (Lines 315-334 roughly)
content = re.sub(
    r"    # --- Wheels ---.*?    # --- Central LED ---",
    """    # --- Wheels ---
    if wheels != "None":
        wheel_r = int(5 * s) if wheels == "Standard" else int(7 * s) if wheels == "Off-Road" else int(4 * s)
        wheel_color = accent_col if wheels == "Racing" else (100, 100, 100)
        offsets = [(-int(20 * s), int(18 * s)), (int(20 * s), int(18 * s))]
        for ox, oy in offsets:
            wx, wy = cx + ox, cy + oy
            cv2.circle(sim, (wx, wy), wheel_r, wheel_color, -1)
            cv2.circle(sim, (wx, wy), wheel_r, accent_col, 1)

    # --- Lidar ---
    if lidar == "On":
        cv2.circle(sim, (cx, cy - int(15 * s)), int(5 * s), (0, 0, 255), -1)
        pulse_lidar = abs(math.sin(t * 10))
        cv2.circle(sim, (cx, cy - int(15 * s)), int(8 * s + pulse_lidar * 5), (0, 0, 255), 1)

    # --- Camera mount ---
    if camera != "None":
        cam_y = cy + int(12 * s)
        cam_w = int(8 * s) if camera == "Standard" else int(14 * s) if camera == "Wide-Angle" else int(10 * s)
        cam_h = int(6 * s)
        cam_color = (200, 200, 200) if camera != "Thermal" else (0, 100, 255)
        cv2.rectangle(sim, (cx - cam_w // 2, cam_y), (cx + cam_w // 2, cam_y + cam_h), cam_color, -1)
        cv2.rectangle(sim, (cx - cam_w // 2, cam_y), (cx + cam_w // 2, cam_y + cam_h), accent_col, 1)
        cv2.circle(sim, (cx, cam_y + cam_h // 2), int(2 * s), (0, 0, 0), -1)

    # --- Arm ---
    if arm == "Extended":
        arm_end_y = cy + int(30 * s)
        cv2.line(sim, (cx, cy), (cx, arm_end_y), (150, 150, 150), int(4 * s))
        cv2.line(sim, (cx - int(5 * s), arm_end_y + int(5 * s)), (cx, arm_end_y), (150, 150, 150), int(2 * s))
        cv2.line(sim, (cx + int(5 * s), arm_end_y + int(5 * s)), (cx, arm_end_y), (150, 150, 150), int(2 * s))
    elif arm == "Retracted":
        cv2.line(sim, (cx, cy), (cx, cy + int(10 * s)), (150, 150, 150), int(4 * s))

    # --- Central LED ---""",
    content,
    flags=re.DOTALL
)

# Remove draw_builder_sim and draw_builder_hud, and add update_and_draw_toolbox
content = re.sub(
    r"def draw_builder_sim\(sim\):.*?def draw_landmarks_on_image",
    """def update_and_draw_toolbox(sim, cx, cy, pinching):
    global selected_parts, last_click_time
    import time
    
    cv2.putText(sim, "Toolbox", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    start_y = 80
    box_w = 260
    box_h = 50
    padding = 15
    start_x = 20
    
    cursor_in_ui = False
    
    if 0 <= cx <= start_x + box_w + 30 and 0 <= cy <= start_y + len(toolbox_categories) * (box_h + padding) + 20:
        cursor_in_ui = True
        
    now = time.time()
    for i, cat in enumerate(toolbox_categories):
        bx = start_x
        by = start_y + i * (box_h + padding)
        
        is_hover = (bx <= cx <= bx + box_w) and (by <= cy <= by + box_h)
        color = toolbox_colors[i]
        
        if is_hover:
            cv2.rectangle(sim, (bx, by), (bx + box_w, by + box_h), color, -1)
            cv2.rectangle(sim, (bx-2, by-2), (bx + box_w + 2, by + box_h + 2), (255, 255, 255), 2)
            
            if pinching and (now - last_click_time > CLICK_COOLDOWN):
                selected_parts[cat] = (selected_parts[cat] + 1) % len(toolbox_parts[cat])
                last_click_time = now
        else:
            cv2.rectangle(sim, (bx, by), (bx + box_w, by + box_h), color, -1)
            
        active_part = toolbox_parts[cat][selected_parts[cat]]
        text = f"{cat}: {active_part}"
        cv2.putText(sim, text, (bx + 15, by + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    return cursor_in_ui


def draw_landmarks_on_image""",
    content,
    flags=re.DOTALL
)

# Update build info status text
content = re.sub(
    r"    # Build info\n.*?cv2\.putText\(sim, f\"Build: \{chassis\} / \{color\}\",\n                \(20, 160\), cv2\.FONT_HERSHEY_SIMPLEX, 0\.5, accent_color, 1\)",
    "",
    content,
    flags=re.DOTALL
)

content = re.sub(
    r"_, accent_color = _get_drone_colors\(\)",
    "accent_color = (255, 200, 0)",
    content,
    flags=re.DOTALL
)


# Replace main loop entirely to simplify changes
main_loop_replacement = """# ------------------ MAIN LOOP ------------------

def main():
    global current_gesture

    print("\\n" + "=" * 60)
    print("  J.A.R.V.I.S. GESTURE DRONE SIMULATOR + TOOLBOX")
    print("=" * 60)
    print("\\n  Controls:")
    print("  • Fist   = LAND")
    print("  • Palm   = HOVER")
    print("  • 1 finger  = FORWARD")
    print("  • 2 fingers = BACKWARD")
    print("  • 3 fingers = RIGHT")
    print("  • 4 fingers = LEFT")
    print("  • PINCH over toolbox on the left to change parts!")
    print("\\n  Press 'Q' or ESC to quit")
    print("=" * 60 + "\\n")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera!")
        return

    window_name = "3D Gesture Drone Simulator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, TOTAL_W, TOTAL_H)
    
    cursor_x, cursor_y = -100, -100
    pinching = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (CAM_W, CAM_H))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = hand_landmarker.detect(mp_image)
        pose_results = pose_landmarker.detect(mp_image)

        current_gesture = "NONE"
        pinching = False

        if pose_results.pose_landmarks:
            frame = draw_arc_reactor(frame, pose_results.pose_landmarks[0], CAM_W, CAM_H)

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            frame = draw_landmarks_on_image(frame, results.hand_landmarks, CAM_W, CAM_H)
            
            # Extract cursor
            cursor_x = int(hand_landmarks[8].x * SIM_W)
            cursor_y = int(hand_landmarks[8].y * SIM_H)
            
            dist = get_pinch_distance(results.hand_landmarks)
            pinching = dist < 0.05
            
            current_gesture = classify_gesture(hand_landmarks)
            
        update_stable_gesture(current_gesture)

        sim = np.zeros((SIM_H, SIM_W, 3), dtype=np.uint8)

        # Pre-check if cursor is in UI bounds
        cursor_in_ui = False
        if results.hand_landmarks and 0 <= cursor_x <= 20 + 260 + 30 and 0 <= cursor_y <= 80 + 5 * 65 + 20:
            cursor_in_ui = True

        if not cursor_in_ui:
            update_drone()
            
        draw_sim(sim)
        
        # Draw UI on top
        update_and_draw_toolbox(sim, cursor_x, cursor_y, pinching)
        
        # Draw cursor
        if results.hand_landmarks:
            color = (0, 255, 0) if pinching else (0, 200, 255)
            cv2.circle(sim, (cursor_x, cursor_y), 10, color, -1)
            cv2.circle(sim, (cursor_x, cursor_y), 14, (255, 255, 255), 2)

        frame = draw_hud_frame(frame, current_gesture, stable_gesture)

        combined = np.zeros((TOTAL_H, TOTAL_W, 3), dtype=np.uint8)
        combined[:SIM_H, :SIM_W] = sim
        cam_y_offset = (TOTAL_H - CAM_H) // 2
        combined[cam_y_offset:cam_y_offset + CAM_H, SIM_W:SIM_W + CAM_W] = frame

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
"""

content = re.sub(
    r"# ------------------ MAIN LOOP ------------------.*",
    main_loop_replacement,
    content,
    flags=re.DOTALL
)

with open('main.py', 'w') as f:
    f.write(content)
