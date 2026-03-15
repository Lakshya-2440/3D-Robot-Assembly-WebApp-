import re

with open('main.py', 'r') as f:
    content = f.read()

# 1. Update window resolution & remove aspect ratio letterboxing
# Make SIM and CAM the same size, overlay CAM on top of SIM instead of side-by-side.
# Or just make SIM full screen and put HUD on top.
content = re.sub(
    r"CAM_W, CAM_H = 640, 480\nSIM_W, SIM_H = 900, 700\nTOTAL_W, TOTAL_H = CAM_W \+ SIM_W, max\(CAM_H, SIM_H\)",
    """# Make SIM full screen, we won't show the raw camera side-by-side anymore.
# We'll assume a standard 16:9 ratio like 1280x720 for the full playground.
CAM_W, CAM_H = 640, 480
SIM_W, SIM_H = 1280, 720
TOTAL_W, TOTAL_H = SIM_W, SIM_H""",
    content,
    flags=re.DOTALL
)

# 2. Update Toolbox Drawing (Horizontal)
content = re.sub(
    r"def update_and_draw_toolbox\(sim, cx, cy, pinching\):.*?return cursor_in_ui",
    """def update_and_draw_toolbox(sim, cx, cy, pinching):
    global selected_parts, last_click_time
    import time
    
    cv2.putText(sim, "Toolbox", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    box_w = 200
    box_h = 50
    padding = 15
    start_x = 20
    start_y = 65
    
    cursor_in_ui = False
    if start_x <= cx <= start_x + len(toolbox_categories) * (box_w + padding) and start_y <= cy <= start_y + box_h:
        cursor_in_ui = True
        
    now = time.time()
    for i, cat in enumerate(toolbox_categories):
        bx = start_x + i * (box_w + padding)
        by = start_y
        
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
        cv2.putText(sim, text, (bx + 15, by + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    return cursor_in_ui""",
    content,
    flags=re.DOTALL
)

# 3. Main Loop: Remove Arc Reactor & Fix Window Compositing
main_loop_new = """# ------------------ MAIN LOOP ------------------

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
    print("  • PINCH over toolbox on the top to change parts!")
    print("\\n  Press 'Q' or ESC to quit")
    print("=" * 60 + "\\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return

    window_name = "3D Gesture Drone Simulator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, TOTAL_W, TOTAL_H)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        # We process mediapipe on the original CAM_W, CAM_H to keep it fast
        frame_small = cv2.resize(frame, (CAM_W, CAM_H))
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = hand_landmarker.detect(mp_image)
        # pose_results = pose_landmarker.detect(mp_image) # Arc reactor removed

        current_gesture = "NONE"
        pinching = False
        cursor_x, cursor_y = -100, -100

        # We will draw the simulator on the full SIM_W x SIM_H
        sim = np.zeros((SIM_H, SIM_W, 3), dtype=np.uint8)

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            
            # Map index finger (8) to FULL simulator coordinates
            cursor_x = int(hand_landmarks[8].x * SIM_W)
            cursor_y = int(hand_landmarks[8].y * SIM_H)
            
            dist = get_pinch_distance(results.hand_landmarks)
            pinching = dist < 0.05
            
            current_gesture = classify_gesture(hand_landmarks)
            
        update_stable_gesture(current_gesture)

        # Pre-check if cursor is over the UI bounds (horizontal now)
        # Box width is 200, padding 15, 5 boxes. Total width = 5 * 215 = 1075. Y is 65 to 115.
        cursor_in_ui = False
        if results.hand_landmarks and 0 <= cursor_x <= 1100 and 65 <= cursor_y <= 130:
            cursor_in_ui = True

        if not cursor_in_ui:
            update_drone()
            
        draw_sim(sim)
        update_and_draw_toolbox(sim, cursor_x, cursor_y, pinching)
        
        # Draw HUD over simulation
        # Instead of drawing HUD on the raw camera frame, we just draw the camera frame small in the corner, or not at all?
        # Let's draw the camera frame picture-in-picture in the bottom right corner
        pip_w, pip_h = 320, 240
        pip_frame = cv2.resize(frame, (pip_w, pip_h))
        
        # Draw hand landmarks on PIP if wanted, but using the original draw_landmarks takes the small frame
        if results.hand_landmarks:
           pip_frame = draw_landmarks_on_image(pip_frame, results.hand_landmarks, pip_w, pip_h)
        
        # Draw HUD frame logic directly onto sim? The HUD had brackets and text.
        # Let's just put the PIP in bottom right.
        sim[SIM_H - pip_h - 20 : SIM_H - 20, SIM_W - pip_w - 20 : SIM_W - 20] = pip_frame
        
        # Draw a border around PIP
        cv2.rectangle(sim, (SIM_W - pip_w - 20, SIM_H - pip_h - 20), (SIM_W - 20, SIM_H - 20), (255, 200, 0), 2)
        cv2.putText(sim, "CAMERA / TRACKING", (SIM_W - pip_w - 20, SIM_H - pip_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Draw cursor
        if results.hand_landmarks:
            color = (0, 255, 0) if pinching else (0, 200, 255)
            cv2.circle(sim, (cursor_x, cursor_y), 10, color, -1)
            cv2.circle(sim, (cursor_x, cursor_y), 14, (255, 255, 255), 2)

        cv2.imshow(window_name, sim)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


"""

content = re.sub(
    r"# ------------------ MAIN LOOP ------------------.*",
    main_loop_new + "\nif __name__ == '__main__':\n    main()\n",
    content,
    flags=re.DOTALL
)

with open('main.py', 'w') as f:
    f.write(content)
