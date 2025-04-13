import cv2
import numpy as np
import serial
import time
# ---------------------- HSV Color Ranges ----------------------

COLOR_RANGES = {
    "Red1":   ([0, 150, 100], [10, 255, 255]),      # Lower red hue
    "Red2":   ([170, 150, 100], [180, 255, 255]),   # Upper red hue
    "G":  ([40, 70, 70], [80, 255, 255]),       # Green range
    "B":   ([90, 100, 100], [140, 255, 255]),    # Blue range
}

# MORPHOLOGICAL KERNEL
KERNEL = np.ones((5, 5), np.uint8)
#MINIMAL CONTOUR AREA TO DRAW THE CONTOUR
MIN_CONTOUR_AREA = 1000
#----------------------- UART Communication Configuration ----------------------
ser = serial.Serial('COM6', baudrate=115200, timeout=1)
# ---------------------- Webcam Capture : 0 for PC WEBCAM ----------------------

cap = cv2.VideoCapture(0)
#CAMERA ERROR HANDLING
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ---------------------- Main Loop ----------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Convert for HSV for better Color detection 

    # Detect Red, Green, and Blue
    for color_name in ["R", "G", "B"]:
        lower, upper = COLOR_RANGES[color_name] if color_name != "R" else (COLOR_RANGES["Red1"], COLOR_RANGES["Red2"])

        if color_name == "R":  # For Red, we need to combine two ranges
            mask1 = cv2.inRange(hsv_frame, np.array(lower[0]), np.array(lower[1]))
            mask2 = cv2.inRange(hsv_frame, np.array(upper[0]), np.array(upper[1]))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                print(color_name)
                ser.write(color_name.encode())
            

    # Show result
    cv2.imshow("Color Detection", frame)
    cv2.imshow("HSV Capture",hsv_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------- Cleanup ----------------------

cap.release()
cv2.destroyAllWindows()
