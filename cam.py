# import cv2

# def initialize_camera():
#     for index in range(5):  # Try indices 0 to 4
#         cap = cv2.VideoCapture(index, cv2.CAP_FFMPEG)  # Use V4L2 backend explicitly
#         if cap.isOpened():
#             print(f"Camera initialized successfully at index {index}")
#             return cap
#         else:
#             cap.release()

#     print("Error: Failed to open webcam. Please check the camera connection or permissions.")
#     return None

# cap = initialize_camera()
# if not cap:
#     exit()  # Exit if no camera is available

# # Use the `cap` object for further processing


# import cv2

# for backend in [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_FFMPEG, cv2.CAP_DSHOW]:
#     cap = cv2.VideoCapture(0, backend)
#     if cap.isOpened():
#         print(f"Backend {backend} works!")
#         cap.release()
#     else:
#         print(f"Backend {backend} failed.")


import cv2

# Explicitly use the V4L2 backend
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open webcam using V4L2 backend.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
