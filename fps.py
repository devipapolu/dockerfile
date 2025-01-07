import cv2

def get_stream_info(rtsp_url):
    # Open the RTSP stream
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        return None, None, None

    # Get FPS, resolution (width and height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()
    return fps, width, height

# Replace with your RTSP stream URLs
rtsp_url1 = "rtsp://admin:123456@192.168.1.142:554/Streaming/Channels/101"
rtsp_url2 = "rtsp://admin:123456@192.168.1.43:554/Streaming/Channels/101"

# Get FPS and resolution for the first stream
fps1, width1, height1 = get_stream_info(rtsp_url1)
if fps1 is None:
    print(f"Error: Unable to open RTSP stream {rtsp_url1}")
else:
    print(f"Stream 1 - FPS: {fps1}, Resolution: {width1}x{height1}")

# Get FPS and resolution for the second stream
fps2, width2, height2 = get_stream_info(rtsp_url2)
if fps2 is None:
    print(f"Error: Unable to open RTSP stream {rtsp_url2}")
else:
    print(f"Stream 2 - FPS: {fps2}, Resolution: {width2}x{height2}")

# Compare the streams
if fps1 and fps2:
    print("\nComparison:")
    print(f"Stream 1 FPS: {fps1}, Stream 2 FPS: {fps2}")
    if fps1 > fps2:
        print("Stream 1 has a higher frame rate.")
    elif fps1 < fps2:
        print("Stream 2 has a higher frame rate.")
    else:
        print("Both streams have the same frame rate.")
