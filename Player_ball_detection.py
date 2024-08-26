import cv2
import numpy as np

def detect_ball_and_player(img):
    # Split channels
    b, g, r = cv2.split(img)

    # Remove ground
    g_greaterthan_r = np.greater(g, r)
    g_greaterthan_b = np.greater(g, b)
    ground_removed = np.logical_and(g_greaterthan_r, g_greaterthan_b)

    # Inverse 0 and 1 to make foreground objects white (1) and background black (0)
    ground_removed = np.logical_not(ground_removed).astype(np.uint8)

    # Morphological operation
    kernel = np.ones((11,11), np.uint8)
    result = cv2.morphologyEx(ground_removed, cv2.MORPH_CLOSE, kernel)    

    # Find connected components
    connected_components = cv2.connectedComponentsWithStats(result, 4, cv2.CV_32S)
    (num_components, component_ids, values, centroid) = connected_components

    # Filter out useful components by area
    for i in range(1, num_components):
        area = values[i, cv2.CC_STAT_AREA]
        left = values[i, cv2.CC_STAT_LEFT] 
        top = values[i, cv2.CC_STAT_TOP] 
        width = values[i, cv2.CC_STAT_WIDTH] 
        height = values[i, cv2.CC_STAT_HEIGHT]     

        # Mark ball
        if area > 50 and area < 100:
            cv2.rectangle(img, (left, top), (left + width, top + height), (0, 0, 255), 3)
        
        # Mark player
        if area > 200 and area < 2500:
            cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 255), 3)
            
    return img

# Read the video and process frame by frame in chunks
input_video_path = 'D:/IIT Jodhpur/new_implement/2.mp4'
output_video_path = 'D:/IIT Jodhpur/new_implement/output2.mp4'

cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

# Define chunk size (number of frames per chunk)
chunk_size = 30  # Adjust based on performance

frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Detect the ball and players in the frame
    processed_frame = detect_ball_and_player(frame)
    frames.append(processed_frame)
    
    if out is None and len(frames) > 0:
        # Initialize the VideoWriter with the same width, height, and FPS as the input video
        height, width, _ = frame.shape
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if len(frames) == chunk_size:
        for f in frames:
            out.write(f)
        frames = []

# Write remaining frames (if any)
if out is not None:
    for f in frames:
        out.write(f)

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to {output_video_path}")
