import cv2

def get_video_info(video_path:str) -> cv2.VideoCapture:
    '''
    Purpose: Read a video file and return its properties.
    Args:
        video_path (str): Path to the input video file.
    Returns:
        dict: A dictionary containing the video's properties such as fps, width, and height.
    '''
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return {"fps": fps, "frame_width": frame_width, "frame_height":frame_height, "frame_count": frame_count}