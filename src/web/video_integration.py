# -*- coding: utf-8 -*-
"""video_integration.py
Video streaming integration for web interface
by fcascan 2025
"""
import cv2
import threading
import time
from queue import Queue

class VideoStreamManager:
    """Manages video streaming for web interface"""
    
    def __init__(self):
        self.frame_buffer = Queue(maxsize=5)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.active = False
        
    def start(self):
        """Start the video stream manager"""
        self.active = True
        
    def stop(self):
        """Stop the video stream manager"""
        self.active = False
        with self.frame_lock:
            self.latest_frame = None
            
    def update_frame(self, frame):
        """Update the latest frame for streaming
        
        Args:
            frame: OpenCV frame (numpy array)
        """
        if not self.active:
            return
            
        with self.frame_lock:
            self.latest_frame = frame.copy()
            
        # Add to buffer for processing
        try:
            if not self.frame_buffer.full():
                self.frame_buffer.put_nowait(frame.copy())
        except:
            pass  # Ignore if buffer is full
            
    def get_latest_frame(self):
        """Get the latest frame for streaming
        
        Returns:
            OpenCV frame or None if no frame available
        """
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
            
    def get_frame_buffer(self):
        """Get frames from buffer for processing
        
        Returns:
            Generator of frames
        """
        while self.active:
            try:
                frame = self.frame_buffer.get(timeout=1.0)
                yield frame
            except:
                continue

# Global video stream manager instance
video_stream_manager = VideoStreamManager()

def get_video_stream_manager():
    """Get the global video stream manager instance"""
    return video_stream_manager