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
        # Multi-camera support
        self.camera_frames = {}
        self.camera_locks = {}
        self.num_cameras = 1
        
    def start(self):
        """Start the video stream manager"""
        self.active = True
        
    def stop(self):
        """Stop the video stream manager"""
        self.active = False
        with self.frame_lock:
            self.latest_frame = None
        # Clear all camera frames
        for camera_id in list(self.camera_frames.keys()):
            with self.camera_locks.get(camera_id, threading.Lock()):
                self.camera_frames[camera_id] = None
                
    def set_camera_count(self, count):
        """Set the number of active cameras"""
        self.num_cameras = count
        # Initialize locks for each camera
        for i in range(count):
            if i not in self.camera_locks:
                self.camera_locks[i] = threading.Lock()
                self.camera_frames[i] = None
            
    def update_frame(self, frame, camera_id=0):
        """Update the latest frame for streaming
        
        Args:
            frame: OpenCV frame (numpy array)
            camera_id: ID of the camera (default 0 for backward compatibility)
        """
        if not self.active:
            return
            
        # Update main frame (for backward compatibility)
        if camera_id == 0:
            with self.frame_lock:
                self.latest_frame = frame.copy()
                
        # Update specific camera frame
        if camera_id in self.camera_locks:
            with self.camera_locks[camera_id]:
                self.camera_frames[camera_id] = frame.copy()
            
        # Add to buffer for processing
        try:
            if not self.frame_buffer.full():
                self.frame_buffer.put_nowait(frame.copy())
        except:
            pass  # Ignore if buffer is full
            
    def get_latest_frame(self, camera_id=None):
        """Get the latest frame for streaming
        
        Args:
            camera_id: ID of the camera, if None returns main frame
            
        Returns:
            OpenCV frame or None if no frame available
        """
        if camera_id is None:
            # Return main frame (backward compatibility)
            with self.frame_lock:
                return self.latest_frame.copy() if self.latest_frame is not None else None
        else:
            # Return specific camera frame
            if camera_id in self.camera_locks:
                with self.camera_locks[camera_id]:
                    frame = self.camera_frames.get(camera_id)
                    return frame.copy() if frame is not None else None
            return None
            
    def get_camera_count(self):
        """Get the number of active cameras"""
        return self.num_cameras
            
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