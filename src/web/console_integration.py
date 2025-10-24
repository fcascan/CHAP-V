# -*- coding: utf-8 -*-
"""console_integration.py
Console logging integration for web interface
by fcascan 2025
"""
import sys
import time
import threading
from queue import Queue, Empty
from io import StringIO

class ConsoleCapture:
    """Captures console output for web interface"""
    
    def __init__(self, max_messages=1000):
        self.max_messages = max_messages
        self.message_queue = Queue(maxsize=max_messages)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.capturing = False
        self.captured_output = StringIO()
        
    def start_capture(self):
        """Start capturing console output"""
        if self.capturing:
            return
            
        self.capturing = True
        
        # Replace stdout and stderr
        sys.stdout = self
        sys.stderr = self
        
    def stop_capture(self):
        """Stop capturing console output"""
        if not self.capturing:
            return
            
        self.capturing = False
        
        # Restore original stdout and stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
    def write(self, text):
        """Write method called by print statements"""
        # Write to original stdout/stderr
        self.original_stdout.write(text)
        
        # Capture for web interface
        if self.capturing and text.strip():
            self._add_message('INFO', text.strip())
            
    def flush(self):
        """Flush method required by stdout/stderr interface"""
        self.original_stdout.flush()
        
    def _add_message(self, level, message):
        """Add a message to the queue"""
        timestamp = time.strftime('%H:%M:%S')
        
        msg_data = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        
        try:
            # Remove old messages if queue is full
            if self.message_queue.full():
                try:
                    self.message_queue.get_nowait()
                except Empty:
                    pass
                    
            self.message_queue.put_nowait(msg_data)
        except:
            pass  # Ignore if we can't add the message
            
    def add_log_message(self, level, message):
        """Manually add a log message"""
        self._add_message(level, message)
        
    def get_messages(self, max_count=50):
        """Get messages from the queue
        
        Args:
            max_count: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        messages = []
        count = 0
        
        while count < max_count:
            try:
                message = self.message_queue.get_nowait()
                messages.append(message)
                count += 1
            except Empty:
                break
                
        return messages
        
    def get_all_messages(self):
        """Get all messages from the queue"""
        messages = []
        
        while True:
            try:
                message = self.message_queue.get_nowait()
                messages.append(message)
            except Empty:
                break
                
        return messages
        
    def clear_messages(self):
        """Clear all messages from the queue"""
        while True:
            try:
                self.message_queue.get_nowait()
            except Empty:
                break

class WebLogger:
    """Logger that integrates with console capture for web display"""
    
    def __init__(self, console_capture):
        self.console_capture = console_capture
        
    def info(self, message):
        """Log an info message"""
        print(f"[INFO] {message}")
        self.console_capture.add_log_message('INFO', f"[INFO] {message}")

    def warning(self, message):
        """Log a warning message"""
        print(f"[WARNING] {message}")
        self.console_capture.add_log_message('WARNING', f"[WARNING] {message}")

    def error(self, message):
        """Log an error message"""
        print(f"[ERROR] {message}")
        self.console_capture.add_log_message('ERROR', f"[ERROR] {message}")

    def debug(self, message):
        """Log a debug message"""
        # Only show debug in console, not in web
        pass

# Global console capture instance
console_capture = ConsoleCapture()
web_logger = WebLogger(console_capture)

def get_console_capture():
    """Get the global console capture instance"""
    return console_capture

def get_web_logger():
    """Get the web logger instance"""
    return web_logger

def start_console_capture():
    """Start console capture globally"""
    console_capture.start_capture()
    
def stop_console_capture():
    """Stop console capture globally"""
    console_capture.stop_capture()