# -*- coding: utf-8 -*-
"""Shared helpers for processing overlays and rolling FPS metrics."""

import cv2


def calculate_recent_average_ms(samples):
    """Return the average duration in milliseconds for a list of sample durations."""
    if not samples:
        return 0.0
    return (sum(samples) / len(samples)) * 1000.0


def calculate_recent_fps(timestamps):
    """Return the FPS computed from a list of monotonically increasing timestamps."""
    if len(timestamps) < 2:
        return 0.0

    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return 0.0

    return (len(timestamps) - 1) / elapsed


def draw_processing_overlay(frame, enabled, title_text, inference_time_ms=None, fps_value=None, text_size=0.5):
    """Draw the standard processing overlay if enabled."""
    if not enabled or frame is None:
        return frame

    cv2.putText(frame, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), 2)

    if inference_time_ms is not None:
        cv2.putText(frame, f"Inf time: {inference_time_ms:.1f} ms", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 255), 2)

    if fps_value is not None:
        cv2.putText(frame, f"FPS: {fps_value:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 0), 2)

    return frame