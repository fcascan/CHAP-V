# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 fcascan
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


def draw_processing_overlay(frame, enabled, title_text, inference_time_ms=None, fps_value=None, text_size=0.5, text_color=(0, 255, 0)):
    """Draw the standard processing overlay if enabled.
    
    Args:
        frame: OpenCV frame to draw on
        enabled: Whether to draw the overlay
        title_text: Text for the title line
        inference_time_ms: Inference time in milliseconds (optional)
        fps_value: FPS value (optional)
        text_size: Font scale for text
        text_color: BGR color tuple for all overlay text (default: green)
    """
    if not enabled or frame is None:
        return frame

    cv2.putText(frame, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 2)

    if inference_time_ms is not None:
        cv2.putText(frame, f"Inf time: {inference_time_ms:.1f} ms", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 2)

    if fps_value is not None:
        cv2.putText(frame, f"FPS: {fps_value:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 2)

    return frame