#!/bin/bash
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 fcascan
"""start_web.py
Start CHAP-V Web Interface
by fcascan 2026
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and start web server
if __name__ == "__main__":
    import argparse
    from src.web.web_server import create_web_server
    
    parser = argparse.ArgumentParser(description='CHAP-V Web Interface')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to (default: 8080)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    try:
        web_server = create_web_server(host=args.host, port=args.port)
        web_server.run(debug=args.debug)
    except KeyboardInterrupt:
        print("\n[INFO] Web server stopped by user")
    except Exception as e:
        print(f"[ERROR] Error starting web server: {e}")
        sys.exit(1)