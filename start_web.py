#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 fcascan
"""start_web.py
Start CHAP-V Web Interface
"""

import sys
import os

# Put the project root, src/ and src/rockchip/ on sys.path. The rockchip modules use flat,
# co-located imports (e.g. `from coco_utils import ...`), so without src/rockchip on the path the
# first Start-Processing crashes with ModuleNotFoundError. main.py gets this via app_launcher; this
# standalone launcher must set it up itself.
project_root = os.path.dirname(os.path.abspath(__file__))
for _p in (project_root, os.path.join(project_root, "src"), os.path.join(project_root, "src", "rockchip")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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