#!/usr/bin/env python3
"""
Simple script to start the FastAPI web server for Electricity Signal Analyzer.
"""

import uvicorn
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("🔌 Starting Electricity Signal Analyzer Web Server")
    print("📊 Open your browser to: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "web_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
