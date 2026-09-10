import sys
import os

# Add root directory to sys.path so modules like gold_api, broker_api, etc. resolve properly
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from server import app

# Export app for Vercel Serverless Function
application = app
handler = app
