import sys
import os
import urllib.parse

# Add root directory to sys.path so modules like gold_api, broker_api, etc. resolve properly
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from server import app

class VercelPathMiddleware:
    """
    WSGI Middleware to restore original request path from Vercel rewrite parameter.
    Allows single serverless function to route all /api/* subpaths seamlessly.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        qs = environ.get("QUERY_STRING", "")
        params = urllib.parse.parse_qs(qs)
        if "__route" in params:
            route = params["__route"][0]
            if not route.startswith("/"):
                route = "/" + route
            environ["PATH_INFO"] = "/api" + route
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathMiddleware(app.wsgi_app)

# Export app for Vercel Serverless Function
application = app
handler = app
