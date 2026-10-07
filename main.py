from flask import Flask, send_from_directory
import logging
from pathlib import Path
from routes import initialize_routes
from cache import create_cache

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend" / "public"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# Initialize the cache database
create_cache()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize API routes
initialize_routes(app)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
