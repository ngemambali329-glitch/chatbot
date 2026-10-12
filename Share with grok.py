"""Optional: share the chatbot through a public ngrok link (handy from Colab or a laptop).

    pip install pyngrok
    export NGROK_AUTHTOKEN=...        # free token from https://dashboard.ngrok.com
    python scripts/share_with_ngrok.py
"""
import os
import subprocess
import sys

from pyngrok import ngrok

PORT = 8501
token = os.environ.get("NGROK_AUTHTOKEN")
if not token:
    sys.exit("Set the NGROK_AUTHTOKEN environment variable first.")
ngrok.set_auth_token(token)

server = subprocess.Popen([
    sys.executable, "-m", "streamlit", "run", "app.py",
    "--server.port", str(PORT), "--server.headless", "true",
    "--server.enableCORS", "false", "--server.enableXsrfProtection", "false",
])
try:
    print("Public URL:", ngrok.connect(PORT).public_url)
    server.wait()
finally:
    ngrok.kill()
    server.terminate()
