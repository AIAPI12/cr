#!/usr/bin/env bash
set -e
# Oracle VM setup script — run once on a fresh Oracle free-tier Ampere A1 instance
# Usage: bash setup_oracle.sh

echo "=== Oracle VM Setup for Clash Royale AI ==="

# Update system
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip nginx docker.io docker-compose

# Create checkpoint directory
sudo mkdir -p /opt/cr-checkpoints
sudo chown -R ubuntu:ubuntu /opt/cr-checkpoints

# Create project directory
mkdir -p /home/ubuntu/cr
cd /home/ubuntu/cr

# Clone repo on first setup (manual, needs token)
# git clone https://github.com/AIAPI12/cr.git .

# Create checkpoint server
cat > /home/ubuntu/cr/checkpoint_server.py << 'PYEOF'
#!/usr/bin/env python3
"""Minimal checkpoint file server — push/pull/list via HTTP."""
import http.server, os, json, sys, mimetypes

CHECKPOINT_DIR = '/opt/cr-checkpoints'
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/list':
            files = os.listdir(CHECKPOINT_DIR)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(files).encode())
        elif self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')
        else:
            fname = self.path.lstrip('/')
            fpath = os.path.join(CHECKPOINT_DIR, fname)
            if os.path.exists(fpath):
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.end_headers()
                with open(fpath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()

    def do_PUT(self):
        fname = self.path.lstrip('/')
        if not fname:
            self.send_response(400); self.end_headers(); return
        length = int(self.headers.get('Content-Length', 0))
        data = self.rfile.read(length)
        fpath = os.path.join(CHECKPOINT_DIR, fname)
        with open(fpath, 'wb') as f:
            f.write(data)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ok')

    def log_message(self, format, *args):
        pass  # quiet

httpd = http.server.HTTPServer(('0.0.0.0', PORT), Handler)
print(f'Checkpoint server on :{PORT}')
httpd.serve_forever()
PYEOF

chmod +x /home/ubuntu/cr/checkpoint_server.py

# Create systemd service
sudo tee /etc/systemd/system/cr-checkpoint.service > /dev/null << 'UNIT'
[Unit]
Description=Clash Royale Checkpoint Server
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/ubuntu/cr/checkpoint_server.py 8080
WorkingDirectory=/home/ubuntu/cr
Restart=always
User=ubuntu
Group=ubuntu

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable cr-checkpoint
sudo systemctl start cr-checkpoint

# Firewall
sudo ufw allow 8080/tcp 2>/dev/null || true

echo "=== Setup complete ==="
echo "Checkpoint server running on port 8080"
echo "Checkpoints stored in /opt/cr-checkpoints/"
echo ""
echo "To push checkpoint: curl -X PUT --data-binary @latest.pt http://<HOST>:8080/latest.pt"
echo "To pull checkpoint:  curl -o latest.pt http://<HOST>:8080/latest.pt"
echo "To list checkpoints: curl http://<HOST>:8080/list"
