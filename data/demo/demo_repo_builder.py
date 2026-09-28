"""
Demo Repo Generator: Creates a realistic multi-component Git repository with authentic evolution history for offline/demo reviews.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DEMO_DATA_DIR


DEMO_REPO_PATH = DEMO_DATA_DIR / "softwarepulse_demo_repo"


def create_demo_repository(force_recreate: bool = False) -> Path:
    """Creates a local Git repository with multi-commit evolution, branches, and modular dependencies."""
    if not force_recreate and DEMO_REPO_PATH.exists() and (DEMO_REPO_PATH / ".git").exists():
        # Quick check if it has multiple authors
        try:
            res = subprocess.run(["git", "log", "--format=%an"], cwd=str(DEMO_REPO_PATH), capture_output=True, text=True)
            authors = set(res.stdout.strip().split("\n"))
            if len(authors) >= 2:
                return DEMO_REPO_PATH
        except Exception:
            pass

    if DEMO_REPO_PATH.exists():
        shutil.rmtree(DEMO_REPO_PATH, ignore_errors=True)

    DEMO_REPO_PATH.mkdir(parents=True, exist_ok=True)
    services_dir = DEMO_REPO_PATH / "services"
    core_dir = DEMO_REPO_PATH / "core"
    api_dir = DEMO_REPO_PATH / "api"
    utils_dir = DEMO_REPO_PATH / "utils"

    for d in [services_dir, core_dir, api_dir, utils_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Initialize Git
    subprocess.run(["git", "init"], cwd=str(DEMO_REPO_PATH), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Alex Developer"], cwd=str(DEMO_REPO_PATH), capture_output=True)
    subprocess.run(["git", "config", "user.email", "alex@softwarepulse.dev"], cwd=str(DEMO_REPO_PATH), capture_output=True)

    # File 1: utils/logger.py
    logger_code = '''"""Logging utilities."""
import datetime

class PulseLogger:
    def __init__(self, name: str):
        self.name = name

    def info(self, msg: str):
        print(f"[{datetime.datetime.now()}] [INFO] [{self.name}] {msg}")

    def error(self, msg: str):
        print(f"[{datetime.datetime.now()}] [ERROR] [{self.name}] {msg}")
'''
    (utils_dir / "logger.py").write_text(logger_code, encoding="utf-8")

    # File 2: core/database.py
    db_code = '''"""Database interface module."""
from utils.logger import PulseLogger

class DatabaseConnection:
    def __init__(self, uri: str = "sqlite:///pulse.db"):
        self.uri = uri
        self.logger = PulseLogger("Database")
        self.is_connected = False

    def connect(self):
        if not self.is_connected:
            self.logger.info("Opening DB connection")
            self.is_connected = True
        return self.is_connected

    def execute_query(self, query: str, params: dict = None):
        if not self.is_connected:
            self.connect()
        self.logger.info(f"Executing query: {query}")
        return [{"id": 1, "status": "active"}]
'''
    (core_dir / "database.py").write_text(db_code, encoding="utf-8")

    # File 3: services/auth_service.py
    auth_code = '''"""Authentication and Token verification service."""
from core.database import DatabaseConnection
from utils.logger import PulseLogger

class AuthService:
    def __init__(self):
        self.db = DatabaseConnection()
        self.logger = PulseLogger("AuthService")
        self.token_cache = {}

    def authenticate_user(self, username: str, token: str) -> bool:
        if not username or not token:
            self.logger.error("Empty credentials")
            return False
        
        if username in self.token_cache:
            return self.token_cache[username] == token
            
        res = self.db.execute_query("SELECT token FROM users WHERE username = :u", {"u": username})
        if res:
            self.token_cache[username] = token
            return True
        return False

    def revoke_token(self, username: str):
        if username in self.token_cache:
            del self.token_cache[username]
            self.logger.info(f"Revoked token for {username}")
'''
    (services_dir / "auth_service.py").write_text(auth_code, encoding="utf-8")

    # File 4: services/payment_service.py (High complexity, high churn, multi-branching)
    payment_code = '''"""Payment Processing Gateway & Transaction Engine."""
import time
from core.database import DatabaseConnection
from services.auth_service import AuthService
from utils.logger import PulseLogger

class PaymentProcessor:
    def __init__(self):
        self.db = DatabaseConnection()
        self.auth = AuthService()
        self.logger = PulseLogger("PaymentService")
        self.retry_limit = 3

    def process_transaction(self, user: str, token: str, amount: float, currency: str, card_info: dict) -> dict:
        if not self.auth.authenticate_user(user, token):
            self.logger.error("Authentication failed during payment processing")
            return {"status": "error", "code": 401, "message": "Unauthorized"}

        if amount <= 0:
            self.logger.error("Invalid transaction amount")
            return {"status": "failed", "code": 400, "message": "Non-positive amount"}

        if currency not in ["USD", "EUR", "GBP", "JPY"]:
            if currency == "CAD" or currency == "AUD":
                amount = amount * 0.75
            else:
                return {"status": "rejected", "code": 422, "message": "Unsupported currency"}

        # Nested validation logic & high cyclomatic complexity
        card_num = card_info.get("number", "")
        if len(card_num) < 16:
            for digit in card_num:
                if not digit.isdigit():
                    return {"status": "invalid_card", "code": 400}
            return {"status": "short_card_number", "code": 400}

        retries = 0
        success = False
        while retries < self.retry_limit and not success:
            try:
                self.logger.info(f"Attempting charge {amount} {currency} for {user}")
                self.db.execute_query("INSERT INTO transactions VALUES (:u, :a)", {"u": user, "a": amount})
                success = True
            except Exception as e:
                retries += 1
                self.logger.error(f"Charge attempt {retries} failed: {e}")
                time.sleep(0.01)

        if not success:
            return {"status": "gateway_timeout", "code": 504}

        return {"status": "success", "tx_id": "TX99482", "amount": amount}
'''
    (services_dir / "payment_service.py").write_text(payment_code, encoding="utf-8")

    # File 5: api/routes.py
    routes_code = '''"""API HTTP Routing Controller."""
from services.payment_service import PaymentProcessor
from services.auth_service import AuthService
from utils.logger import PulseLogger

class ApiController:
    def __init__(self):
        self.payment_proc = PaymentProcessor()
        self.auth = AuthService()
        self.logger = PulseLogger("ApiController")

    def handle_checkout(self, request_payload: dict):
        user = request_payload.get("user")
        token = request_payload.get("token")
        amount = request_payload.get("amount", 0.0)
        card = request_payload.get("card", {})
        return self.payment_proc.process_transaction(user, token, amount, "USD", card)

    def handle_login(self, request_payload: dict):
        return self.auth.authenticate_user(request_payload.get("user"), request_payload.get("token"))
'''
    (api_dir / "routes.py").write_text(routes_code, encoding="utf-8")

    # Commit 1: Initial commit
    subprocess.run(["git", "add", "."], cwd=str(DEMO_REPO_PATH), capture_output=True)
    subprocess.run(["git", "commit", "--author=Aditya <aditya@softwarepulse.dev>", "-m", "feat: initial commit with modular microservices and auth"], cwd=str(DEMO_REPO_PATH), capture_output=True)

    # Commit 2: Add validation to payment_service
    payment_code += "\n    def refund_transaction(self, tx_id: str, reason: str):\n        self.logger.info(f'Refund requested for {tx_id}: {reason}')\n        return True\n"
    (services_dir / "payment_service.py").write_text(payment_code, encoding="utf-8")
    subprocess.run(["git", "add", "services/payment_service.py"], cwd=str(DEMO_REPO_PATH), capture_output=True)
    subprocess.run(["git", "commit", "--author=Harsh <harsh@softwarepulse.dev>", "-m", "feat: implement refund transaction capabilities"], cwd=str(DEMO_REPO_PATH), capture_output=True)

    # Commit 3: Bugfix commit in payment_service
    payment_code = payment_code.replace("self.retry_limit = 3", "self.retry_limit = 5  # fix: increase retry limits on high load")
    (services_dir / "payment_service.py").write_text(payment_code, encoding="utf-8")
    subprocess.run(["git", "add", "services/payment_service.py"], cwd=str(DEMO_REPO_PATH), capture_output=True)
    subprocess.run(["git", "commit", "--author=Yash <yash@softwarepulse.dev>", "-m", "fix: resolve timeout and transaction crash under concurrency leak"], cwd=str(DEMO_REPO_PATH), capture_output=True)

    # Commit 4: Bugfix in auth_service
    auth_code = auth_code.replace("return True", "return True # fix: secure token validation patch")
    (services_dir / "auth_service.py").write_text(auth_code, encoding="utf-8")
    subprocess.run(["git", "add", "services/auth_service.py"], cwd=str(DEMO_REPO_PATH), capture_output=True)
    subprocess.run(["git", "commit", "--author=Pranav <pranav@softwarepulse.dev>", "-m", "fix: patch vulnerability in session revocation bug"], cwd=str(DEMO_REPO_PATH), capture_output=True)

    # Commit 5: Update database & routes
    routes_code += "\n    def health_check(self):\n        return {'status': 'healthy'}\n"
    (api_dir / "routes.py").write_text(routes_code, encoding="utf-8")
    subprocess.run(["git", "add", "api/routes.py"], cwd=str(DEMO_REPO_PATH), capture_output=True)
    subprocess.run(["git", "commit", "--author=Harsh <harsh@softwarepulse.dev>", "-m", "chore: add system health check telemetry endpoint"], cwd=str(DEMO_REPO_PATH), capture_output=True)

    # Commit 6: Churn in payment service
    payment_code += "\n    def verify_webhook_signature(self, sig: str):\n        return len(sig) == 64\n"
    (services_dir / "payment_service.py").write_text(payment_code, encoding="utf-8")
    subprocess.run(["git", "add", "services/payment_service.py"], cwd=str(DEMO_REPO_PATH), capture_output=True)
    subprocess.run(["git", "commit", "--author=Harsh <harsh@softwarepulse.dev>", "-m", "fix: repair webhook verification error and payload parsing"], cwd=str(DEMO_REPO_PATH), capture_output=True)

    return DEMO_REPO_PATH


if __name__ == "__main__":
    p = create_demo_repository()
    print(f"Created demo repository at: {p}")
