from datetime import datetime, timezone
import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.services.api.routers.auth import get_auth_service, router
from backend.services.api.user_app.schemas.user import TokenResponse, UserResponse


class RecordingAuthService:
    def __init__(self):
        self.calls = []

    async def login_local_single_user(self, *, ip_address=None, user_agent=None):
        self.calls.append({"ip_address": ip_address, "user_agent": user_agent})
        return TokenResponse(
            access_token="header.payload.signature",
            refresh_token="refresh-token",
            expires_in=900,
            user=UserResponse(
                user_id="10000001",
                tenant_id="default",
                username="admin",
                email="admin@quantmind.com",
                is_active=True,
                is_verified=True,
                is_admin=True,
                created_at=datetime.now(timezone.utc),
                last_login_at=datetime.now(timezone.utc),
            ),
        )


class LocalSingleUserAuthTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        self.service = RecordingAuthService()
        app.dependency_overrides[get_auth_service] = lambda: self.service
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_local_session_is_hidden_when_mode_is_disabled(self):
        with patch.dict(
            os.environ,
            {
                "LOCAL_SINGLE_USER_MODE": "false",
                "LOCAL_SINGLE_USER_BIND_ADDRESS": "127.0.0.1",
            },
        ):
            response = self.client.post(
                "/api/v1/auth/local-session",
                headers={
                    "host": "localhost:8000",
                    "origin": "http://localhost:3000",
                },
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.service.calls, [])

    def test_local_session_rejects_non_loopback_hosts(self):
        bad_headers = (
            ("192.168.3.22:8000", "http://localhost:3000"),
            ("localhost:8000", "http://192.168.3.22:3000"),
            ("localhost.attacker.test:8000", "http://localhost:3000"),
            ("localhost:not-a-port", "http://localhost:3000"),
            ("user@localhost:8000", "http://localhost:3000"),
        )
        with patch.dict(
            os.environ,
            {
                "LOCAL_SINGLE_USER_MODE": "true",
                "LOCAL_SINGLE_USER_BIND_ADDRESS": "127.0.0.1",
            },
        ):
            for host, origin in bad_headers:
                with self.subTest(host=host, origin=origin):
                    response = self.client.post(
                        "/api/v1/auth/local-session",
                        headers={"host": host, "origin": origin},
                    )
                    self.assertEqual(response.status_code, 403)

        self.assertEqual(self.service.calls, [])

    def test_local_session_returns_admin_jwt_for_loopback(self):
        with patch.dict(
            os.environ,
            {
                "LOCAL_SINGLE_USER_MODE": "true",
                "LOCAL_SINGLE_USER_BIND_ADDRESS": "127.0.0.1",
            },
        ):
            response = self.client.post(
                "/api/v1/auth/local-session",
                headers={
                    "host": "127.0.0.1:8000",
                    "origin": "http://localhost:3000",
                    "user-agent": "local-session-test",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["user"]["username"], "admin")
        self.assertTrue(payload["user"]["is_admin"])
        self.assertEqual(len(payload["access_token"].split(".")), 3)
        self.assertEqual(len(self.service.calls), 1)
        self.assertEqual(
            self.service.calls[0]["user_agent"], "local-session-test"
        )


if __name__ == "__main__":
    unittest.main()
