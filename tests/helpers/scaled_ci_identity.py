"""Generate an ephemeral ES256 identity and self-host env for scaled CI."""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from uuid import UUID

import jwt
from cryptography.hazmat.primitives.asymmetric import ec
from jwt.algorithms import ECAlgorithm

USER_ID = UUID("d7cba9c8-55f4-4ffc-8e81-279fa4b41c03")
KID = "scaled-ci-es256"
ISSUER = "http://ci-auth:8000/auth/v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jwks-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()

    key = ec.generate_private_key(ec.SECP256R1())
    jwk = ECAlgorithm.to_jwk(key.public_key(), as_dict=True)
    jwk.update({"kid": KID, "use": "sig", "alg": "ES256"})
    target = args.jwks_root / "auth/v1/.well-known/jwks.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    jwks_json = json.dumps({"keys": [jwk]}, separators=(",", ":"))
    target.write_text(jwks_json, encoding="utf-8")

    now = int(time.time())
    token = jwt.encode(
        {
            "sub": str(USER_ID),
            "aud": "authenticated",
            "iss": ISSUER,
            "iat": now,
            "exp": now + 3600,
            "role": "authenticated",
        },
        key,
        algorithm="ES256",
        headers={"kid": KID},
    )
    values = {
        "APP_URL": "http://127.0.0.1:3000",
        "API_URL": "http://127.0.0.1:8000",
        "MCP_URL": "http://127.0.0.1:8080",
        "SUPABASE_URL": "http://ci-auth:8000",
        "SUPABASE_ANON_KEY": "ephemeral-ci-only",
        "CI_JWKS_B64": base64.b64encode(jwks_json.encode()).decode("ascii"),
        "DATABASE_URL": "postgresql://postgres:postgres@ci-postgres:5432/postgres",
        "DIRECT_DATABASE_URL": "postgresql://postgres:postgres@ci-postgres:5432/postgres",
        "CI_POSTGRES_PORT": "55434",
        "MINIO_ROOT_USER": "scaled-ci-root",
        "MINIO_ROOT_PASSWORD": "scaled-ci-root-secret",
        "AWS_ACCESS_KEY_ID": "scaled-ci-access",
        "AWS_SECRET_ACCESS_KEY": "scaled-ci-secret",
        "AWS_REGION": "us-east-1",
        "S3_BUCKET": "llmwiki-scaled-ci",
        "S3_ENDPOINT_URL": "http://minio:9000",
        "CONVERTER_SECRET": "scaled-ci-converter-secret",
        "STAGE": "test",
        "DURABLE_JOBS_ENABLED": "true",
        "TUS_MULTIPART_ENABLED": "true",
        "SCALED_COMPOSE_TEST": "1",
        "SCALED_TEST_API_URL": "http://127.0.0.1:8000",
        "SCALED_TEST_DATABASE_URL": "postgresql://postgres:postgres@127.0.0.1:55434/postgres",
        "SCALED_TEST_TOKEN": token,
        "SCALED_TEST_USER_ID": str(USER_ID),
    }
    args.env_file.write_text("".join(f"{name}={value}\n" for name, value in values.items()), encoding="utf-8")


if __name__ == "__main__":
    main()
