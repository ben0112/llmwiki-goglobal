#!/usr/bin/env python3
"""生成局域网自部署所需的全部密钥,输出 KEY=VALUE 行(供 bootstrap.sh 拼装 .env.lan)。

关键点:api 服务只认 JWKS(``{SUPABASE_URL}/auth/v1/.well-known/jwks.json``,
按 kid 匹配),所以 GoTrue 必须用非对称密钥签发 —— 这里生成 EC P-256 私钥,
装成带 kid 的私有 JWK 交给 GOTRUE_JWT_KEYS;GoTrue 会自动在 JWKS 端点公开
对应公钥。SUPABASE_ANON_KEY 也用同一把钥签(supabase-js 要求它是合法 JWT)。

通常不直接运行,由 bootstrap.sh 在一次性 python:3.11-alpine 容器里调用:
  docker run --rm -v "$PWD/gen_secrets.py:/gen.py:ro" python:3.11-alpine \
    sh -c 'pip install -q "PyJWT[crypto]" && python /gen.py'
"""
import json
import secrets
import time

import jwt
from cryptography.hazmat.primitives.asymmetric import ec


def main() -> None:
    # ES256 签名密钥(GoTrue 签发 / api 经 JWKS 验签)
    key = ec.generate_private_key(ec.SECP256R1())
    kid = secrets.token_hex(8)

    private_jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(key))
    private_jwk.update({"kid": kid, "use": "sig", "alg": "ES256"})
    gotrue_jwt_keys = json.dumps([private_jwk], separators=(",", ":"))

    # supabase-js 的 anon key:角色 anon 的长效 JWT(10 年),同一把钥签发
    now = int(time.time())
    anon_key = jwt.encode(
        {"role": "anon", "iss": "supabase", "iat": now, "exp": now + 315360000},
        key,
        algorithm="ES256",
        headers={"kid": kid},
    )

    print(f"POSTGRES_PASSWORD={secrets.token_urlsafe(24)}")
    print(f"JWT_SECRET={secrets.token_urlsafe(32)}")
    print(f"MINIO_ROOT_PASSWORD={secrets.token_urlsafe(24)}")
    print(f"AWS_SECRET_ACCESS_KEY={secrets.token_urlsafe(24)}")
    print(f"CONVERTER_SECRET={secrets.token_urlsafe(24)}")
    # JSON 串放 env 文件安全:以 [ 开头、无 $,compose 不会做变量展开
    print(f"GOTRUE_JWT_KEYS={gotrue_jwt_keys}")
    print(f"SUPABASE_ANON_KEY={anon_key}")


if __name__ == "__main__":
    main()
