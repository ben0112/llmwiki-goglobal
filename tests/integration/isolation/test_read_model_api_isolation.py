from tests.helpers.jwt import auth_headers
from tests.integration.isolation.conftest import KB_B_ID, USER_A_ID


async def test_new_read_endpoints_hide_other_tenant(client):
    base = f"/v1/knowledge-bases/{KB_B_ID}"
    headers = auth_headers(USER_A_ID)
    requests = [
        ("get", f"{base}/documents/browse?path=/", None),
        ("get", f"{base}/wiki/pages", None),
        ("get", f"{base}/documents/resolve?document_number=1", None),
        ("post", f"{base}/documents/status", {"document_numbers": [1]}),
        ("post", f"{base}/documents/upload-preflight", {"items": [{"path": "/", "filename": "probe.md", "size": 1}]}),
    ]
    for method, url, body in requests:
        kwargs = {"headers": headers}
        if body is not None:
            kwargs["json"] = body
        response = await getattr(client, method)(url, **kwargs)
        assert response.status_code == 404
        assert str(KB_B_ID) not in response.text
