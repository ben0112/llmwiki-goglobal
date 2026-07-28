from tests.helpers.jwt import auth_headers
from tests.integration.isolation.conftest import KB_B_ID, USER_A_ID


async def test_corpus_and_graph_summaries_hide_other_tenant(client):
    headers = auth_headers(USER_A_ID)
    for suffix in ("corpus/entries", "corpus/summary", "graph/summary"):
        response = await client.get(f"/v1/knowledge-bases/{KB_B_ID}/{suffix}", headers=headers)
        assert response.status_code == 404
        assert str(KB_B_ID) not in response.text
