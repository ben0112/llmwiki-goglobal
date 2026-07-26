import pytest
import vaultfs.postgres as postgres_module
from vaultfs.postgres import PostgresVaultFS

from llmwiki_core.postgres_retrieval import PostgresLexicalQuery
from llmwiki_core.search import SearchQuery


@pytest.mark.asyncio
async def test_serving_postgres_adapter_calls_shared_lexical_compiler(monkeypatch):
    user_id = "00000000-0000-0000-0000-000000000001"
    knowledge_base_id = "00000000-0000-0000-0000-000000000002"
    query = SearchQuery.build(text="private", limit=1, candidate_limit=2)
    compiled = PostgresLexicalQuery(
        sql="SELECT 'shared-compiler' WHERE $1::text = 'shared-param'",
        params=("shared-param",),
    )
    compiler_calls = []
    query_calls = []

    def compile_query(received_user_id, received_knowledge_base_id, received_query):
        compiler_calls.append(
            (received_user_id, received_knowledge_base_id, received_query)
        )
        return compiled

    async def scoped_query(received_user_id, sql, *params):
        query_calls.append((received_user_id, sql, params))
        return ()

    monkeypatch.setattr(
        postgres_module.postgres_retrieval,
        "compile_postgres_lexical_query",
        compile_query,
    )
    monkeypatch.setattr(postgres_module, "scoped_query", scoped_query)

    result = await PostgresVaultFS(user_id).retrieve(knowledge_base_id, query)

    assert compiler_calls == [(user_id, knowledge_base_id, query)]
    assert query_calls == [(user_id, compiled.sql, compiled.params)]
    assert result.profile == "lexical"
    assert result.hits == ()
