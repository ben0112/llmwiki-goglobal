from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_nine_production_callsites_have_json_contract_assertions():
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "tests/integration/test_durable_failure_matrix.py",
            ROOT / "tests/unit/jobs/test_dispatcher.py",
            ROOT / "tests/unit/test_hosted_quota.py",
            ROOT / "tests/unit/test_hosted_tus_multipart.py",
        )
    )
    compact = "".join(sources.split())
    for event in (
        "durable_job_dispatched",
        "durable_job_finished",
        "durable_job_lease_reaped",
        "tus_session_created",
        "tus_session_completed",
        "tus_session_stale",
        "quota_reserved",
        "quota_released",
        "upload_cleanup_finished",
    ):
        assert f'assert_telemetry_event(caplog,"{event}"' in compact
