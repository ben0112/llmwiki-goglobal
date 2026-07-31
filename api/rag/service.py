"""Authenticated application service for durable server-side RAG runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import NoReturn
from uuid import UUID, uuid4

import asyncpg
from jobs.models import JobCreate, JobRecord, JobState, JobType
from jobs.service import JobResourceNotFound, JobService

from llmwiki_core.rag import (
    RAG_ERROR_CONTRACTS,
    RagBudget,
    RagDomainError,
    RagErrorContract,
    RagRunConfig,
    new_rag_error,
)
from llmwiki_core.signals import sanitized_boundary_signal_or_unknown

from . import repository
from .model import ResolvedRagModelProfile, resolve_model_profiles
from .records import RagRunRecord, RagStepRecord
from .retrieval import validate_hosted_rag_retrieval_settings

_BUDGET_FIELDS = frozenset(asdict(RagBudget()))
_IDEMPOTENCY_KEY_MAX_BYTES = 800
_UUID_INT_DESCRIPTOR = UUID.__dict__["int"]
_CREATE_PREFLIGHT_RAG_ERROR_CONTRACT_NAMES = frozenset(
    {
        "disabled",
        "invalid_request",
        "model_profile_unavailable",
        "retrieval_profile_unavailable",
    }
)
_RESUME_PREFLIGHT_RAG_ERROR_CONTRACT_NAMES = _CREATE_PREFLIGHT_RAG_ERROR_CONTRACT_NAMES | {"resume_not_allowed"}
_CREATE_ROOT_RAG_ERRORS = frozenset(
    RAG_ERROR_CONTRACTS[name] for name in repository.CREATE_ROOT_RAG_ERROR_CONTRACT_NAMES
)
_CREATE_RESUME_RAG_ERRORS = frozenset(
    RAG_ERROR_CONTRACTS[name] for name in repository.CREATE_RESUME_RAG_ERROR_CONTRACT_NAMES
)
_CREATE_RAG_ERRORS = _CREATE_ROOT_RAG_ERRORS | frozenset(
    RAG_ERROR_CONTRACTS[name] for name in _CREATE_PREFLIGHT_RAG_ERROR_CONTRACT_NAMES
)
_RESUME_RAG_ERRORS = _CREATE_RESUME_RAG_ERRORS | frozenset(
    RAG_ERROR_CONTRACTS[name] for name in _RESUME_PREFLIGHT_RAG_ERROR_CONTRACT_NAMES
)
_READ_RAG_ERRORS = frozenset({RAG_ERROR_CONTRACTS["invalid_request"]})
_NO_RAG_ERRORS: frozenset[RagErrorContract] = frozenset()


def _error(contract_name: str) -> RagDomainError:
    return new_rag_error(contract_name)


def _fresh_error(contract: RagErrorContract) -> RagDomainError:
    return RagDomainError(contract.code, contract.public_message, contract.retryable)


def _mapped_failure(
    failure: BaseException,
    *,
    allowed_contracts: frozenset[RagErrorContract],
    invalid_request: bool = False,
    allow_job_resource_not_found: bool = False,
) -> BaseException:
    control = sanitized_boundary_signal_or_unknown(failure)
    if control is not None:
        return _error("internal_error") if type(control) is BaseException else control
    if invalid_request:
        return _error("invalid_request")
    if type(failure) is RagDomainError:
        try:
            code = failure.code
            public_message = failure.public_message
            retryable = failure.retryable
            if type(code) is not str or type(public_message) is not str or type(retryable) is not bool:
                raise TypeError
            candidate = RagErrorContract(code, public_message, retryable)
            matched = next(
                (contract for contract in allowed_contracts if candidate == contract),
                None,
            )
        except BaseException:  # noqa: BLE001 - corrupted public error objects fail closed.
            return _error("internal_error")
        if matched is not None:
            return _fresh_error(matched)
    if allow_job_resource_not_found and isinstance(failure, JobResourceNotFound):
        return JobResourceNotFound("referenced job resource was not found")
    return _error("internal_error")


def _raise_sanitized(
    failure: BaseException,
    *,
    allowed_contracts: frozenset[RagErrorContract],
    invalid_request: bool = False,
    allow_job_resource_not_found: bool = False,
) -> NoReturn:
    raise _mapped_failure(
        failure,
        allowed_contracts=allowed_contracts,
        invalid_request=invalid_request,
        allow_job_resource_not_found=allow_job_resource_not_found,
    ) from None


def _canonical_uuid(value: object) -> UUID:
    if type(value) is UUID:
        return value
    if not isinstance(value, UUID):
        raise TypeError("UUID required")
    raw_int = _UUID_INT_DESCRIPTOR.__get__(value, UUID)
    if type(raw_int) is not int or not 0 <= raw_int < 1 << 128:
        raise TypeError("UUID required")
    return UUID(int=raw_int)


def _canonical_digest(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _budget_payload(budget: RagBudget) -> dict[str, int]:
    if type(budget) is not RagBudget:
        raise TypeError("budget must be a RagBudget")
    values = asdict(budget)
    if set(values) != _BUDGET_FIELDS or any(type(value) is not int for value in values.values()):
        raise TypeError("budget is invalid")
    return values


@dataclass(frozen=True, slots=True, repr=False)
class CreateRagRun:
    """Normalized immutable application command for a new root run."""

    knowledge_base_id: UUID
    goal: str
    target_path_prefix: str
    model_profile: str
    retrieval_profile: str = "lexical"
    dry_run: bool = False
    budget: RagBudget = field(default_factory=RagBudget)

    def __post_init__(self) -> None:
        failure: BaseException | None = None
        try:
            knowledge_base_id = _canonical_uuid(self.knowledge_base_id)
            if type(self.goal) is not str or type(self.target_path_prefix) is not str:
                raise TypeError
            if type(self.model_profile) is not str or type(self.retrieval_profile) is not str:
                raise TypeError
            if type(self.dry_run) is not bool or type(self.budget) is not RagBudget:
                raise TypeError
            normalized = RagRunConfig.build(
                knowledge_base_id=knowledge_base_id,
                goal=self.goal,
                target_path_prefix=self.target_path_prefix,
                model_profile=self.model_profile,
                retrieval_profile=self.retrieval_profile,
                dry_run=self.dry_run,
                budget=self.budget,
            )
            _budget_payload(normalized.budget)
        except BaseException as caught:  # noqa: BLE001 - sanitize the public command boundary.
            failure = caught
        if failure is not None:
            _raise_sanitized(
                failure,
                allowed_contracts=_READ_RAG_ERRORS,
                invalid_request=True,
            )
        object.__setattr__(self, "knowledge_base_id", knowledge_base_id)
        object.__setattr__(self, "goal", normalized.goal)
        object.__setattr__(self, "target_path_prefix", normalized.target_path_prefix)
        object.__setattr__(self, "model_profile", normalized.model_profile)
        object.__setattr__(self, "retrieval_profile", normalized.retrieval_profile)
        object.__setattr__(self, "budget", normalized.budget)

    @property
    def config(self) -> RagRunConfig:
        return RagRunConfig(
            knowledge_base_id=self.knowledge_base_id,
            goal=self.goal,
            target_path_prefix=self.target_path_prefix,
            model_profile=self.model_profile,
            retrieval_profile=self.retrieval_profile,
            dry_run=self.dry_run,
            budget=self.budget,
        )

    @property
    def request_digest(self) -> str:
        return _canonical_digest(
            {
                "knowledge_base_id": str(self.knowledge_base_id),
                "goal": self.goal,
                "target_path_prefix": self.target_path_prefix,
                "model_profile": self.model_profile,
                "retrieval_profile": self.retrieval_profile,
                "dry_run": self.dry_run,
                "budget": _budget_payload(self.budget),
            }
        )


@dataclass(frozen=True, slots=True, repr=False)
class ResumeRagRun:
    """Normalized immutable application command for a resumed attempt."""

    parent_run_id: UUID
    budget: RagBudget

    def __post_init__(self) -> None:
        failure: BaseException | None = None
        try:
            parent_run_id = _canonical_uuid(self.parent_run_id)
            if type(self.budget) is not RagBudget:
                raise TypeError
            _budget_payload(self.budget)
        except BaseException as caught:  # noqa: BLE001 - sanitize the public command boundary.
            failure = caught
        if failure is not None:
            _raise_sanitized(
                failure,
                allowed_contracts=_READ_RAG_ERRORS,
                invalid_request=True,
            )
        object.__setattr__(self, "parent_run_id", parent_run_id)

    @property
    def request_digest(self) -> str:
        return _canonical_digest(
            {
                "parent_run_id": str(self.parent_run_id),
                "budget": _budget_payload(self.budget),
            }
        )


def _require_uuid(value: object) -> UUID:
    try:
        return _canonical_uuid(value)
    except (TypeError, ValueError, AttributeError):
        raise _error("invalid_request") from None


def _require_idempotency_key(value: object) -> str:
    try:
        encoded = value.encode("utf-8") if type(value) is str else b""
    except UnicodeEncodeError:
        encoded = b""
    if (
        type(value) is not str
        or not 1 <= len(value) <= 200
        or not encoded
        or len(encoded) > _IDEMPOTENCY_KEY_MAX_BYTES
        or value.strip() != value
        or "\x00" in value
    ):
        raise _error("invalid_request")
    return value


def _materialize_budget_override(value: object) -> dict[str, int]:
    failure: BaseException | None = None
    try:
        if not isinstance(value, Mapping):
            raise TypeError
        materialized: dict[str, int] = {}
        for index, key in enumerate(iter(value)):
            if index >= len(_BUDGET_FIELDS):
                raise ValueError
            if type(key) is not str or key not in _BUDGET_FIELDS or key in materialized:
                raise ValueError
            item = value[key]
            if type(item) is not int:
                raise TypeError
            materialized[key] = item
    except BaseException as caught:  # noqa: BLE001 - hostile Mapping boundary.
        failure = caught
    if failure is not None:
        _raise_sanitized(
            failure,
            allowed_contracts=_READ_RAG_ERRORS,
            invalid_request=True,
        )
    return materialized


def _resume_budget(parent: RagBudget, overrides: Mapping[str, int]) -> RagBudget:
    failure: BaseException | None = None
    try:
        if type(parent) is not RagBudget or type(overrides) is not dict:
            raise TypeError
        if any(value <= getattr(parent, key) for key, value in overrides.items()):
            raise ValueError
        budget = replace(parent, **overrides)
        _budget_payload(budget)
    except BaseException as caught:  # noqa: BLE001 - detach invalid input details.
        failure = caught
    if failure is not None:
        _raise_sanitized(
            failure,
            allowed_contracts=_READ_RAG_ERRORS,
            invalid_request=True,
        )
    return budget


class RagService:
    """Own authenticated, atomic run creation and resume transactions."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        settings: object,
        *,
        job_service: JobService | None = None,
    ) -> None:
        self._pool = pool
        self._settings = settings
        self._job_service = job_service or JobService(pool)

    def _require_enabled(self) -> None:
        failure: BaseException | None = None
        try:
            enabled = self._settings.SERVER_RAG_ENABLED  # type: ignore[attr-defined]
        except BaseException as caught:  # noqa: BLE001 - settings are an adapter boundary.
            failure = caught
        if failure is not None:
            _raise_sanitized(failure, allowed_contracts=_NO_RAG_ERRORS)
        if enabled is not True:
            raise _error("disabled")

    def _resolve_profile(self, name: str) -> ResolvedRagModelProfile:
        failure: BaseException | None = None
        try:
            profiles = resolve_model_profiles(self._settings)
            profile = profiles[name]
            if (
                not isinstance(profile, ResolvedRagModelProfile)
                or profile.name != name
                or not 1 <= len(profile.version) <= 128
            ):
                raise ValueError
        except BaseException as caught:  # noqa: BLE001 - profile details and keys are private.
            failure = caught
        if failure is not None:
            control = sanitized_boundary_signal_or_unknown(failure)
            if control is not None:
                raise control from None
            if isinstance(failure, RagDomainError):
                raise _mapped_failure(failure, allowed_contracts=_NO_RAG_ERRORS) from None
            raise _error("model_profile_unavailable") from None
        return profile

    async def _require_retrieval_profile(self, name: str) -> None:
        if name == "lexical":
            return
        failure: BaseException | None = None
        try:
            if name != "hybrid":
                raise ValueError
            validate_hosted_rag_retrieval_settings(self._settings, request_limit=1)
        except BaseException as caught:  # noqa: BLE001 - sanitize configuration failures.
            failure = caught
        if failure is not None:
            control = sanitized_boundary_signal_or_unknown(failure)
            if control is not None:
                raise control from None
            if isinstance(failure, RagDomainError):
                raise _mapped_failure(failure, allowed_contracts=_NO_RAG_ERRORS) from None
            raise _error("retrieval_profile_unavailable") from None

    async def create(
        self,
        command: CreateRagRun,
        *,
        authenticated_user_id: UUID,
        idempotency_key: str,
    ) -> RagRunRecord:
        failure: BaseException | None = None
        try:
            if type(command) is not CreateRagRun:
                raise _error("invalid_request")
            user_id = _require_uuid(authenticated_user_id)
            key = _require_idempotency_key(idempotency_key)
            self._require_enabled()
            profile = self._resolve_profile(command.model_profile)
            await self._require_retrieval_profile(command.retrieval_profile)
            run_id = uuid4()
            job_command = JobCreate(
                job_type=JobType.BUILD_WIKI,
                user_id=user_id,
                knowledge_base_id=command.knowledge_base_id,
                payload={"run_id": str(run_id)},
                idempotency_key=key,
            )
            async with self._pool.acquire() as conn, conn.transaction():
                job = await self._create_job_in_transaction(
                    conn,
                    job_command,
                    authenticated_user_id=user_id,
                )
                return await repository.create_root(
                    conn,
                    run_id=run_id,
                    job_id=job.id,
                    user_id=user_id,
                    config=command.config,
                    idempotency_key=key,
                    request_digest=command.request_digest,
                    model_profile_version=profile.version,
                )
        except BaseException as caught:  # noqa: BLE001 - sanitize database/adapter failures.
            failure = caught
        raise _mapped_failure(
            failure,
            allowed_contracts=_CREATE_RAG_ERRORS,
            allow_job_resource_not_found=True,
        ) from None

    async def get(
        self,
        run_id: UUID,
        *,
        authenticated_user_id: UUID,
    ) -> RagRunRecord | None:
        failure: BaseException | None = None
        try:
            checked_run_id = _require_uuid(run_id)
            user_id = _require_uuid(authenticated_user_id)
            async with self._pool.acquire() as conn:
                return await repository.get_for_user(conn, checked_run_id, user_id)
        except BaseException as caught:  # noqa: BLE001 - sanitize database/adapter failures.
            failure = caught
        raise _mapped_failure(failure, allowed_contracts=_READ_RAG_ERRORS) from None

    async def steps(
        self,
        run_id: UUID,
        *,
        authenticated_user_id: UUID,
        after_sequence: int,
        limit: int,
    ) -> tuple[RagStepRecord, ...] | None:
        failure: BaseException | None = None
        try:
            checked_run_id = _require_uuid(run_id)
            user_id = _require_uuid(authenticated_user_id)
            if type(after_sequence) is not int or after_sequence < 0:
                raise _error("invalid_request")
            if type(limit) is not int or not 1 <= limit <= 100:
                raise _error("invalid_request")
            async with self._pool.acquire() as conn:
                if await repository.get_for_user(conn, checked_run_id, user_id) is None:
                    return None
                return await repository.list_steps_for_user(
                    conn,
                    run_id=checked_run_id,
                    user_id=user_id,
                    after_sequence=after_sequence,
                    limit=limit,
                )
        except BaseException as caught:  # noqa: BLE001 - sanitize database/adapter failures.
            failure = caught
        raise _mapped_failure(failure, allowed_contracts=_READ_RAG_ERRORS) from None

    async def resume(
        self,
        run_id: UUID,
        *,
        authenticated_user_id: UUID,
        idempotency_key: str,
        budget_override: Mapping[str, int],
    ) -> RagRunRecord | None:
        failure: BaseException | None = None
        try:
            checked_run_id = _require_uuid(run_id)
            user_id = _require_uuid(authenticated_user_id)
            key = _require_idempotency_key(idempotency_key)
            overrides = _materialize_budget_override(budget_override)
            self._require_enabled()

            async with self._pool.acquire() as conn:
                initial_parent = await repository.get_for_user(conn, checked_run_id, user_id)
            if initial_parent is None:
                return None
            profile = self._resolve_profile(initial_parent.model_profile)
            await self._require_retrieval_profile(initial_parent.retrieval_profile)
            if profile.version != initial_parent.model_profile_version:
                raise _error("model_profile_unavailable")

            async with self._pool.acquire() as conn, conn.transaction():
                identity = await conn.fetchrow(
                    "SELECT root_run_id FROM rag_runs WHERE id=$1 AND user_id=$2",
                    checked_run_id,
                    user_id,
                )
                if identity is None:
                    return None
                await conn.fetchrow("SELECT id FROM rag_runs WHERE id=$1 FOR UPDATE", identity["root_run_id"])
                await conn.fetchrow(
                    "SELECT id FROM rag_runs WHERE id=$1 AND user_id=$2 FOR UPDATE",
                    checked_run_id,
                    user_id,
                )
                parent = await repository.get_for_user(conn, checked_run_id, user_id)
                if parent is None:  # pragma: no cover - the locked row cannot disappear.
                    return None
                if (
                    parent.model_profile != initial_parent.model_profile
                    or parent.model_profile_version != profile.version
                    or parent.retrieval_profile != initial_parent.retrieval_profile
                ):
                    raise _error("model_profile_unavailable")
                state = await conn.fetchval(
                    "SELECT state::text FROM background_jobs WHERE id=$1 AND user_id=$2",
                    parent.job_id,
                    user_id,
                )
                if state != JobState.FAILED.value:
                    raise _error("resume_not_allowed")

                budget = _resume_budget(parent.budget, overrides)
                command = ResumeRagRun(parent_run_id=parent.id, budget=budget)
                resumed_run_id = uuid4()
                job = await self._create_job_in_transaction(
                    conn,
                    JobCreate(
                        job_type=JobType.BUILD_WIKI,
                        user_id=user_id,
                        knowledge_base_id=parent.knowledge_base_id,
                        payload={"run_id": str(resumed_run_id)},
                        idempotency_key=key,
                    ),
                    authenticated_user_id=user_id,
                )
                return await repository.create_resume(
                    conn,
                    run_id=resumed_run_id,
                    job_id=job.id,
                    parent=parent,
                    budget=command.budget,
                    idempotency_key=key,
                    request_digest=command.request_digest,
                )
        except BaseException as caught:  # noqa: BLE001 - sanitize database/adapter failures.
            failure = caught
        raise _mapped_failure(
            failure,
            allowed_contracts=_RESUME_RAG_ERRORS,
            allow_job_resource_not_found=True,
        ) from None

    async def _create_job_in_transaction(
        self,
        conn: asyncpg.Connection,
        command: JobCreate,
        *,
        authenticated_user_id: UUID,
    ) -> JobRecord:
        failure: BaseException | None = None
        try:
            return await self._job_service.create_in_transaction(
                conn,
                command,
                authenticated_user_id=authenticated_user_id,
            )
        except BaseException as caught:  # noqa: BLE001 - JobService never owns RAG domain errors.
            failure = caught
        raise _mapped_failure(
            failure,
            allowed_contracts=_NO_RAG_ERRORS,
            allow_job_resource_not_found=True,
        ) from None


__all__ = ["CreateRagRun", "RagService", "ResumeRagRun"]
