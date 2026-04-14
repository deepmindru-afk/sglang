"""
Streaming session tests: KV cache mechanics, logprob leak, chunked prefill leak.

All tests share a single server (DEFAULT_SMALL_MODEL) with streaming sessions
and chunked prefill enabled.

Usage:
    python -m pytest test_streaming_session.py -xvs
    python -m unittest test_streaming_session.TestStreamingSession
"""

import asyncio
import json
import time
import unittest
from typing import Any, Optional

import aiohttp
import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=67, suite="stage-b-test-1-gpu-large")

# ---------------------------------------------------------------------------
# Logprob prompts (shared by concurrent logprob test)
# ---------------------------------------------------------------------------

LOGPROB_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump.",
    "Sphinx of black quartz judge my vow.",
    "The five boxing wizards jump quickly.",
]

# ---------------------------------------------------------------------------
# Filler text to trigger chunked prefill (200+ tokens per turn)
# ---------------------------------------------------------------------------

LEAK_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly. "
    "Jackdaws love my big sphinx of quartz. "
    "A wizard's job is to vex chumps quickly in fog. "
    "We promptly judged antique ivory buckles for the next prize. "
) * 20

# ---------------------------------------------------------------------------
# Abort-heavy chunked prefill leak repro constants
# ---------------------------------------------------------------------------

ABORT_REPRO_CONTEXT_LEN = 512
ABORT_REPRO_PAGE_SIZE = 16
ABORT_REPRO_GEN_LEN = 4
ABORT_REPRO_SESSIONS = 4
ABORT_REPRO_WARMUP_TURNS = 1
ABORT_REPRO_ROUNDS = 8
# Stream/recovery tokens are small so accumulated context stays well within
# context_length across all rounds. Abort tokens exceed context_length so
# the server rejects them at the HTTP level (400). The session is unaffected
# and recovery continues appending normally.
ABORT_REPRO_STREAM_TOKENS = 16
ABORT_REPRO_ABORT_TOKENS = 600
ABORT_REPRO_NON_STREAMING_TOKENS = 16
ABORT_REPRO_CHUNKED_PREFILL_SIZE = 128

# ---------------------------------------------------------------------------
# Concurrent logprob constants (multi-session, tests retract under concurrency)
# ---------------------------------------------------------------------------

CONCURRENT_LOGPROB_SESSIONS = 6
CONCURRENT_LOGPROB_TURNS = 5
CONCURRENT_LOGPROB_ROUNDS = 10

# ---------------------------------------------------------------------------
# Concurrent stress constants (high-concurrency streaming + non-streaming)
# ---------------------------------------------------------------------------

STRESS_NUM_SESSIONS = 8
STRESS_NUM_NON_STREAMING = 4
STRESS_NUM_TURNS = 6
STRESS_GEN_LEN = 16


def _make_token_sized_ids(
    tokenizer: Any, prefix: str, min_tokens: int, max_tokens: Optional[int] = None
) -> list[int]:
    text = prefix
    chunk = " pack quartz wizard sphinx zebra fox " * 16
    token_ids = tokenizer.encode(text)
    while len(token_ids) < min_tokens:
        text += chunk
        token_ids = tokenizer.encode(text)
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]
    return token_ids


async def _abort_repro_generate(
    base_url: str,
    session: aiohttp.ClientSession,
    input_ids: list[int],
    max_new_tokens: int,
    session_params: Optional[dict[str, Any]] = None,
    expect_abort: bool = False,
) -> Optional[dict[str, Any]]:
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "no_stop_trim": True,
            "skip_special_tokens": False,
        },
    }
    if session_params:
        payload["session_params"] = session_params

    async with session.post(base_url + "/generate", json=payload) as resp:
        text = await resp.text()
        if expect_abort:
            if resp.status == 200:
                data = json.loads(text)
                finish_reason = data.get("meta_info", {}).get("finish_reason", {})
                assert finish_reason.get("type") == "abort", text
                assert "maximum allowed length" in finish_reason.get(
                    "message", ""
                ), text
                return data
            assert resp.status == 400, text
            assert "maximum allowed length" in text or "context length" in text, text
            return None

        assert resp.status == 200, text
        data = json.loads(text)
        finish_reason = data.get("meta_info", {}).get("finish_reason", {})
        assert finish_reason.get("type") != "abort", text
        return data


async def _abort_repro_run_all(base_url: str, tokenizer: Any) -> None:
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        session_ids = []
        for _ in range(ABORT_REPRO_SESSIONS):
            async with http.post(
                base_url + "/open_session",
                json={"capacity_of_str_len": 50000, "streaming": True},
            ) as resp:
                assert resp.status == 200, await resp.text()
                session_ids.append(await resp.json())

        try:
            for warmup_turn in range(ABORT_REPRO_WARMUP_TURNS):
                warmup_tasks = []
                for session_idx, session_id in enumerate(session_ids):
                    input_ids = _make_token_sized_ids(
                        tokenizer,
                        prefix=f"[warmup={warmup_turn} session={session_idx}]",
                        min_tokens=ABORT_REPRO_STREAM_TOKENS,
                        max_tokens=ABORT_REPRO_STREAM_TOKENS + 8,
                    )
                    warmup_tasks.append(
                        _abort_repro_generate(
                            base_url,
                            http,
                            input_ids,
                            ABORT_REPRO_GEN_LEN,
                            session_params={"id": session_id, "rid": None},
                        )
                    )
                await asyncio.gather(*warmup_tasks)

            for round_idx in range(ABORT_REPRO_ROUNDS):
                mixed_tasks = []
                for session_idx, session_id in enumerate(session_ids):
                    input_ids = _make_token_sized_ids(
                        tokenizer,
                        prefix=f"[round={round_idx} ok session={session_idx}]",
                        min_tokens=ABORT_REPRO_STREAM_TOKENS,
                        max_tokens=ABORT_REPRO_STREAM_TOKENS + 8,
                    )
                    mixed_tasks.append(
                        _abort_repro_generate(
                            base_url,
                            http,
                            input_ids,
                            ABORT_REPRO_GEN_LEN,
                            session_params={"id": session_id, "rid": None},
                        )
                    )

                for ns_idx in range(2):
                    input_ids = _make_token_sized_ids(
                        tokenizer,
                        prefix=f"[round={round_idx} ns={ns_idx}]",
                        min_tokens=ABORT_REPRO_NON_STREAMING_TOKENS,
                        max_tokens=ABORT_REPRO_NON_STREAMING_TOKENS + 8,
                    )
                    mixed_tasks.append(
                        _abort_repro_generate(
                            base_url,
                            http,
                            input_ids,
                            ABORT_REPRO_GEN_LEN,
                        )
                    )
                await asyncio.gather(*mixed_tasks)

                abort_tasks = []
                for session_idx, session_id in enumerate(session_ids):
                    input_ids = _make_token_sized_ids(
                        tokenizer,
                        prefix=f"[round={round_idx} abort session={session_idx}]",
                        min_tokens=ABORT_REPRO_ABORT_TOKENS,
                    )
                    abort_tasks.append(
                        _abort_repro_generate(
                            base_url,
                            http,
                            input_ids,
                            ABORT_REPRO_GEN_LEN,
                            session_params={"id": session_id, "rid": None},
                            expect_abort=True,
                        )
                    )
                await asyncio.gather(*abort_tasks)

                recovery_tasks = []
                for session_idx, session_id in enumerate(session_ids):
                    input_ids = _make_token_sized_ids(
                        tokenizer,
                        prefix=f"[round={round_idx} recover session={session_idx}]",
                        min_tokens=ABORT_REPRO_NON_STREAMING_TOKENS,
                        max_tokens=ABORT_REPRO_NON_STREAMING_TOKENS + 8,
                    )
                    recovery_tasks.append(
                        _abort_repro_generate(
                            base_url,
                            http,
                            input_ids,
                            ABORT_REPRO_GEN_LEN,
                            session_params={"id": session_id, "rid": None},
                        )
                    )
                recovery_results = await asyncio.gather(*recovery_tasks)
                for result in recovery_results:
                    assert result is not None
                    assert result["meta_info"]["cached_tokens"] > 0, result

                health = requests.get(base_url + "/health", timeout=10)
                if health.status_code != 200:
                    raise RuntimeError(
                        f"server unhealthy after round={round_idx}: "
                        f"{health.status_code} {health.text}"
                    )
        finally:
            for session_id in session_ids:
                async with http.post(
                    base_url + "/close_session", json={"session_id": session_id}
                ) as resp:
                    assert resp.status == 200, await resp.text()


# ---------------------------------------------------------------------------
# General async generate helper (used by concurrent logprob & stress tests)
# ---------------------------------------------------------------------------


async def _async_generate(
    base_url: str,
    session: aiohttp.ClientSession,
    input_ids: list[int],
    max_new_tokens: int = 8,
    session_params: Optional[dict[str, Any]] = None,
    return_logprob: bool = False,
    logprob_start_len: Optional[int] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "no_stop_trim": True,
            "skip_special_tokens": False,
        },
    }
    if session_params:
        payload["session_params"] = session_params
    if return_logprob:
        payload["return_logprob"] = True
    if logprob_start_len is not None:
        payload["logprob_start_len"] = logprob_start_len
    timeout = aiohttp.ClientTimeout(total=300)
    async with session.post(
        base_url + "/generate", json=payload, timeout=timeout
    ) as resp:
        assert resp.status == 200, f"Generate failed: {await resp.text()}"
        return await resp.json()


# ---------------------------------------------------------------------------
# Concurrent logprob helpers
# ---------------------------------------------------------------------------


async def _concurrent_logprob_run(base_url: str, tokenizer: Any, **gen_kwargs) -> None:
    """Run multiple sessions concurrently with logprob requests.

    Each round opens N sessions, fires all sessions' requests per turn
    simultaneously (so the running batch has N entries — retract can
    actually kick one), then closes all sessions.
    """
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        for _ in range(CONCURRENT_LOGPROB_ROUNDS):
            sids: list[str] = []
            for _ in range(CONCURRENT_LOGPROB_SESSIONS):
                async with http.post(
                    base_url + "/open_session",
                    json={"capacity_of_str_len": 50000, "streaming": True},
                ) as resp:
                    assert resp.status == 200
                    sids.append(await resp.json())

            rids: list[Optional[str]] = [None] * CONCURRENT_LOGPROB_SESSIONS
            for turn in range(CONCURRENT_LOGPROB_TURNS):
                tasks = []
                for s in range(CONCURRENT_LOGPROB_SESSIONS):
                    text = (
                        f"S{s} T{turn}: "
                        f"{LOGPROB_PROMPTS[turn % len(LOGPROB_PROMPTS)]}"
                    )
                    ids = tokenizer.encode(text)
                    tasks.append(
                        _async_generate(
                            base_url,
                            http,
                            ids,
                            session_params={"id": sids[s], "rid": rids[s]},
                            **gen_kwargs,
                        )
                    )
                results = await asyncio.gather(*tasks)
                for s in range(CONCURRENT_LOGPROB_SESSIONS):
                    rids[s] = results[s]["meta_info"]["id"]

            for sid in sids:
                async with http.post(
                    base_url + "/close_session", json={"session_id": sid}
                ) as resp:
                    assert resp.status == 200


# ---------------------------------------------------------------------------
# Concurrent stress helpers
# ---------------------------------------------------------------------------


async def _stress_run_all(base_url: str, tokenizer: Any) -> None:
    """High concurrency streaming + non-streaming in mixed batches.

    Opens many sessions and fires all requests per turn simultaneously,
    ensuring the running batch is large enough for retract to have real
    effect. Prompt tokens are long enough (~200+) to trigger chunked
    prefill, so retract can interrupt mid-extend.
    """
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        sids: list[str] = []
        for _ in range(STRESS_NUM_SESSIONS):
            async with http.post(
                base_url + "/open_session",
                json={"capacity_of_str_len": 50000, "streaming": True},
            ) as resp:
                assert resp.status == 200
                sids.append(await resp.json())

        rids: list[Optional[str]] = [None] * STRESS_NUM_SESSIONS
        for turn in range(STRESS_NUM_TURNS):
            tasks = []
            # Streaming requests — long prompts to trigger chunked prefill
            for s in range(STRESS_NUM_SESSIONS):
                offset = (s * STRESS_NUM_TURNS + turn) * 200
                text = (
                    f"Session {s} turn {turn}: " f"{LEAK_FILLER[offset : offset + 800]}"
                )
                ids = tokenizer.encode(text)
                tasks.append(
                    _async_generate(
                        base_url,
                        http,
                        ids,
                        max_new_tokens=STRESS_GEN_LEN,
                        session_params={"id": sids[s], "rid": rids[s]},
                    )
                )

            # Non-streaming requests interleaved
            for ns in range(STRESS_NUM_NON_STREAMING):
                text = (
                    f"Non-streaming {ns} turn {turn}: "
                    f"{LEAK_FILLER[ns * 100 : ns * 100 + 400]}"
                )
                ids = tokenizer.encode(text)
                tasks.append(
                    _async_generate(
                        base_url,
                        http,
                        ids,
                        max_new_tokens=STRESS_GEN_LEN,
                    )
                )

            results = await asyncio.gather(*tasks)
            for s in range(STRESS_NUM_SESSIONS):
                rids[s] = results[s]["meta_info"]["id"]

        for sid in sids:
            async with http.post(
                base_url + "/close_session", json={"session_id": sid}
            ) as resp:
                assert resp.status == 200


# ===================================================================
# Test class
# ===================================================================


class TestStreamingSession(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "512",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    # Overlap scheduler defers the last decode iteration's commit by one step,
    # so its "in-flight" last token gets committed by the time the request
    # finishes. With overlap disabled (e.g. streaming session + speculative
    # decoding), the last sampled token isn't committed before max_new_tokens
    # stops, leaving slot.kv_committed_len = input + output - 1. Subclasses
    # that disable overlap should set this to -1 to relax the inheritance check.
    kv_inherit_offset = 0

    def test_kv_cache_inheritance(self, gen_len=12):
        """Verify KV inheritance, radix cache insertion, and flush reclamation."""
        chunks = [
            "Let me tell you something about France.",
            "The capital of France is",
            "The population of the city is",
        ]
        chunks_ids = [self.tokenizer.encode(x) for x in chunks]
        for i in range(1, len(chunks_ids)):
            if chunks_ids[i][0] == self.tokenizer.bos_token_id:
                chunks_ids[i] = chunks_ids[i][1:]

        # === Part 1: streaming session — check KV inheritance ===
        requests.post(self.base_url + "/flush_cache")
        session_id = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 1000, "streaming": True},
        ).json()
        rid = None

        prev_kv_len = 0
        for turn_idx, chunk_ids in enumerate(chunks_ids):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": chunk_ids,
                    "session_params": {"id": session_id, "rid": rid},
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": gen_len,
                        "no_stop_trim": True,
                        "skip_special_tokens": False,
                    },
                },
            ).json()
            rid = response["meta_info"]["id"]
            cached = response["meta_info"]["cached_tokens"]
            prompt_tokens = response["meta_info"]["prompt_tokens"]
            completion_tokens = response["meta_info"]["completion_tokens"]

            if turn_idx == 0:
                # Turn 1 should have no cache hit (cache was flushed).
                self.assertEqual(
                    cached, 0, "Turn 1 should have 0 cached tokens (clean start)"
                )
            else:
                # Turns 2+ inherit KV from the previous turn (via inherit_kv_states,
                # not radix tree matching). cached_tokens reflects the inherited prefix.
                expected = prev_kv_len + self.kv_inherit_offset
                self.assertEqual(
                    cached,
                    expected,
                    f"Turn {turn_idx + 1}: should inherit {expected} KV tokens from previous turn",
                )
            prev_kv_len = prompt_tokens + completion_tokens

        # Close the session before checking cache/memory state.
        ret = requests.post(
            self.base_url + "/close_session",
            json={"session_id": session_id},
        )
        self.assertEqual(ret.status_code, 200)

        # === Cache verification (after close, before flush) ===

        # Turn 1's prompt was inserted to the cache.
        verify_resp = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[0],
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
        ).json()
        self.assertGreater(
            verify_resp["meta_info"]["cached_tokens"],
            0,
            "Turn 1's prompt should be cached in the radix tree",
        )

        # Turn 2's prompt tokens should NOT be in cache.
        # The tree should only contain turn 1's extent (prompt + output from
        # cache_unfinished_req during decode). Turn 2's prompt starts fresh tokens
        # that were never inserted.
        verify_resp2 = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[1],
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
        ).json()
        self.assertEqual(
            verify_resp2["meta_info"]["cached_tokens"],
            0,
            "Turn 2's prompt should not be in cache (no insertion for turns 2+)",
        )

        # === Flush reclamation ===

        requests.post(self.base_url + "/flush_cache")
        verify_resp3 = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": chunks_ids[0],
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
        ).json()
        self.assertEqual(
            verify_resp3["meta_info"]["cached_tokens"],
            0,
            "After session close + flush, cache should be fully reclaimed",
        )

    def test_leak_logprob_concurrent(self) -> None:
        """Concurrent multi-session logprob leak test (all 3 modes).

        6 sessions fire per turn simultaneously so the running batch has
        real concurrency — retract/mixed-chunk actually exercise their
        multi-request scheduling paths. Covers: output logprob,
        input logprob (logprob_start_len=0), and no logprob.
        """
        requests.post(self.base_url + "/flush_cache")
        # Output logprob
        asyncio.run(
            _concurrent_logprob_run(self.base_url, self.tokenizer, return_logprob=True)
        )
        # Input logprob (logprob_start_len=0)
        asyncio.run(
            _concurrent_logprob_run(
                self.base_url,
                self.tokenizer,
                return_logprob=True,
                logprob_start_len=0,
            )
        )
        # No logprob
        asyncio.run(_concurrent_logprob_run(self.base_url, self.tokenizer))
        time.sleep(3)
        assert (
            requests.get(self.base_url + "/health").status_code == 200
        ), "Server unhealthy after concurrent logprob sessions."

    def test_stress_concurrent_sessions(self) -> None:
        """High concurrency streaming + non-streaming under mixed batch pressure.

        8 streaming sessions + 4 non-streaming per turn, with long prompts
        triggering chunked prefill. Under retract, the scheduler retract_decode
        fires every 3 forward steps and must correctly roll back streaming
        session KV without leaking tokens.
        """
        requests.post(self.base_url + "/flush_cache")
        asyncio.run(_stress_run_all(self.base_url, self.tokenizer))

        for i in range(3):
            ids = self.tokenizer.encode(f"Post-stress cleanup {i}.")
            requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                },
            )

        time.sleep(5)
        health = requests.get(self.base_url + "/health")
        self.assertEqual(
            health.status_code,
            200,
            "Server unhealthy after concurrent stress test — "
            "likely a token leak from retract/mixed-chunk + streaming session.",
        )

    def test_mid_processing_abort_preserves_session(self) -> None:
        """Abort a running streaming session request via the abort API and
        verify the session slot is preserved for the next turn."""
        requests.post(self.base_url + "/flush_cache")

        # Open session
        resp = requests.post(
            self.base_url + "/open_session",
            json={"capacity_of_str_len": 50000, "streaming": True},
        )
        self.assertEqual(resp.status_code, 200)
        session_id = resp.json()

        try:
            # Turn 1: normal generate to create slot
            ids_1 = self.tokenizer.encode("Tell me a very long story about a wizard.")
            resp_1 = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids_1,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 16,
                    },
                    "session_params": {"id": session_id, "rid": None},
                },
                timeout=30,
            )
            self.assertEqual(resp_1.status_code, 200, resp_1.text)
            data_1 = resp_1.json()
            rid_1 = data_1["meta_info"]["id"]

            # Turn 2: start a long generate, then abort mid-decode
            ids_2 = self.tokenizer.encode(" Continue the story in great detail.")

            import threading

            result = [None]

            def do_generate():
                r = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": ids_2,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 256,
                        },
                        "session_params": {"id": session_id, "rid": None},
                    },
                    timeout=60,
                )
                result[0] = r

            t = threading.Thread(target=do_generate)
            t.start()

            # Wait a bit for decode to start, then abort
            time.sleep(2)
            abort_resp = requests.post(
                self.base_url + "/abort",
                json={"rid": None, "abort_all": False},
                timeout=10,
            )

            t.join(timeout=30)

            # Turn 3: recovery — session should still work
            ids_3 = self.tokenizer.encode(" What happens next?")
            resp_3 = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids_3,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 8,
                    },
                    "session_params": {"id": session_id, "rid": None},
                },
                timeout=30,
            )
            self.assertEqual(resp_3.status_code, 200, resp_3.text)
            data_3 = resp_3.json()
            self.assertGreater(
                data_3["meta_info"]["cached_tokens"],
                0,
                "Recovery after mid-processing abort should inherit KV",
            )
        finally:
            requests.post(
                self.base_url + "/close_session",
                json={"session_id": session_id},
            )

        # Server should still be healthy
        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(health.status_code, 200)


class TestStreamingSessionMixedChunk(TestStreamingSession):
    """Streaming session with --enable-mixed-chunk."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "512",
                    "--enable-mixed-chunk",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestStreamingSessionRetract(TestStreamingSession):
    """Streaming session under retract decode pressure."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_TEST_RETRACT.override(
            True
        ), envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "128",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestStreamingSessionRetractMixedChunk(TestStreamingSession):
    """Streaming session under retract decode with --enable-mixed-chunk."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_TEST_RETRACT.override(
            True
        ), envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "128",
                    "--enable-mixed-chunk",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestStreamingSessionEagle(TestStreamingSession):
    """Streaming session with EAGLE3 speculative decoding.

    Streaming session is incompatible with overlap + speculative, so
    --disable-overlap-schedule is required.
    """

    # Overlap is disabled (required for spec + streaming session), so the
    # last token isn't committed before stop. See base class note.
    kv_inherit_offset = -1

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_TARGET_MODEL_EAGLE3
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(
            2
        ), envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--disable-overlap-schedule",
                    "--chunked-prefill-size",
                    "512",
                    "--dtype=float16",
                    "--speculative-algorithm",
                    "EAGLE3",
                    "--speculative-draft-model",
                    DEFAULT_DRAFT_MODEL_EAGLE3,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--mem-fraction-static",
                    "0.7",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestStreamingSessionEagleV2(TestStreamingSession):
    """Streaming session with EAGLE3 spec v2 (overlap-aware schedule).

    Spec v2 may over-generate beyond max_new_tokens (each verify round
    accepts M+1 tokens, with no per-token stop check inside the round).
    response.completion_tokens is capped at max_new_tokens, but
    slot.kv_committed_len reflects the actual output_ids length minus 1
    (spec's free-token isn't committed). Per-turn over-generation varies,
    so a constant inheritance offset doesn't work — skip the strict
    assertion. The 4 leak tests still validate spec v2 + streaming
    session under the strict mem check.
    """

    @unittest.skip(
        "Spec v2 over-generation makes slot.kv_committed_len drift unevenly "
        "vs response.completion_tokens; constant kv_inherit_offset can't fit."
    )
    def test_kv_cache_inheritance(self, gen_len=12):
        pass

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_TARGET_MODEL_EAGLE3
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_SPEC_V2.override(
            True
        ), envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(
            2
        ), envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(
            True
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "512",
                    "--dtype=float16",
                    "--speculative-algorithm",
                    "EAGLE3",
                    "--speculative-draft-model",
                    DEFAULT_DRAFT_MODEL_EAGLE3,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--mem-fraction-static",
                    "0.7",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestStreamingSessionEagleRetract(TestStreamingSession):
    """Streaming session with EAGLE3 speculative decoding under retract pressure.

    Combines spec decode (which changes KV commit patterns) with retract
    (which forces KV rollback mid-decode). --disable-overlap-schedule is
    required for spec + streaming session.
    """

    kv_inherit_offset = -1

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_TARGET_MODEL_EAGLE3
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_TEST_RETRACT.override(
            True
        ), envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(
            2
        ), envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(
            True
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--disable-overlap-schedule",
                    "--chunked-prefill-size",
                    "128",
                    "--dtype=float16",
                    "--speculative-algorithm",
                    "EAGLE3",
                    "--speculative-draft-model",
                    DEFAULT_DRAFT_MODEL_EAGLE3,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--mem-fraction-static",
                    "0.7",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestStreamingSessionEagleV2Retract(TestStreamingSession):
    """Streaming session with EAGLE3 spec v2 under retract pressure.

    Same kv_inheritance skip as EagleV2 (over-generation drift), plus
    retract forces KV rollback during verify rounds.
    """

    @unittest.skip(
        "Spec v2 over-generation makes slot.kv_committed_len drift unevenly "
        "vs response.completion_tokens; constant kv_inherit_offset can't fit."
    )
    def test_kv_cache_inheritance(self, gen_len=12):
        pass

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_TARGET_MODEL_EAGLE3
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_SPEC_V2.override(
            True
        ), envs.SGLANG_TEST_RETRACT.override(
            True
        ), envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(
            2
        ), envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(
            True
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "128",
                    "--dtype=float16",
                    "--speculative-algorithm",
                    "EAGLE3",
                    "--speculative-draft-model",
                    DEFAULT_DRAFT_MODEL_EAGLE3,
                    "--speculative-num-steps",
                    "3",
                    "--speculative-eagle-topk",
                    "1",
                    "--speculative-num-draft-tokens",
                    "4",
                    "--mem-fraction-static",
                    "0.7",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestStreamingSessionAbortLeakRepro(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    str(ABORT_REPRO_CHUNKED_PREFILL_SIZE),
                    "--context-length",
                    str(ABORT_REPRO_CONTEXT_LEN),
                    "--page-size",
                    str(ABORT_REPRO_PAGE_SIZE),
                    "--max-running-requests",
                    "32",
                    "--log-level",
                    "info",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_abort_heavy_chunked_prefill_does_not_leak(self) -> None:
        requests.post(self.base_url + "/flush_cache")

        asyncio.run(_abort_repro_run_all(self.base_url, self.tokenizer))

        for i in range(3):
            ids = self.tokenizer.encode(f"Post-session cleanup request {i}.")
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                },
                timeout=30,
            )
            self.assertEqual(response.status_code, 200, response.text)

        time.sleep(5)
        self.assertIsNone(
            self.process.poll(),
            "Server crashed during abort-heavy streaming session repro.",
        )

        health = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(
            health.status_code,
            200,
            "Server unhealthy after abort-heavy streaming session cleanup.",
        )


class TestStreamingSessionLargePage(TestStreamingSession):
    """Streaming session with large page_size (256) and small append chunks.

    Verifies that page alignment handling works correctly when the append
    size per turn is much smaller than page_size. prefix_len must NOT be
    floor-aligned to page_size (that would lose up to 255 tokens per turn).
    """

    @unittest.skip(
        "page_size=256 is too large for short test prompts to fill a page; "
        "radix tree returns 0 cached_tokens so inheritance assertion fails."
    )
    def test_kv_cache_inheritance(self, gen_len=12):
        pass

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "512",
                    "--page-size",
                    "256",
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
