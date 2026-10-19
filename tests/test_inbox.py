from __future__ import annotations

import time
from functools import partial

import pytest

import unikin


def test_messages_are_read_oldest_first_and_capped_per_cycle(config):
    for index in range(unikin.MAX_INBOX_PER_CYCLE + 2):
        unikin.post_message(config, f"Nachricht {index}", "Test")

    inbox = unikin.read_inbox(config)

    assert [message["text"] for _, message in inbox] == [
        f"Nachricht {index}" for index in range(unikin.MAX_INBOX_PER_CYCLE)
    ]


def test_overlong_message_is_rejected(config):
    with pytest.raises(ValueError):
        unikin.post_message(config, "x" * (unikin.MAX_MESSAGE_CHARS + 1), "Test")


def test_unreadable_message_is_set_aside(config):
    directory = unikin.inbox_dir(config)
    directory.mkdir(parents=True)
    (directory / "kaputt.json").write_text("{kein json", encoding="utf-8")
    good = unikin.post_message(config, "lesbar", "Test")

    assert [message["id"] for _, message in unikin.read_inbox(config)] == [good["id"]]
    assert (directory / "kaputt.rejected").exists()


def test_cycle_delivers_message_and_records_reply(config, fake_model):
    fake_model.payload["reply"] = "Hallo zurück"
    message = unikin.post_message(config, "Wie geht es dir?", "Test")
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, unikin.read_inbox(config))

    assert [m["text"] for m in fake_model.user_payload()["inbox"]] == ["Wie geht es dir?"]
    entry = unikin.load_memory(config)["journal"][-1]
    assert entry["inbox_ids"] == [message["id"]]
    assert entry["inbox"][0]["sender"] == "Test"
    assert entry["reply"] == "Hallo zurück"
    assert not list(unikin.inbox_dir(config).glob("*.json"))


def test_message_stays_in_inbox_when_model_is_unreachable(config, fake_model):
    fake_model.error = ConnectionError("LM Studio nicht erreichbar")
    message = unikin.post_message(config, "Bist du da?", "Test")
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, unikin.read_inbox(config))

    entry = unikin.load_memory(config)["journal"][-1]
    assert "inbox_ids" not in entry
    assert [m["id"] for _, m in unikin.read_inbox(config)] == [message["id"]]


def test_trigger_without_text_is_not_shown_to_model(config, fake_model):
    trigger = unikin.post_message(config, "", "Test")
    state = unikin.load_memory(config)

    unikin.run_cycle(config, fake_model, state, unikin.read_inbox(config))

    assert fake_model.user_payload()["inbox"] == []
    entry = unikin.load_memory(config)["journal"][-1]
    assert entry["inbox_ids"] == [trigger["id"]]
    assert "inbox" not in entry


def test_wait_wakes_early_only_for_new_messages(config):
    stop = unikin.GracefulExit()
    old = unikin.post_message(config, "schon gesehen", "Test")
    wake = partial(unikin.has_new_messages, config, {old["id"]})

    started = time.monotonic()
    assert stop.wait(0.3, wake=wake) is False
    assert time.monotonic() - started >= 0.3

    unikin.post_message(config, "neu", "Test")
    started = time.monotonic()
    assert stop.wait(5.0, wake=wake, min_seconds=0.2) is False
    assert 0.2 <= time.monotonic() - started < 2.0
