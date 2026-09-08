import unittest
from unittest.mock import patch

from tests.test_telegram_missing_coverage import _bot_context


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class TestTelegramPollResilience(unittest.TestCase):
    def test_compute_poll_backoff_seconds_dns_multiplier_and_cap(self):
        with _bot_context() as (bot, _stubs):
            with patch.object(bot.random, "uniform", return_value=0.0):
                # non-DNS: 5, 10, 20...
                self.assertEqual(bot._compute_poll_backoff_seconds(1, dns_failure=False), 5.0)
                self.assertEqual(bot._compute_poll_backoff_seconds(2, dns_failure=False), 10.0)
                # DNS failures apply multiplier (default x2)
                self.assertEqual(bot._compute_poll_backoff_seconds(1, dns_failure=True), 10.0)
                self.assertEqual(bot._compute_poll_backoff_seconds(2, dns_failure=True), 20.0)
                # Capped at configured max (default 60)
                self.assertEqual(bot._compute_poll_backoff_seconds(10, dns_failure=True), 60.0)

    def test_poll_retries_after_dns_error_then_recovers(self):
        with _bot_context() as (bot, stubs):
            bot._POLL_FAILURES_TOTAL = 0
            bot._POLL_DNS_FAILURES_TOTAL = 0
            bot._POLL_LAST_SUCCESS_EPOCH = 0.0

            responses = [
                Exception("NameResolutionError: Failed to resolve 'api.telegram.org'"),
                _FakeResp({"result": []}),
                KeyboardInterrupt(),
            ]

            def _side_effect(*_a, **_kw):
                item = responses.pop(0)
                if isinstance(item, BaseException):
                    raise item
                return item

            stubs["requests"].get.side_effect = _side_effect

            sleep_calls = []

            def _sleep_spy(seconds):
                sleep_calls.append(seconds)

            with patch.object(bot.random, "uniform", return_value=0.0), \
                 patch.object(bot.time, "sleep", side_effect=_sleep_spy), \
                 patch("builtins.print") as print_mock:
                bot.poll()

            # one failed poll should increment counters and sleep once (DNS multiplier => 10s)
            self.assertEqual(bot._POLL_FAILURES_TOTAL, 1)
            self.assertEqual(bot._POLL_DNS_FAILURES_TOTAL, 1)
            self.assertTrue(bot._POLL_LAST_SUCCESS_EPOCH > 0)
            self.assertEqual(len(sleep_calls), 1)
            self.assertEqual(sleep_calls[0], 10.0)

            printed = "\n".join(str(c.args[0]) for c in print_mock.call_args_list if c.args)
            self.assertIn("Poll error: type=Exception dns=True", printed)
            self.assertIn("Poll recovered after 1 failure(s)", printed)


if __name__ == "__main__":
    unittest.main()
