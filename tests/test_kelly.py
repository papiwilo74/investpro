"""Tests for KellyCalculator (both JSON and DB modes)."""
from bot.strategy import KellyCalculator


class TestKellyCalculatorJson:
    def test_default_win_rate_zero(self, tmp_path):
        fp = str(tmp_path / "kelly_empty.json")
        k = KellyCalculator(file_path=fp)
        assert k.win_rate == 0.0

    def test_default_kelly_pct_positive(self, tmp_path):
        fp = str(tmp_path / "kelly_empty.json")
        k = KellyCalculator(file_path=fp)
        assert k.kelly_pct > 0

    def test_record_trade(self, tmp_path):
        fp = str(tmp_path / "kelly_test.json")
        k = KellyCalculator(file_path=fp)
        assert len(k.trades) == 0
        k.record(0.05)
        assert len(k.trades) == 1

    def test_win_rate_after_trades(self, tmp_path):
        fp = str(tmp_path / "kelly_test.json")
        k = KellyCalculator(file_path=fp)
        k.record(0.05)
        k.record(0.03)
        k.record(-0.02)
        assert k.win_rate == 2 / 3

    def test_reset_clears_trades(self, tmp_path):
        fp = str(tmp_path / "kelly_test.json")
        k = KellyCalculator(file_path=fp)
        k.record(0.05)
        assert len(k.trades) == 1
        k.reset()
        assert len(k.trades) == 0

    def test_persistence_across_instances(self, tmp_path):
        fp = str(tmp_path / "kelly_persist.json")
        k1 = KellyCalculator(file_path=fp)
        k1.record(0.10)
        k1.record(0.05)
        k2 = KellyCalculator(file_path=fp)
        assert len(k2.trades) == 2

    def test_to_dict_fields(self, tmp_path):
        fp = str(tmp_path / "kelly_test.json")
        k = KellyCalculator(file_path=fp)
        k.record(0.05)
        k.record(-0.02)
        d = k.to_dict()
        assert "win_rate" in d
        assert "kelly_pct" in d
        assert "total_trades" in d
        assert d["total_trades"] == 2


class TestKellyCalculatorDB:
    def test_db_mode_uses_repo(self, _clean_db):
        k = KellyCalculator(session=_clean_db)
        assert k._use_db is True
        assert k._repo is not None

    def test_db_record_trade(self, _clean_db):
        k = KellyCalculator(session=_clean_db)
        k.record(0.05)
        k.record(-0.02)
        assert len(k.trades) == 2

    def test_db_win_rate(self, _clean_db):
        k = KellyCalculator(session=_clean_db)
        k.record(0.05)
        k.record(0.03)
        k.record(-0.02)
        assert k.win_rate == 2 / 3

    def test_db_reset(self, _clean_db):
        k = KellyCalculator(session=_clean_db)
        k.record(0.05)
        assert len(k.trades) == 1
        k.reset()
        assert len(k.trades) == 0

    def test_db_persistence(self, _clean_db):
        k = KellyCalculator(session=_clean_db)
        k.record(0.10)
        k.record(0.05)
        k2 = KellyCalculator(session=_clean_db)
        assert len(k2.trades) == 2
