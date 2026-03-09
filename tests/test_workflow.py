from app.service import run_analysis


def test_workflow_report_shape() -> None:
    report = run_analysis("NVDA")
    assert report["ticker"] == "NVDA"
    assert "factor_score" in report
    assert "strategy" in report
    assert 0 <= report["factor_score"]["total"] <= 100
