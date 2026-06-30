import pytest
import datetime
from app.tools.datetime_tool import DateTimeTool

@pytest.fixture
def dt_tool():
    return DateTimeTool()

def test_datetime_availability(dt_tool):
    assert dt_tool.available() is True

def test_datetime_execution(dt_tool):
    res = dt_tool.execute("What time is it?")
    assert res["success"] is True
    assert "local_time" in res["data"]
    assert "utc_time" in res["data"]
    assert "day_of_week" in res["data"]
    assert "date" in res["data"]
    assert "formatted_date" in res["data"]

    # Verify formatting match
    assert len(res["data"]["date"]) == 10 # YYYY-MM-DD
    assert "UTC" in res["data"]["utc_time"]
    assert len(res["content"]) > 0
