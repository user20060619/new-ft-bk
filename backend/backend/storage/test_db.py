import json

from database import MetadataDB


def _make_db(tmp_path, filename="test_metadata.db"):
    return MetadataDB(str(tmp_path / filename))


def test_database(tmp_path):
    db = _make_db(tmp_path)

    db.add_image(
        image_id="test_001",
        date="2026-09-02",
        sensor_type="optical",
        bounding_box=json.dumps({"lat": 19.0, "lon": 72.8}),
        resolution=10.0,
    )

    result = db.get_image("test_001")
    db.close()

    assert result is not None
    assert result["image_id"] == "test_001"
    assert result["acquisition_date"] == "2026-09-02"
    assert result["sensor_type"] == "optical"
    assert result["bounding_box"] == {"lat": 19.0, "lon": 72.8}
    assert result["resolution"] == 10.0
    assert result["created_at"]


def test_database_is_fresh_across_tests(tmp_path):
    """A second test can reuse the same image_id without hitting the UNIQUE
    constraint that the old shared, committed test_metadata.db used to trigger
    on a second run."""
    db = _make_db(tmp_path)

    db.add_image(
        image_id="test_001",
        date="2026-09-02",
        sensor_type="optical",
        bounding_box=json.dumps({"lat": 19.0, "lon": 72.8}),
        resolution=10.0,
    )

    assert db.get_image("test_001") is not None
    db.close()
