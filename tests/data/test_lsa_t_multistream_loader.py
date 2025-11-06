import pytest


pytest.importorskip("torch")

from slt.data.lsa_t_multistream import _load_csv_with_auto_delimiter


def test_auto_delimiter_handles_semicolon_with_commas(tmp_path):
    pd = pytest.importorskip("pandas")
    csv_path = tmp_path / "clips.csv"
    csv_path.write_text(
        "video_id;texto;split\nvid1;hola;train,val\n",
        encoding="utf-8",
    )

    df = _load_csv_with_auto_delimiter(pd, str(csv_path))

    assert list(df.columns) == ["video_id", "texto", "split"]
    assert df.loc[0, "split"] == "train,val"
