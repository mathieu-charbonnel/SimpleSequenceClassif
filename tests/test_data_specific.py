import pandas as pd
import pytest

from modules.data_specific import cleaning


class TestCleaning:
    def test_removes_allele_column(self) -> None:
        df = pd.DataFrame({"allele": ["HLA-A*02:01"], "hit": [1]})
        cleaning(df)
        assert "allele" not in df.columns

    def test_extracts_class(self) -> None:
        df = pd.DataFrame({"allele": ["HLA-A*02:01"], "hit": [1]})
        cleaning(df)
        assert df["class"].iloc[0] == "A"

    def test_extracts_gene(self) -> None:
        df = pd.DataFrame({"allele": ["HLA-A*02:01"], "hit": [1]})
        cleaning(df)
        assert df["gene"].iloc[0] == "02"

    def test_extracts_variant(self) -> None:
        df = pd.DataFrame({"allele": ["HLA-A*02:01"], "hit": [1]})
        cleaning(df)
        assert df["variant"].iloc[0] == "01"

    def test_removes_asterisk(self) -> None:
        df = pd.DataFrame({"allele": ["HLA-A*02:01"], "hit": [1]})
        cleaning(df)
        for col in ["class", "gene", "variant"]:
            assert "*" not in str(df[col].iloc[0])

    def test_multiple_rows(self) -> None:
        df = pd.DataFrame({
            "allele": ["HLA-A*02:01", "HLA-B*07:02"],
            "hit": [1, 0],
        })
        cleaning(df)
        assert len(df) == 2
        assert df["class"].iloc[1] == "B"
        assert df["gene"].iloc[1] == "07"

    def test_modifies_in_place(self) -> None:
        df = pd.DataFrame({"allele": ["HLA-A*02:01"], "hit": [1]})
        result = cleaning(df)
        assert result is None
        assert "class" in df.columns
