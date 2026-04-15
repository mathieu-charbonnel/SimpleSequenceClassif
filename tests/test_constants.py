from modules.constants import AMINO_ACID_ALPHABET, MAX_SEQUENCE_SIZE


class TestConstants:
    def test_max_sequence_size(self) -> None:
        assert MAX_SEQUENCE_SIZE == 15

    def test_alphabet_length(self) -> None:
        assert len(AMINO_ACID_ALPHABET) == 20

    def test_alphabet_contains_standard_amino_acids(self) -> None:
        expected = {"A", "R", "N", "D", "C", "E", "Q", "G", "H", "I",
                    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"}
        assert set(AMINO_ACID_ALPHABET) == expected

    def test_alphabet_has_no_duplicates(self) -> None:
        assert len(AMINO_ACID_ALPHABET) == len(set(AMINO_ACID_ALPHABET))
