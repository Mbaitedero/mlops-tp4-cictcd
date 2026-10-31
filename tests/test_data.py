from src.data import load_data


def test_no_missing_values():
    df = load_data()
    assert df.isnull().sum().sum() == 0


def test_labels_are_binary():
    df = load_data()
    assert set(df["label"].unique()).issubset({0, 1})


def test_text_not_empty():
    df = load_data()
    assert (df["text"].str.strip() != "").all()


def test_class_balance_is_reasonable():
    df = load_data()
    counts = df["label"].value_counts()
    assert len(counts) == 2
    assert counts.min() >= 5
