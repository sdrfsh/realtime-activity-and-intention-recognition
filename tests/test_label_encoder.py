from modeling.label_encoder import LabelEncoder


def test_encode_is_stable_and_zero_indexed():
    encoder = LabelEncoder(["run", "walk", "run", "jump"])
    assert encoder.num_classes == 3
    indices = {encoder.encode("jump"), encoder.encode("run"), encoder.encode("walk")}
    assert indices == {0, 1, 2}


def test_encode_all_maps_each_label():
    encoder = LabelEncoder(["run", "walk"])
    assert encoder.encode_all(["walk", "run", "walk"]) == [
        encoder.encode("walk"),
        encoder.encode("run"),
        encoder.encode("walk"),
    ]


def test_decode_reverses_encode():
    encoder = LabelEncoder(["entering", "passing_by", "entering"])
    for label in ("entering", "passing_by"):
        assert encoder.decode(encoder.encode(label)) == label


def test_classes_are_ordered_by_index():
    encoder = LabelEncoder(["passing_by", "entering", "entering"])
    classes = encoder.classes
    assert classes == sorted(classes)
    assert [encoder.encode(label) for label in classes] == list(range(len(classes)))
