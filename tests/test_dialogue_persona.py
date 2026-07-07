from imperator.dialogue import persona


class _Cfg:
    dialogue_num_predict = 512


def test_register_bands():
    assert persona.register_for_salience(0.1) == "terse"
    assert persona.register_for_salience(0.4) == "measured"
    assert persona.register_for_salience(0.7) == "present"
    assert persona.register_for_salience(0.95) == "urgent"


def test_register_clamps_out_of_range():
    assert persona.register_for_salience(-1.0) == "terse"
    assert persona.register_for_salience(2.0) == "urgent"


def test_num_predict_scales():
    assert persona.num_predict_for_register(
        "terse", _Cfg()
    ) < persona.num_predict_for_register("urgent", _Cfg())


def test_system_prompt_contains_axiom_and_context():
    p = persona.build_system_prompt("measured", "CTX", _Cfg())
    assert "center of its existence" in p
    assert "CTX" in p
    assert "JSON" in p  # the output contract is stated
