from experiments.verdict_parser import explicit_security_verdict


def test_last_explicit_verdict():
    assert explicit_security_verdict("The code is vulnerable. Finally: no vulnerability exists.") is False
    assert explicit_security_verdict("Not vulnerable? Actually, a vulnerability exists.") is True


def test_no_imputation_from_ambiguous_language():
    assert explicit_security_verdict(None) is None
    assert explicit_security_verdict("This might be unsafe or exploitable.") is None
