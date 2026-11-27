"""Regression tests for full-failure-mode coverage."""
from sandbox.verifier import VerifyConfig, Verifier
import sandbox.verifier.verifier as verifier_module


def test_all_failures_runs_later_functions(monkeypatch, tmp_path):
    seen = []

    def fails():
        seen.append('fails')
        raise ValueError('intentional failure')

    def succeeds():
        seen.append('succeeds')

    monkeypatch.setattr(verifier_module, 'get_funcs_by_label',
                        lambda name: [(fails, 'default'), (succeeds, 'default')])
    monkeypatch.setattr(verifier_module, 'get_params', lambda name, mark: [])
    verifier = Verifier(VerifyConfig(run_name='test', run_dir=str(tmp_path), anti_hack=False))
    result = verifier.run_tests('example', max_failures='all', strict_check=True)
    assert seen == ['fails', 'succeeds']
    assert result.info == {'total': 2, 'failed': 1, 'success': 1}
    assert result.success is False


def test_finite_failure_limit_stops_later_functions(monkeypatch, tmp_path):
    seen = []

    def fails():
        seen.append('fails')
        raise ValueError('intentional failure')

    def succeeds():
        seen.append('succeeds')

    monkeypatch.setattr(verifier_module, 'get_funcs_by_label',
                        lambda name: [(fails, 'default'), (succeeds, 'default')])
    monkeypatch.setattr(verifier_module, 'get_params', lambda name, mark: [])
    verifier = Verifier(VerifyConfig(run_name='test', run_dir=str(tmp_path), anti_hack=False))
    result = verifier.run_tests('example', max_failures=1, strict_check=True)
    assert seen == ['fails']
    assert result.info == {'total': 1, 'failed': 1, 'success': 0}
