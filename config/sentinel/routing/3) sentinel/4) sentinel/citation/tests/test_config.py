import pathlib, yaml

def test_defaults_yaml_has_expected_values():
    cfg = yaml.safe_load(pathlib.Path('config/defaults.yaml').read_text())
    assert cfg['moe_top_k'] == 4
    assert abs(cfg['tau'] - 0.92) < 1e-9
    assert cfg['early_exit'] is True
    assert cfg['precision'] == 'FP4+adaptive'
    assert cfg['token_thinning'] == 'v3.4'
