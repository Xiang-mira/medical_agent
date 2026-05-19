def test_import_cli():
    import cli_anything.medai.medai_cli as cli
    assert hasattr(cli, "main")
