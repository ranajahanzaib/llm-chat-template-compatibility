from typer.testing import CliRunner

from templateparity.cli import app

runner = CliRunner()


def test_version_option_and_fixture_list():
    version = runner.invoke(app, ["--version"])
    fixtures = runner.invoke(app, ["fixtures", "list"])
    assert version.exit_code == 0 and version.stdout.strip() == "0.1.0"
    assert fixtures.exit_code == 0 and "simple-user" in fixtures.stdout


def test_demo_and_report_commands_write_reloadable_artifact(tmp_path):
    created = runner.invoke(app, ["demo", "--output-dir", str(tmp_path), "--json"])
    assert created.exit_code == 0, created.output
    report = runner.invoke(
        app, ["report", "synthetic-demo", "--output-dir", str(tmp_path), "--json"]
    )
    assert report.exit_code == 0, report.output
    assert '"synthetic": true' in report.stdout
