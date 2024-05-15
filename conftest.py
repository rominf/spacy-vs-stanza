from collections.abc import Mapping
from pathlib import Path

from pytest import FixtureRequest, Parser, TempPathFactory, fixture

from eval import get_spacy_models, prepare_ud
from lib import SpacyModel


def pytest_addoption(parser: Parser):
    parser.addoption(
        "--ud-archive",
        type=Path,
    )


@fixture(scope="session")
def ud_archive(request: FixtureRequest) -> Path | None:
    return request.config.getoption("--ud-archive")


@fixture(scope="session")
def ud_path(tmp_path_factory: TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("ud")


@fixture(scope="session")
def ud(ud_path: Path, ud_archive: Path | None) -> Path:
    if ud_archive is not None:
        (ud_path / ud_archive.name).symlink_to(ud_archive.absolute())
    return prepare_ud(target=ud_path)


@fixture(scope="session")
def spacy_models() -> Mapping[str, SpacyModel]:
    return get_spacy_models()
