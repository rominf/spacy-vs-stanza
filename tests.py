from collections.abc import Mapping
from pathlib import Path

from more_itertools import ilen
from packaging.version import Version

from eval import (
    get_spacy_eval_args,
    install_spacy_model,
    treebanks_for_language,
    uninstall_spacy_model,
)
from lib import SpacyModel


def is_installed(package_name, target: Path) -> bool:
    return ilen(target.rglob(f"**/{package_name}*.dist-info")) == 1


def test_install_uninstall_spacy_model(tmp_path: Path) -> None:
    model = SpacyModel(
        name="da_core_news_sm",
        language="da",
        version=Version("3.7.0"),
        url="https://github.com/explosion/spacy-models/releases/download/da_core_news_sm-3.7.0/da_core_news_sm-3.7.0-py3-none-any.whl",
    )
    try:
        install_spacy_model(model, target=tmp_path)
        assert is_installed(model.name, target=tmp_path)
    finally:
        uninstall_spacy_model(model, target=tmp_path)
        assert not is_installed(model.name, target=tmp_path)


def test_ud(ud: Path) -> None:
    assert ilen(ud.rglob("UD_*/*test.txt")) > 0
    assert ilen(ud.rglob("UD_*/*test.conllu")) > 0


def test_treebanks_for_language(ud: Path) -> None:
    assert ilen(treebanks_for_language(language="en", ud=ud)) > 0
    assert ilen(treebanks_for_language(language="xx", ud=ud)) == sum(
        1 for _ in ud.iterdir()
    )


def test_get_spacy_models(spacy_models: Mapping[str, SpacyModel]) -> None:
    assert ilen(spacy_models) > 0


def test_spacy_eval_args(spacy_models: Mapping[str, SpacyModel], ud: Path) -> None:
    assert ilen(get_spacy_eval_args(spacy_models=spacy_models, ud_path=ud)) > 0
