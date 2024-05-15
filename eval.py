import datetime
import logging
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
import langcodes
import pandas
import pydantic
import rich
import spacy_conll
import stanza
import stanza.models.common.constant
import stanza.models.lemmatizer
import stanza.pipeline.core
import stanza.utils
import stanza.utils.datasets.common
import stanza.utils.datasets.prepare_lemma_treebank
import stanza.utils.datasets.prepare_tokenizer_treebank
import stanza.utils.training.run_lemma
import typer
from diskcache import Cache
from github import Github
from joblib import Parallel, delayed
from more_itertools import ilen, only
from stanza.models.common.utils import ud_scores

import spacy
from lib import EvalArgs, EvalResult, Library, SpacyModel, StanzaModel, Version

DECIMAL_FOUR_PLACES = Decimal(10) ** -4

UD_URL = httpx.URL(
    "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5502/ud-treebanks-v2.14.tgz"
)
UD_DIR_NAME = Path("ud-treebanks-v2.14")
UD_NAME = "ud-treebanks-v2.14.tgz"

SPACY_MODEL_VERSION_MIN = Version("3.7.0")
SPACY_MODEL_VERSION_MAX = Version("3.8.0")
SPACY_MODELS_GITHUB_REPO_URL = (
    "https://api.github.com/repos/explosion/spaCy-models/releases"
)

CACHING_SECONDS = datetime.timedelta(days=1).total_seconds()


def download_ud(url: httpx.URL = UD_URL) -> BytesIO:
    print("Downloading UD treebanks")
    result = BytesIO()
    with httpx.stream("GET", url, timeout=5 * 60) as response:
        total = int(response.headers["Content-Length"])

        with rich.progress.Progress(
                "[progress.percentage]{task.percentage:>3.0f}%",
                rich.progress.BarColumn(bar_width=None),
                rich.progress.DownloadColumn(),
                rich.progress.TransferSpeedColumn(),
        ) as progress:
            download_task = progress.add_task("Download", total=total)
            for chunk in response.iter_bytes():
                result.write(chunk)
                progress.update(download_task, completed=response.num_bytes_downloaded)
    return result


def unpack_ud(f: BytesIO, target: Path = Path.cwd(), save_tar: bool = False) -> bytes:
    print("Unpacking UD treebanks")

    target.mkdir(parents=True, exist_ok=True)

    if save_tar:
        f.seek(0)
        (target / UD_NAME).write_bytes(f.read())

    f.seek(0)
    tar = tarfile.open(fileobj=f, mode="r:gz")
    pattern_include = re.compile(r".*(test|train)\.(conllu|txt)$")
    treebank_pattern_exclude = re.compile(
        r"UD_English-GUM"  # This treebank raises exceptions on latest stanza
    )
    tar.extraction_filter = (
        lambda member, path: member
        if pattern_include.match(member.name)
           and not treebank_pattern_exclude.search(member.name)
        else None
    )
    tar.extractall(path=target)

    # Remove treebanks with no text (see: v2.14, UD_English_GUMReddit, for example)
    no_text_chars = {" ", "\n", "_"}
    for treebank in (target / UD_DIR_NAME).iterdir():
        if set(only(treebank.glob("*test.txt")).read_text()).issubset(no_text_chars):
            shutil.rmtree(treebank)

    # Remove treebanks with no lemmas (see: v2.14, UD_Chinese-PatentChar, for example)
    for treebank in (target / UD_DIR_NAME).iterdir():
        if not stanza.utils.datasets.prepare_lemma_treebank.check_lemmas(only(treebank.glob("*test.conllu"))):
            shutil.rmtree(treebank)

    # Stanza does not run evaluation if train files are not present, imitate them
    for treebank in (target / UD_DIR_NAME).iterdir():
        for ext in {"conllu", "txt"}:
            if ilen(treebank.glob(f"*train.{ext}")) == 0:
                test_path = treebank.absolute() / only(treebank.glob(f"*test.{ext}"))
                (treebank.absolute() / test_path.name.replace("test", "train")).symlink_to(test_path)


def prepare_ud(
        url: httpx.URL = UD_URL, target: Path = Path.cwd(), save_tar: bool = False
) -> Path:
    result = target / UD_DIR_NAME
    if not result.exists():
        try:
            if Path(target / UD_NAME).exists():
                f = open(target / UD_NAME, "rb")
            else:
                f = download_ud(url)
            unpack_ud(f=f, target=target, save_tar=save_tar)
        finally:
            f.close()
    else:
        print("Skipping download of UD treebanks")
    return result


def get_spacy_models() -> Mapping[str, SpacyModel]:
    # Get spacy models names and wheel urls by fetching releases from GitHub
    github = Github()
    repo = github.get_repo("explosion/spacy-models")
    result = {}
    for release in repo.get_releases():
        name, _, version_str = release.tag_name.rpartition("-")
        version = Version(version_str)
        # To avoid listing releases with old models
        if version < SPACY_MODEL_VERSION_MIN:
            return result
        if (version < SPACY_MODEL_VERSION_MAX) and (
                (model := result.get(name)) is None or version > model.version
        ):
            for asset in release.assets:
                if asset.name.endswith(".whl"):
                    language = name.partition("_")[0]
                    result[name] = SpacyModel(
                        name=name,
                        version=version,
                        language=language,
                        url=asset.browser_download_url,
                    )
    return result


def install_spacy_model(
        models: Collection[SpacyModel], target: Path | None = None
) -> Collection[SpacyModel]:
    models_to_install = [
        model
        for model in models
        if subprocess.run(
            ["uv", "pip", "show", model.name], capture_output=True
        ).returncode
           != 0
    ]
    if models_to_install:
        cmd = ["uv", "pip", "install"] + [str(model.url) for model in models_to_install]
        if target is not None:
            cmd += ["--target", target]
        subprocess.check_call(cmd)
    result = []
    for model in models:
        location_line = only(
            line
            for line in subprocess.run(
                ["uv", "pip", "show", model.name], capture_output=True, text=True,
        ).stdout.splitlines()
            if line.startswith("Location:")
        )
        path = Path(location_line.partition("Location: ")[2]) / model.name
        byte_size = pydantic.ByteSize(sum(file.stat().st_size for file in path.rglob('*')))
        result.append(
            model.copy(update={
                "byte_size": byte_size,
                "size": byte_size.human_readable(),
            })
        )
    return result


def uninstall_spacy_model(
        models: Collection[SpacyModel], target: Path | None = None
) -> None:
    cmd = ["uv", "pip", "uninstall"] + [model.name for model in models]
    if target is not None:
        cmd += ["--target", target]
    subprocess.check_call(cmd)


def prepare_spacy(
        models: Collection[SpacyModel], ud_path: Path = Path.cwd() / UD_DIR_NAME, dest_dir: Path | None = Path.cwd() / "spacy",
) -> Collection[SpacyModel]:
    if not (symlink := (dest_dir / UD_DIR_NAME)).exists():
        symlink.symlink_to(ud_path, target_is_directory=True)
    return install_spacy_model(models=models)


def run_spacy(eval_args: EvalArgs) -> EvalResult:
    if eval_args.use_gpu:
        spacy.require_gpu()
    spacy_workdir = eval_args.workdir
    spacy_workdir.mkdir(parents=True, exist_ok=True)
    try:
        ud_path = eval_args.workdir / UD_DIR_NAME
        treebank_path = ud_path / eval_args.treebank
        txt_path = only(treebank_path.glob("*test.txt"))
        text = txt_path.read_text(encoding="utf8").strip()
        text = re.sub(r"([^\n])\n([^\n])", r"\1 \2", text)
        text = re.sub(r" +", " ", text)
        nlp = spacy_conll.init_parser(
            model_or_lang=eval_args.model.name,
            parser="spacy",
            disable_pandas=True,
            include_headers=True,
            exclude_spacy_components=["ner"],
        )
        nlp.max_length = sys.maxsize
        parser = spacy_conll.ConllParser(nlp)
        conll_str = parser.parse_text_as_conll(
            text=text,
            no_split_on_newline=False,
        )
        gold_conllu_file = only(treebank_path.glob("*test.conllu"))
        system_conllu_file = (
                spacy_workdir / f"{eval_args.model.name}_{gold_conllu_file.name}"
        )
        system_conllu_file.write_text(conll_str + "\n", encoding="utf8")
        evaluation = ud_scores(gold_conllu_file, system_conllu_file)
        el = evaluation["Lemmas"]
        f1 = Decimal(el.f1).quantize(DECIMAL_FOUR_PLACES)
        return EvalResult(eval_args=eval_args, f1=f1)
    except Exception:
        logging.exception(f"Failed to process: {eval_args}")
        return EvalResult(eval_args=eval_args, f1=None)


def get_stanza_model_list(
        language: str,
        processor: str = "lemma",
        models_dir: Path = stanza.resources.common.DEFAULT_MODEL_DIR,
) -> Collection[
    Collection[str, Collection[stanza.resources.common.ModelSpecification]]
]:
    _, _, package, processors = stanza.resources.common.process_pipeline_parameters(
        lang=language,
        model_dir=models_dir,
        package=defaultdict(None),
        processors=processor,
    )
    resources = stanza.resources.common.load_resources_json(models_dir)
    try:
        download_list = stanza.resources.common.maintain_processor_list(
            resources=resources,
            lang=language,
            package=package,
            processors=processors,
        )
        download_list = stanza.resources.common.add_dependencies(resources, language, download_list)
        return download_list
    except KeyError:
        return []


def get_stanza_model_config(
        language: str,
        processor: str = "lemma",
        models_dir: Path = stanza.resources.common.DEFAULT_MODEL_DIR,
) -> Mapping[str, str]:
    resources = stanza.resources.common.load_resources_json(models_dir)
    return stanza.pipeline.core.build_default_config(
        resources=resources,
        lang=language,
        model_dir=models_dir,
        load_list=get_stanza_model_list(language=language, processor=processor),
    )


def get_stanza_model_paths(
        language: str,
        processor: str = "lemma",
        models_dir: Path = stanza.resources.common.DEFAULT_MODEL_DIR,
) -> Iterable[Path]:
    config = get_stanza_model_config(language=language, processor=processor, models_dir=models_dir)
    return (
        Path(value)
        for processor in processor.split(",")
        for key, value in config.items()
        if re.match(fr"{processor}.*_path", key)
    )


def get_stanze_model_size(
        language: str,
        processors: str,
        models_dir: Path = stanza.resources.common.DEFAULT_MODEL_DIR,
) -> pydantic.ByteSize:
    return pydantic.ByteSize(
        sum(
            model_path.stat().st_size
            for processor, _ in get_stanza_model_list(language=language, processor=processors, models_dir=models_dir)
        for model_path in get_stanza_model_paths(language=language, processor=processor, models_dir=models_dir)
        )
    )


def prepare_stanza_treebank(ud_path: Path, treebank: str, dest_dir: Path) -> None:
    short_name = stanza.models.common.constant.treebank_to_short_name(treebank)
    short_language = short_name.split("_")[0]
    with tempfile.TemporaryDirectory() as tokenizer_dir:
        stanza.utils.datasets.prepare_tokenizer_treebank.process_partial_ud_treebank(
            treebank=treebank,
            udbase_dir=ud_path,
            tokenizer_dir=tokenizer_dir,
            short_name=short_name,
            short_language=short_language,
        )
        shards = ("test", "train")
        stanza.utils.datasets.common.convert_conllu_to_txt(tokenizer_dir, short_name, shards=shards)
        for shard in shards:
            stanza.utils.datasets.prepare_tokenizer_treebank.copy_conllu_file(tokenizer_dir, f"{shard}.gold", dest_dir, f"{shard}.in", short_name)
            stanza.utils.datasets.prepare_tokenizer_treebank.copy_conllu_file(dest_dir, f"{shard}.in", dest_dir, f"{shard}.gold", short_name)


def prepare_stanza(
        ud_path: Path,
        models_dir: Path = stanza.resources.common.DEFAULT_MODEL_DIR,
        languages: Collection[str] | None = None,
        treebanks: Collection[str] | None = None,
        use_pos: bool = True,
        dest_dir=Path.cwd() / "stanza",
        use_gpu: Optional[bool] = None,
) -> Collection[EvalArgs]:
    if not (symlink := (dest_dir / UD_DIR_NAME)).exists():
        symlink.symlink_to(ud_path, target_is_directory=True)

    resources = stanza.resources.common.load_resources_json(models_dir)
    languages_available = set()
    pos = ",pos" if use_pos else ""
    processor = f"tokenize,lemma{pos}"
    for language in stanza.resources.common.list_available_languages():
        if language == "multilingual" or not ((languages is None) or (language in languages)):
            continue
        download_list = get_stanza_model_list(
            language=language, processor=processor, models_dir=models_dir
        )
        download_list = stanza.pipeline.core.filter_variants(download_list)
        download_list = stanza.pipeline.core.flatten_processor_list(download_list)
        if not download_list:
            continue
        try:
            stanza.resources.common.download_models(
                download_list,
                resources=resources,
                lang=language,
                model_dir=models_dir,
                log_info=True,
            )
        except ValueError:
            continue
        languages_available.add(language)

    ret = subprocess.run(["uv", "pip", "show", "stanza"], check=True, capture_output=True, text=True)
    version = Version(only(line.removeprefix("Version: ") for line in ret.stdout.splitlines() if line.startswith("Version")))

    result = [
        EvalArgs(
            model=StanzaModel(
                language=language,
                version=version,
                byte_size=(byte_size := get_stanze_model_size(language, processors=processor)),
                size=byte_size.human_readable(),
            ),
            treebank=treebank,
            score_type="lemma",
            use_pos=use_pos,
            workdir=dest_dir,
            use_gpu=use_gpu,
        )
        for language in languages_available
        for treebank in treebanks_for_language(language=language, ud=ud_path)
        if (treebanks is None) or (treebank in treebanks)
    ]
    return result


def run_stanza(eval_args: EvalArgs) -> EvalResult:
    if eval_args.score_type != "lemma":
        return EvalResult(eval_args=eval_args, f1=None)

    pos = ",pos" if eval_args.use_pos else ""
    processors = f"tokenize,lemma{pos}"
    nlp = stanza.Pipeline(eval_args.model.language, processors=processors, tokenize_max_seqlen=sys.maxsize, use_gpu=eval_args.use_gpu)
    treebank_path = eval_args.workdir / UD_DIR_NAME / eval_args.treebank
    text_path = only(treebank_path.glob("*test.txt"))
    text = text_path.read_text()
    doc = nlp(text)
    conll_str = f"{doc:C}"
    gold_conllu_file = only(treebank_path.glob("*test.conllu"))
    system_conllu_file = (
            eval_args.workdir / f"{eval_args.model.language}_{gold_conllu_file.name}"
    )
    system_conllu_file.write_text(f"{conll_str}\n\n")
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation["Lemmas"]
    f1 = Decimal(el.f1).quantize(DECIMAL_FOUR_PLACES)
    return EvalResult(eval_args=eval_args, f1=f1)


def cleanup_stanza(dest_dir: Path = Path("data/lemma")) -> None:
    shutil.rmtree(dest_dir)


def run(eval_args: EvalArgs) -> EvalResult:
    if eval_args.model.library == "spacy":
        return run_spacy(eval_args)
    if eval_args.model.library == "stanza":
        return run_stanza(eval_args)


def treebanks_for_language(language: str, ud: Path = Path.cwd() / UD_DIR_NAME) -> Iterable[str]:
    if language == "xx" or language == "multilingual":
        return []
    # For zh-hans (Simplified_Chinese) and zh-hant (Traditional_Chinese)
    language = language.partition("-")[0]
    language_display_name = langcodes.Language.get(language).display_name()
    return (path.name for path in ud.glob(f"UD_{language_display_name}-*"))


def get_spacy_eval_args(
        ud_path: Path | None,
        spacy_models: Collection[SpacyModel],
        languages: Collection[str] | None,
        treebanks: Collection[str] | None,
        workdir: Path,
        use_gpu: bool | None = None,
) -> Iterable[EvalArgs]:
    return (
        EvalArgs(
            model=model,
            use_pos=True,
            treebank=treebank,
            workdir=workdir,
            score_type="lemma",
            use_gpu=use_gpu,
        )
        for model in spacy_models
        for treebank in treebanks_for_language(language=model.language, ud=ud_path)
        if ((languages is None) or (model.language in languages)) and ((treebanks is None) or (treebank in treebanks))
    )


def main(
        n_jobs: int = 1,
        library: Optional[list[Library]] = None,
        language: Optional[list[str]] = None,
        treebank: Optional[list[str]] = None,
        use_pos: Optional[list[bool]] = None,
        workdir: Path = Path.cwd(),
        cleanup_before: bool = True,
        cleanup_after: bool = True,
        cleanup_all: Optional[bool] = None,
        use_gpu: Optional[bool] = None,
) -> None:
    if not library:
        library = list(Library)
    
    if not language:
        language = None

    if not use_pos:
        use_pos = [True, False]
        
    if cleanup_all is not None:
        cleanup_after = cleanup_before = cleanup_all

    ud_path = workdir / UD_DIR_NAME
    prepare_ud(httpx.URL(UD_URL), workdir, save_tar=True)

    with Cache(str(workdir)) as cache:
        spacy_models = cache.memoize(expire=CACHING_SECONDS)(get_spacy_models)()
        treebanks = set(treebank) if treebank else None
        eval_args_list = []
        spacy_workdir = workdir / "spacy"
        stanza_workdirs = [workdir / f"stanza{('-pos' if use_pos else '')}" for use_pos in use_pos]
        workdirs = (
            ([spacy_workdir] if Library.spacy in library else [])
            +
            ([stanza_workdir for stanza_workdir in stanza_workdirs] if Library.stanza in library else [])
        )
        try:
            if cleanup_before:
                for workdir in workdirs:
                    shutil.rmtree(workdir, ignore_errors=True)

            if Library.spacy in library:
                spacy_workdir.mkdir(parents=True, exist_ok=True)
                spacy_models = prepare_spacy(spacy_models.values(), ud_path=ud_path, dest_dir=spacy_workdir)
                eval_args_list += get_spacy_eval_args(
                    ud_path=ud_path,
                    spacy_models=spacy_models,
                    languages=language,
                    treebanks=treebanks,
                    workdir=spacy_workdir,
                    use_gpu=use_gpu,
                )
            if Library.stanza in library:
                for stanza_workdir, use_pos in zip(stanza_workdirs, use_pos):
                    stanza_workdir.mkdir(parents=True, exist_ok=True)
                    stanza_workdirs.append(stanza_workdir)
                    eval_args_list += prepare_stanza(
                        ud_path=ud_path,
                    languages=language,
                         treebanks=treebanks,
                          use_pos=use_pos,
                           dest_dir=stanza_workdir,
                        use_gpu=use_gpu,
                           )

            parallel = Parallel(n_jobs=n_jobs)
            results = parallel(
                delayed(cache.memoize(
                    name=eval_args.key,
                    expire=CACHING_SECONDS,
                    ignore={0}
                )(run))(eval_args)
                for eval_args in eval_args_list
            )
        finally:
            if cleanup_after:
                for workdir in workdirs:
                    shutil.rmtree(workdir, ignore_errors=True)

        result = pandas.DataFrame.from_records(
            result.model_dump()
            for result in results
        )
        print(result)
        result.to_csv("result.csv")


if __name__ == "__main__":
    typer.run(main)
