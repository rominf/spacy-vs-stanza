from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Type

import packaging.version
import pydantic
import pydantic_core
from pydantic import HttpUrl


class Library(StrEnum):
    spacy = "spacy"
    stanza = "stanza"


class VersionPydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Type[Any], handler: pydantic.GetCoreSchemaHandler
    ) -> pydantic_core.core_schema.CoreSchema:
        def validate_from_str(value: str) -> packaging.version.Version:
            return Version(value)

        from_str_schema = pydantic_core.core_schema.chain_schema(
            [
                pydantic_core.core_schema.str_schema(),
                pydantic_core.core_schema.no_info_plain_validator_function(
                    validate_from_str
                ),
            ]
        )
        return pydantic_core.core_schema.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=pydantic_core.core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    pydantic_core.core_schema.is_instance_schema(
                        packaging.version.Version
                    ),
                    from_str_schema,
                ]
            ),
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                lambda instance: str(instance)
            ),
        )


Version = Annotated[packaging.version.Version, VersionPydanticAnnotation]


class Model(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    library: str
    language: str
    version: Version
    name: str = pydantic.Field(default="default")
    path: Path | None = pydantic.Field(default=None, exclude=True)
    byte_size: pydantic.ByteSize | None = pydantic.Field(default=None, exclude=True)
    size: str | None = pydantic.Field(default=None)


class SpacyModel(Model):
    url: HttpUrl = pydantic.Field(exclude=True)
    library: str = "spacy"

    @property
    def key(self) -> str:
        return f"{self.library}-{self.language}-{self.name}-{self.version}"


class StanzaModel(Model):
    library: str = "stanza"

    @property
    def key(self) -> str:
        return f"{self.library}-{self.language}"


class EvalArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    model: Model
    treebank: str
    score_type: Literal["lemma"]
    use_pos: bool = True
    workdir: Path = Path.cwd()
    use_gpu: bool | None = None

    @property
    def key(self) -> str:
        pos = "-pos" if self.use_pos else ""
        return f"{self.model.key}{pos}-{self.treebank}-{self.score_type}"


class EvalResult(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    eval_args: EvalArgs
    f1: Decimal | None = pydantic.Field(default=None)

    @pydantic.model_serializer
    def model_serializer(self) -> Mapping[str, Any]:
        result = {
            **self.eval_args.model.model_dump(),
            "use_pos": self.eval_args.use_pos,
            "treebank": self.eval_args.treebank,
        }
        result[f"{self.eval_args.score_type} f1"] = self.f1
        return result
