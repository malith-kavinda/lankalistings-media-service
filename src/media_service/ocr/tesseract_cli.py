"""A typed wrapper over the Tesseract binary.

This drives the executable directly instead of going through `pytesseract`, for three reasons that
were each verified against the real binary on this machine rather than assumed.

**`--tessdata-dir` cannot be passed through `pytesseract` on Windows.** Its config string is split
with `shlex.split(config, posix=False)`, which keeps quote characters inside the token, so a quoted
path arrives at Tesseract with the quotes attached and an unquoted one is split at the space. The
language data lives under a path containing a space here, and Sinhala lives only there -- the system
install ships `eng` and not `sin` -- so this is the difference between the service reading Sinhala
and not. Passing the directory through the subprocess's own environment sidesteps the quoting
entirely, and because the environment belongs to that one subprocess, nothing in this process is
mutated.

**The `tsv` output mode is requested with `-c tessedit_create_tsv=1`, never the `tsv` configfile.**
Tesseract resolves configfiles relative to `TESSDATA_PREFIX`; point that at a directory holding only
traineddata and the `tsv` file is not found, at which point Tesseract prints **plain text and exits
zero**. A caller parsing that output as TSV gets a confident answer built from nonsense. The `-c`
form needs no configfile and cannot fail that way.

**Timeouts belong to the call.** `OCR_TIMEOUT_SECONDS` is enforced here, where the process can
actually be killed.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from tempfile import mkstemp
from typing import Final

from media_service.api.errors import ServiceError

TSV_COLUMNS: Final = 12
VERSION_PREFIX: Final = "tesseract "


class TesseractRunError(ServiceError):
    def __init__(self, message: str, *, code: str = "OCR_FAILED") -> None:
        super().__init__(status_code=500, code=code, message=message)


class TesseractTimeoutError(TesseractRunError):
    def __init__(self, seconds: float) -> None:
        super().__init__(f"Tesseract did not finish within {seconds:g}s.", code="OCR_TIMEOUT")


@dataclass(frozen=True, slots=True)
class TesseractOptions:
    command: str | None = None
    tessdata_dir: Path | None = None
    languages: str = "sin+eng"
    # 3 is fully automatic page segmentation, which is what a newspaper page needs: the columns are
    # the structure OCR has to find, not something the caller can describe in advance.
    psm: int = 3
    # 3 selects the LSTM engine where available, which is the only one with a Sinhala model.
    oem: int = 3
    timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class TsvRow:
    level: int
    page_num: int
    block_num: int
    par_num: int
    line_num: int
    word_num: int
    left: int
    top: int
    width: int
    height: int
    conf: float
    text: str


class TesseractCli:
    def __init__(self, options: TesseractOptions | None = None) -> None:
        self._options = options or TesseractOptions()
        self._version: str | None = None
        self._languages: frozenset[str] | None = None

    @property
    def options(self) -> TesseractOptions:
        return self._options

    def resolve_command(self) -> str | None:
        return self._options.command or which("tesseract")

    def version(self) -> str | None:
        """The binary's version string, read once per process."""
        if self._version is None:
            output = self._run(["--version"], timeout=10.0)
            if output is None:
                return None
            first = output.splitlines()[0].strip() if output.strip() else ""
            self._version = first.removeprefix(VERSION_PREFIX).lstrip("v") or first
        return self._version

    def languages(self) -> frozenset[str] | None:
        """Language codes the binary can see, read once per process.

        Read through the same environment the recognition call uses, or it would report the system
        install's languages while recognition ran against a different directory.
        """
        if self._languages is None:
            output = self._run(["--list-langs"], timeout=10.0)
            if output is None:
                return None
            # The first line is a human-readable header naming the directory searched.
            lines = [line.strip() for line in output.splitlines()[1:] if line.strip()]
            self._languages = frozenset(lines)
        return self._languages

    def image_to_tsv(self, image_bytes: bytes, *, suffix: str = ".png") -> str:
        """Recognise an image and return Tesseract's TSV, raising on failure."""
        handle, path = mkstemp(suffix=suffix, prefix="ocr-")
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(image_bytes)
            output = self._run(
                [
                    path,
                    "stdout",
                    "-l",
                    self._options.languages,
                    "--psm",
                    str(self._options.psm),
                    "--oem",
                    str(self._options.oem),
                    "-c",
                    "tessedit_create_tsv=1",
                ],
                timeout=self._options.timeout_seconds,
                required=True,
            )
            return output or ""
        finally:
            Path(path).unlink(missing_ok=True)

    # -- internals -----------------------------------------------------------------------------

    def _environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        if self._options.tessdata_dir is not None:
            # Scoped to the child process. The plan's `--tessdata-dir` flag cannot carry this path
            # through pytesseract's quoting on Windows; this can, and it mutates nothing here.
            environment["TESSDATA_PREFIX"] = str(self._options.tessdata_dir)
        # Tesseract's own OpenMP threading fights the worker pool. The pool sets this too; setting
        # it per call keeps a single-threaded caller honest as well.
        environment.setdefault("OMP_THREAD_LIMIT", "1")
        return environment

    def _run(self, arguments: list[str], *, timeout: float, required: bool = False) -> str | None:
        command = self.resolve_command()
        if command is None:
            if required:
                raise TesseractRunError("The Tesseract executable is not installed or on PATH.")
            return None

        try:
            completed = subprocess.run(  # noqa: S603 - a fixed binary with an argument list
                [command, *arguments],
                capture_output=True,
                env=self._environment(),
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TesseractTimeoutError(timeout) from exc
        except OSError as exc:
            if required:
                raise TesseractRunError(f"Tesseract could not be started: {exc}") from exc
            return None

        if completed.returncode != 0:
            if required:
                detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()
                raise TesseractRunError(
                    f"Tesseract exited {completed.returncode}: "
                    f"{detail[0] if detail else 'no error output'}"
                )
            return None

        # UTF-8 with replacement, never a strict decode: a garbled byte in one word must not cost
        # the whole page, and Sinhala output is the reason the encoding is stated explicitly.
        return completed.stdout.decode("utf-8", "replace")


def parse_tsv(output: str) -> list[TsvRow]:
    """Parse Tesseract's TSV.

    The text column is last and may itself contain tabs, so the split is bounded rather than
    greedy. Rows that are short, or whose numbers do not parse, are dropped: Tesseract emits
    structural rows that carry no word, and a parser that guessed at them would invent geometry.
    """
    rows: list[TsvRow] = []
    for index, line in enumerate(output.splitlines()):
        if index == 0 and line.startswith("level\t"):
            continue
        parts = line.split("\t", TSV_COLUMNS - 1)
        if len(parts) < TSV_COLUMNS:
            continue
        try:
            rows.append(
                TsvRow(
                    level=int(parts[0]),
                    page_num=int(parts[1]),
                    block_num=int(parts[2]),
                    par_num=int(parts[3]),
                    line_num=int(parts[4]),
                    word_num=int(parts[5]),
                    left=int(parts[6]),
                    top=int(parts[7]),
                    width=int(parts[8]),
                    height=int(parts[9]),
                    conf=float(parts[10]),
                    text=parts[11],
                )
            )
        except ValueError:
            continue
    return rows


def as_word_data(rows: list[TsvRow]) -> dict[str, list]:
    """Reshape parsed rows into the column-per-key layout `blocks.words_from_tesseract` reads."""
    keys = (
        "level",
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "word_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text",
    )
    return {key: [getattr(row, key) for row in rows] for key in keys}
