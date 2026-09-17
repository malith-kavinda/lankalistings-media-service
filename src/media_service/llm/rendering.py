"""Rendering OCR blocks into the prompt.

The format puts the **id first** so the model can cite it, then geometry so it can resolve columns,
then confidence so it can tell a clean reading from a damaged one:

    [4] (x=112,y=880,w=430,h=64) conf=0.94
    ටොයොටා ප්‍රියස් 2016

Two protections live here, and both are about a page being hostile rather than merely messy.

**The delimiter is escaped inside the text.** The source is wrapped in open and close
`ocr_blocks` tags, so a newspaper image carrying the literal closing tag would end the untrusted
region early and anything after it would read as instructions. Escaping it is what makes AC-013 an
enforced property rather than a hope resting on the system prompt.

**Control and bidirectional-override characters are stripped.** A right-to-left override can make
rendered text read as something other than its bytes, which is a way to hide an instruction in plain
sight. Newlines and tabs survive, because they are layout.

Truncation is on **block boundaries**, never mid-block, and it adds a warning. A block cut in half
would be cited by id as though it were whole, and the citation would point at text the model never
saw in full (Phase 0A item 13: truncate, do not chunk).
"""

from __future__ import annotations

import unicodedata
from typing import Final

from media_service.ocr.types import OcrBlock

OPEN_DELIMITER: Final = "<ocr_blocks>"
CLOSE_DELIMITER: Final = "</ocr_blocks>"

TRUNCATED_WARNING: Final = "OCR_TEXT_TRUNCATED"
DELIMITER_ESCAPED_WARNING: Final = "OCR_DELIMITER_ESCAPED"

# Bidirectional formatting characters. Harmless in a document, a disguise in a prompt.
BIDI_CONTROLS: Final = frozenset(
    "‪‫‬‭‮⁦⁧⁨⁩‎‏"
)

KEEP_CONTROLS: Final = frozenset("\n\t")


def escape_delimiters(text: str) -> tuple[str, bool]:
    """Neutralise the block delimiters inside untrusted text.

    A backslash is enough to break the exact match while leaving the text readable to a reviewer
    comparing it against the scan -- and readable to the model as the content it is.
    """
    escaped = text.replace(CLOSE_DELIMITER, "<\\/ocr_blocks>").replace(
        OPEN_DELIMITER, "<\\ocr_blocks>"
    )
    return escaped, escaped != text


def sanitize(text: str) -> str:
    """NFC, no control characters, no bidirectional overrides.

    NFC and never NFKC: compatibility normalisation decomposes Sinhala conjuncts and the text stops
    round-tripping, which would breach AC-012.
    """
    normalized = unicodedata.normalize("NFC", text)
    return "".join(
        character
        for character in normalized
        if character in KEEP_CONTROLS
        or (character not in BIDI_CONTROLS and unicodedata.category(character) != "Cc")
    )


def render_block(block: OcrBlock) -> str:
    """One block, id first."""
    header = f"[{block.id}]"
    if block.box is not None:
        header += (
            f" (x={block.box.left},y={block.box.top},"
            f"w={block.box.width},h={block.box.height})"
        )
    if block.confidence is not None:
        header += f" conf={block.confidence:.2f}"
    return f"{header}\n{block.text}"


def render_blocks(
    blocks: tuple[OcrBlock, ...] | list[OcrBlock], *, max_chars: int
) -> tuple[str, tuple[str, ...]]:
    """Render every block that fits, and say what had to be left out."""
    warnings: list[str] = []
    rendered: list[str] = []
    used = 0
    escaped_any = False

    for block in blocks:
        text, was_escaped = escape_delimiters(sanitize(block.text))
        escaped_any = escaped_any or was_escaped
        entry = render_block(_with_text(block, text))

        # +2 for the blank line between entries. Checked before appending so the budget is never
        # exceeded, rather than trimmed afterwards.
        cost = len(entry) + (2 if rendered else 0)
        if used + cost > max_chars:
            warnings.append(TRUNCATED_WARNING)
            break
        rendered.append(entry)
        used += cost

    if escaped_any:
        warnings.append(DELIMITER_ESCAPED_WARNING)
    return "\n\n".join(rendered), tuple(warnings)


def _with_text(block: OcrBlock, text: str) -> OcrBlock:
    from dataclasses import replace

    return replace(block, text=text)
