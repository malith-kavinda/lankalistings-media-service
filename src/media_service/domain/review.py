"""What a reviewer is allowed to decide, and what has to be true before they can.

Two rules live here rather than in the service, because both are statements about the domain rather
than about HTTP.

`publication_errors` is FR-REV-008: approval fails with *field-level* errors when required public
values are missing. Field-level matters. "Approval failed" sends a reviewer hunting through six
fields; "title is required" is one click. The check runs against the advertisement as it will be
after the reviewer's edits, never against what the model produced, because the whole point of the
review step is that a person may fix exactly these gaps.

`REJECTION_REASONS` is a closed vocabulary rather than free text. FR-REV-006 needs a reason; PRD
15.5 needs to *group* by it, which free text cannot do. The note beside it stays free-form for the
detail that does not fit a code.
"""

from __future__ import annotations

from typing import Final

from media_service.domain.categories import DEFAULT_CATALOG

# Closed set. A new reason is a deliberate addition here plus a portal label, not a string a
# reviewer invents at 2am that nothing can ever count.
REJECTION_REASONS: Final[dict[str, str]] = {
    "not_an_advertisement": "The region is not an advertisement (masthead, article, notice).",
    "duplicate": "The same advertisement was already captured from another image.",
    "unreadable_source": "The scan is too poor to trust any extracted value.",
    "garbled_extraction": "The text was read, but the extracted fields are wrong.",
    "prohibited_content": "The advertisement may not be published.",
    "incomplete": "Too little of the advertisement is present to publish it.",
    "other": "Something else, described in the note.",
}

# A note is optional for every coded reason except this one: `other` carries no information at all
# without it, which would defeat the purpose of asking.
REASON_REQUIRING_NOTE: Final = "other"

MAX_NOTE_LENGTH: Final = 2000

# What a published advertisement must have. Deliberately short: description, price and phone are
# genuinely optional in print classifieds -- a great many real ads are three words and a number --
# and demanding them would make the queue unclearable rather than make the data better.
REQUIRED_FOR_PUBLICATION: Final = ("title", "category")

MIN_TITLE_LENGTH: Final = 3


def publication_errors(
    *, title: str, category: str, location: str, price: str, description: str, phones: list[str]
) -> list[dict[str, str | None]]:
    """Every reason this candidate cannot be published, as API error details (FR-REV-008).

    All of them, not the first: a reviewer fixing one field at a time, reloading between each, is
    the failure mode this exists to avoid.
    """
    errors: list[dict[str, str | None]] = []

    if not title.strip():
        errors.append(_error("title", "REQUIRED", "A title is required before publishing."))
    elif len(title.strip()) < MIN_TITLE_LENGTH:
        errors.append(
            _error(
                "title",
                "TOO_SHORT",
                f"A title needs at least {MIN_TITLE_LENGTH} characters.",
            )
        )

    if not category.strip():
        errors.append(_error("category", "REQUIRED", "A category is required before publishing."))
    elif not DEFAULT_CATALOG.contains(category):
        errors.append(
            _error("category", "UNKNOWN_VALUE", f"{category!r} is not a known category.")
        )

    # Not required, but a published advertisement with no way to respond to it is useless to a
    # reader, so it is worth saying out loud rather than discovering from complaints.
    if not phones and not description.strip() and not location.strip():
        errors.append(
            _error(
                "phones",
                "NO_CONTACT",
                "Add a phone number, a description, or a location: this advertisement gives a "
                "reader no way to act on it.",
            )
        )

    return errors


def rejection_errors(reason_code: str, note: str | None) -> list[dict[str, str | None]]:
    errors: list[dict[str, str | None]] = []

    if reason_code not in REJECTION_REASONS:
        errors.append(
            _error(
                "reason_code",
                "UNKNOWN_VALUE",
                f"reason_code must be one of {', '.join(sorted(REJECTION_REASONS))}.",
            )
        )
    elif reason_code == REASON_REQUIRING_NOTE and not (note or "").strip():
        errors.append(
            _error(
                "note",
                "REQUIRED",
                f"A note is required when reason_code is {REASON_REQUIRING_NOTE!r}.",
            )
        )

    if note is not None and len(note) > MAX_NOTE_LENGTH:
        errors.append(
            _error("note", "TOO_LONG", f"A note may be at most {MAX_NOTE_LENGTH} characters.")
        )

    return errors


def _error(field: str, code: str, message: str) -> dict[str, str | None]:
    return {"field": field, "code": code, "message": message}
