"""Small, deterministic validation for generated answer citations."""

import re

_CITATION = re.compile(r"\[Source ([1-9][0-9]*)\]")
_SOURCE_REFERENCE = re.compile(r"\[\s*(?i:source)")


def _index_is_in_range(digits: str, source_count: int) -> bool:
    limit = str(source_count)
    return len(digits) < len(limit) or (
        len(digits) == len(limit) and digits <= limit
    )


def validate_citations(answer: str, source_count: int) -> bool:
    """Return whether *answer* has only canonical, in-range source references.

    This validates citation syntax and indexes only.  It does not decide whether
    the cited evidence entails the answer.
    """
    if (
        not isinstance(answer, str)
        or not answer.strip()
        or type(source_count) is not int
        or source_count < 1
    ):
        return False

    citations = list(_CITATION.finditer(answer))
    if not citations or any(
        not _index_is_in_range(match.group(1), source_count) for match in citations
    ):
        return False

    # Treat every bracketed token beginning with "Source" as a citation.  This
    # catches case, spacing, spelling, and unterminated-reference mistakes even
    # when a valid citation appears elsewhere in the answer.
    for marker in _SOURCE_REFERENCE.finditer(answer):
        end = answer.find("]", marker.start())
        if end < 0 or _CITATION.fullmatch(answer[marker.start() : end + 1]) is None:
            return False
    return True
