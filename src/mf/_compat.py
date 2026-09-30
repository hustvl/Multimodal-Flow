from __future__ import annotations

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)


try:
    from typing import Self
except ImportError:
    from typing_extensions import Self  # noqa: UP035


__all__ = ["Self", "StrEnum", "UTC"]
