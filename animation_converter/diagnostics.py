"""Structured diagnostics shared by parsers, validators, and front ends."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Severity(IntEnum):
    INFO = 10
    WARNING = 20
    ERROR = 30
    FATAL = 40


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    severity: Severity
    category: str
    source_file: str | None = None
    object_id: str | None = None
    remediation: str | None = None
    can_continue: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "level": self.severity.name,
            "code": self.code,
            "category": self.category,
            "message": self.message,
            "canContinue": self.can_continue,
        }
        if self.source_file:
            data["sourceFile"] = self.source_file
        if self.object_id:
            data["objectId"] = self.object_id
        if self.remediation:
            data["suggestedRemediation"] = self.remediation
        if self.details:
            data["details"] = self.details
        return data


class DiagnosticCollection:
    def __init__(self, items: Iterable[Diagnostic] = ()) -> None:
        self._items = list(items)

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        *,
        category: str,
        source_file: str | None = None,
        object_id: str | None = None,
        remediation: str | None = None,
        can_continue: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> Diagnostic:
        diagnostic = Diagnostic(
            code=code,
            message=message,
            severity=severity,
            category=category,
            source_file=source_file,
            object_id=object_id,
            remediation=remediation,
            can_continue=(severity < Severity.FATAL if can_continue is None else can_continue),
            details=details or {},
        )
        self._items.append(diagnostic)
        return diagnostic

    def info(self, code: str, message: str, *, category: str, **kwargs: Any) -> Diagnostic:
        return self.add(Severity.INFO, code, message, category=category, **kwargs)

    def warning(self, code: str, message: str, *, category: str, **kwargs: Any) -> Diagnostic:
        return self.add(Severity.WARNING, code, message, category=category, **kwargs)

    def error(self, code: str, message: str, *, category: str, **kwargs: Any) -> Diagnostic:
        return self.add(Severity.ERROR, code, message, category=category, **kwargs)

    def fatal(self, code: str, message: str, *, category: str, **kwargs: Any) -> Diagnostic:
        kwargs.setdefault("can_continue", False)
        return self.add(Severity.FATAL, code, message, category=category, **kwargs)

    def append(self, diagnostic: Diagnostic) -> None:
        self._items.append(diagnostic)

    def extend(self, diagnostics: Iterable[Diagnostic] | DiagnosticCollection) -> None:
        self._items.extend(diagnostics)

    def has_at_least(self, severity: Severity) -> bool:
        return any(item.severity >= severity for item in self._items)

    @property
    def has_errors(self) -> bool:
        return self.has_at_least(Severity.ERROR)

    @property
    def has_fatal(self) -> bool:
        return self.has_at_least(Severity.FATAL)

    @property
    def has_warnings(self) -> bool:
        return self.has_at_least(Severity.WARNING)

    def by_category(self, category: str) -> list[Diagnostic]:
        return [item for item in self._items if item.category == category]

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items]

    def counts(self) -> dict[str, int]:
        return {severity.name: sum(item.severity == severity for item in self._items) for severity in Severity}
