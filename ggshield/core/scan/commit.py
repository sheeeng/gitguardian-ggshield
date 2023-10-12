import re
from functools import cached_property
from pathlib import Path
from typing import Callable, Iterable, List, NamedTuple, Optional, Set, Tuple

from ggshield.core.text_utils import STYLE, format_text
from ggshield.utils.files import is_filepath_excluded
from ggshield.utils.git_shell import Filemode, git

from .scannable import Scannable, StringScannable


_RX_HEADER_LINE_SEPARATOR = re.compile("[\n\0]:", re.MULTILINE)


REGEX_HEADER_INFO = re.compile(
    r"Author:\s(?P<author>.+?) <(?P<email>.+?)>\nDate:\s+(?P<date>.+)?\n"
)


class PatchParseError(Exception):
    """
    Raised by Commit.get_files() if it fails to parse its patch.
    """

    pass


def _parse_patch(
    patch: str, exclusion_regexes: Optional[Set[re.Pattern]]
) -> Iterable[Scannable]:
    """
    Parse the patch generated with `git show` (or `git diff`)

    If the patch represents a merge commit, then `patch` actually contains multiple
    commits, one per parent, because we call `git show` with the `-m` option to force it
    to generate one single-parent commit per parent. This makes later code simpler and
    ensures we see *all* the changes.
    """
    if exclusion_regexes is None:
        exclusion_regexes = set()

    for commit in patch.split("\0commit "):
        tokens = commit.split("\0diff ", 1)
        if len(tokens) == 1:
            # No diff, carry on to next commit
            continue
        header, rest = tokens

        names_and_modes = _parse_patch_header(header)

        diffs = re.split(r"^diff ", rest, flags=re.MULTILINE)
        for (filename, filemode), diff in zip(names_and_modes, diffs):
            if is_filepath_excluded(filename, exclusion_regexes):
                continue

            # extract document from diff: we must skip diff extended headers
            # (lines like "old mode 100644", "--- a/foo", "+++ b/foo"...)
            try:
                end_of_headers = diff.index("\n@@")
            except ValueError:
                # No content
                continue
            # +1 because we searched for the '\n'
            content = diff[end_of_headers + 1 :]

            yield StringScannable(filename, content, filemode=filemode)


def _parse_patch_header(header: str) -> Iterable[Tuple[str, Filemode]]:
    """
    Parse the header of a raw patch, generated with -z --raw
    """

    if header[0] == ":":
        # If the patch has been generated by `git diff` and not by `git show` then
        # there is no commit info and message, add a blank line to simulate commit info
        # otherwise the split below is going to skip the first file of the patch.
        header = "\n" + header

    # First item returned by split() contains commit info and message, skip it
    for line in _RX_HEADER_LINE_SEPARATOR.split(header)[1:]:
        yield _parse_patch_header_line(f":{line}")


class CommitInformation(NamedTuple):
    author: str
    email: str
    date: str

    @staticmethod
    def from_patch_header(patch: str) -> "CommitInformation":
        match = REGEX_HEADER_INFO.search(patch)
        assert match is not None
        return CommitInformation(**match.groupdict())


_UNKNOWN_COMMIT_INFORMATION = CommitInformation("unknown", "", "")


# Command line arguments passed to git to get parsable patches
_PATCH_COMMON_ARGS = [
    "--raw",  # shows a header with the files touched by the commit
    "-z",  # separate file names in the raw header with \0
    "--patch",  # force output of the diff (--raw disables it)
    "-m",  # split multi-parent (aka merge) commits into several one-parent commits
]


class Commit:
    """
    Commit represents a commit which is a list of commit files.
    """

    def __init__(
        self,
        sha: Optional[str],
        patch_parser: Callable[[], Iterable[Scannable]],
        info: Optional[CommitInformation] = None,
    ):
        """
        Internal constructor. Used by the `from_*` static methods and by some tests.
        Real code should use the `from_*` methods.
        """
        self.sha = sha
        self._patch_parser = patch_parser
        self.info = info or _UNKNOWN_COMMIT_INFORMATION

    @staticmethod
    def from_sha(
        sha: str,
        exclusion_regexes: Optional[Set[re.Pattern]] = None,
        cwd: Optional[Path] = None,
    ) -> "Commit":
        patch_header = git(["show", "--no-patch", sha], cwd=cwd)
        info = CommitInformation.from_patch_header(patch_header)

        def parser() -> Iterable[Scannable]:
            patch = git(["show", sha] + _PATCH_COMMON_ARGS, cwd=cwd)
            try:
                yield from _parse_patch(patch, exclusion_regexes)
            except Exception as exc:
                raise PatchParseError(f"Could not parse patch (sha: {sha}): {exc}")

        return Commit(sha, parser, info)

    @staticmethod
    def from_staged(
        exclusion_regexes: Optional[Set[re.Pattern]] = None, cwd: Optional[Path] = None
    ) -> "Commit":
        def parser() -> Iterable[Scannable]:
            patch = git(["diff", "--cached"] + _PATCH_COMMON_ARGS, cwd=cwd)
            try:
                yield from _parse_patch(patch, exclusion_regexes)
            except Exception as exc:
                raise PatchParseError(f"Could not parse patch: {exc}")

        return Commit(sha=None, patch_parser=parser)

    @staticmethod
    def from_patch(
        patch: str,
        exclusion_regexes: Optional[Set[re.Pattern]] = None,
    ) -> "Commit":
        """This one is for tests"""
        info = CommitInformation.from_patch_header(patch)

        def parser() -> Iterable[Scannable]:
            try:
                yield from _parse_patch(patch, exclusion_regexes)
            except Exception as exc:
                raise PatchParseError(f"Could not parse patch: {exc}")

        return Commit(sha=None, patch_parser=parser, info=info)

    @property
    def optional_header(self) -> str:
        """Return the formatted patch."""
        return (
            format_text(f"\ncommit {self.sha}\n", STYLE["commit_info"])
            + f"Author: {self.info.author} <{self.info.email}>\n"
            + f"Date: {self.info.date}\n"
        )

    @cached_property
    def files(self) -> List[Scannable]:
        return list(self.get_files())

    def get_files(self) -> Iterable[Scannable]:
        """
        Parse the patch into files and extract the changes for each one of them.
        """
        yield from self._patch_parser()

    def __repr__(self) -> str:
        return f"<Commit sha={self.sha}>"


def _parse_patch_header_line(line: str) -> Tuple[str, Filemode]:
    """
    Parse a file line in the raw patch header, returns a tuple of filename, filemode

    See https://github.com/git/git/blob/master/Documentation/diff-format.txt for details
    on the format.
    """

    prefix, name, *rest = line.rstrip("\0").split("\0")

    if rest:
        # If the line has a new name, we want to use it
        name = rest[0]

    # for a non-merge commit, prefix is
    # :old_perm new_perm old_sha new_sha status_and_score
    #
    # for a 2 parent commit, prefix is
    # ::old_perm1 old_perm2 new_perm old_sha1 old_sha2 new_sha status_and_score
    #
    # We can ignore most of it, because we only care about the status.
    #
    # status_and_score is one or more status letters, followed by an optional numerical
    # score. We can ignore the score, but we need to check the status letters.
    status = prefix.rsplit(" ", 1)[-1].rstrip("0123456789")

    # There is one status letter per commit parent. In the case of a non-merge commit
    # the situation is simple: there is only one letter.
    # In the case of a merge commit we must look at all letters: if one parent is marked
    # as D(eleted) and the other as M(odified) then we use MODIFY as filemode because
    # the end result contains modifications. To ensure this, the order of the `if` below
    # matters.

    if "M" in status:  # modify
        return name, Filemode.MODIFY
    elif "C" in status:  # copy
        return name, Filemode.NEW
    elif "A" in status:  # add
        return name, Filemode.NEW
    elif "T" in status:  # type change
        return name, Filemode.NEW
    elif "R" in status:  # rename
        return name, Filemode.RENAME
    elif "D" in status:  # delete
        return name, Filemode.DELETE
    else:
        raise ValueError(f"Can't parse header line {line}: unknown status {status}")
