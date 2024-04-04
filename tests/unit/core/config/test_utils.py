from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest

from ggshield.core.config.utils import (
    find_local_config_path,
    remove_common_dict_items,
    remove_url_trailing_slash,
    replace_dash_in_keys,
    update_from_other_instance,
)
from ggshield.utils.os import cd
from tests.repository import Repository


def test_replace_dash_in_keys():
    data = {"last-found-secrets": {"XXX"}}
    replace_dash_in_keys(data)
    assert data == {"last_found_secrets": {"XXX"}}


@dataclass
class ExampleSubConfig:
    an_int: Optional[int] = None
    an_str: Optional[str] = None


@dataclass
class ExampleConfig:
    path: Optional[str] = None
    backup: bool = True
    subconf: ExampleSubConfig = field(default_factory=ExampleSubConfig)
    a_list: List[str] = field(default_factory=list)
    a_set: Set[str] = field(default_factory=set)


def test_update_from_other_instance():
    """
    GIVEN two dataclass instances
    WHEN update_from_other_instance(dst_conf, src_conf) is called
    THEN dst_conf is updated from src_conf fields
    """

    dst_conf = ExampleConfig(path="dst_path")
    dst_conf.subconf.an_int = 1
    dst_conf.a_list = ["i1", "i2"]
    dst_conf.a_set = {"i1", "i2"}

    src_conf = ExampleConfig(backup=False)
    src_conf.subconf.an_str = "src_subconf_str"
    src_conf.a_list = ["i2", "i3"]
    src_conf.a_set = {"i2", "i3"}

    update_from_other_instance(dst_conf, src_conf)

    assert dst_conf.path == "dst_path"
    assert dst_conf.backup is False
    assert dst_conf.subconf.an_int == 1
    assert dst_conf.subconf.an_str == "src_subconf_str"
    assert dst_conf.a_list == ["i1", "i2", "i2", "i3"]
    assert dst_conf.a_set == {"i1", "i2", "i3"}


@pytest.mark.parametrize(
    ["src", "reference", "expected"],
    [
        pytest.param(
            {"modified": 2, "same": 3},
            {"modified": 1, "same": 3, "absent": 4},
            {"modified": 2},
            id="simple",
        ),
        pytest.param(
            {"lst": [3, 4]},
            {"lst": [1, 2]},
            {"lst": [3, 4]},
            id="replace-list",
        ),
        pytest.param(
            {"lst1": [], "lst2": ["a"]},
            {"lst1": [], "lst2": ["a"]},
            {},
            id="identical-lists-are-removed",
        ),
        pytest.param(
            {"lst": []},
            {"lst": ["a"]},
            {"lst": []},
            id="empty-list-not-removed-if-ref-not-empty",
        ),
        pytest.param(
            {"outer": {"inner1": "modified", "inner3": "same"}},
            {"outer": {"inner1": "original", "inner2": "absent", "inner3": "same"}},
            {"outer": {"inner1": "modified"}},
            id="nested-merge",
        ),
        pytest.param(
            {"outer": {"inner1": "same"}},
            {"outer": {"inner1": "same", "inner2": "absent"}},
            {},
            id="nested-dict-is-removed",
        ),
    ],
)
def test_remove_common_dict_items(
    src: Dict[str, Any], reference: Dict[str, Any], expected: Dict[str, Any]
):
    result = remove_common_dict_items(src, reference)
    assert result == expected


def test_remove_url_trailing_slash():
    result = remove_url_trailing_slash("https://dashboard.gitguardian.com/")
    assert result == "https://dashboard.gitguardian.com"


def test_find_config_in_root(tmp_path: Path):
    """
    GIVEN a repo with a config file in the root and a subdirectory
    WHEN trying to find the local config while inside the subdirectory
    THEN the config in the root is returned
    """
    Repository.create(tmp_path)

    config_path = tmp_path / ".gitguardian.yml"
    config_path.touch()

    dir_path = tmp_path / "dir"
    dir_path.mkdir()

    with cd(str(dir_path)):
        assert find_local_config_path() == config_path
