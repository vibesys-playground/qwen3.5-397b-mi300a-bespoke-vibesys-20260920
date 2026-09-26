"""host_numa: node-list parsing and that apply() is a no-op without flags."""

import host_numa


def test_parse_nodes() -> None:
    assert host_numa.parse_nodes("") == []
    assert host_numa.parse_nodes("1,2,3") == [1, 2, 3]
    assert host_numa.parse_nodes("3,1-2, 2") == [1, 2, 3]


def test_apply_without_flags_is_noop() -> None:
    assert host_numa.apply({}) == ""


def test_node_cpus(tmp_path) -> None:  # noqa: ANN001
    (tmp_path / "node1").mkdir()
    (tmp_path / "node1" / "cpulist").write_text("24-26,120\n")
    assert host_numa.node_cpus(1, root=tmp_path) == {24, 25, 26, 120}
