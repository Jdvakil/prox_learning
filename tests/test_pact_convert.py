"""Path-only pact.py convert: dest mapping, no registry names."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts')]

from pact import convert_command, converted_destination, require_place_rows


def test_converted_destination_mirrors_data_tree():
    src = ROOT / 'data/pact_pick_n_place_v2/data/v12'
    dest = converted_destination(src)
    assert dest == ROOT / 'act_style_data/pact_pick_n_place_v2/data/v12'
    assert converted_destination(src / 'rows') == dest
    hallway = converted_destination(ROOT / 'data/pact_place_corridor_v5')
    assert hallway == ROOT / 'act_style_data/pact_place_corridor_v5'


def test_converted_destination_outside_data_uses_folder_name():
    dest = converted_destination(Path('/tmp/my_dump'))
    assert dest == ROOT / 'act_style_data/my_dump'
    dest_rows = converted_destination(Path('/tmp/my_dump/rows'))
    assert dest_rows == ROOT / 'act_style_data/my_dump'


def test_convert_command_uses_paths_not_registry_names():
    src = ROOT / 'data/pact_pick_n_place_v2/data/v12'
    command = convert_command(src)
    dest = command[command.index('--dst') + 1]
    assert dest.endswith('/data/v12')
    assert command[command.index('--task_name') + 1] == 'v12'
    custom = convert_command(src, ROOT / 'act_style_data/custom_v12')
    assert custom[custom.index('--dst') + 1].endswith('/custom_v12')
    assert custom[custom.index('--task_name') + 1] == 'custom_v12'


def test_require_place_rows_rejects_empty_or_obstacle_layout(tmp_path):
    with pytest.raises(ValueError, match='not a directory'):
        require_place_rows(tmp_path / 'missing')
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(ValueError, match='convert_obstacle_to_act'):
        require_place_rows(empty)
    house = empty / 'house_1'
    house.mkdir()
    (house / 'trajectories_batch_1_of_1.h5').write_bytes(b'')
    with pytest.raises(ValueError, match='convert_obstacle_to_act'):
        require_place_rows(empty)
    row = tmp_path / 'rows' / '000_abc'
    row.mkdir(parents=True)
    (row / 'trajectory.h5').write_bytes(b'')
    assert require_place_rows(tmp_path) == tmp_path


def test_convert_flag_is_an_alias_for_the_subcommand(tmp_path, capsys):
    from pact import main
    row = tmp_path / 'rows' / '000_abc'
    row.mkdir(parents=True)
    (row / 'trajectory.h5').write_bytes(b'')
    main(['--convert', str(tmp_path), '--dry-run'])
    printed = capsys.readouterr().out
    assert 'scripts.convert_pact_place_to_act' in printed
    assert str(tmp_path) in printed
