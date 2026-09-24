import pytest

import numpy as np

from ase.atoms import Atoms
from neighbor_list_skin.neighbors import NeighborListSkin
try:
    from matscipy.neighbours import neighbour_list as neighbor_list_simple
except:
    from ase.neighborlist import neighbor_list as neighbor_list_simple

def _compare_lists (ijdS0, ijdS1, r_cut):
    """compare two neighbor lists and ensure that all entries within cutoff are
    identical"""

    # sort as tuples, which will result in lexical order on all the fields
    sort_order_0 = np.asarray(sorted(range(len(ijdS0[0])), key=lambda i: tuple([ijdS0[0][i], ijdS0[1][i]] + ijdS0[3][i].tolist())))
    sort_order_1 = np.asarray(sorted(range(len(ijdS1[0])), key=lambda i: tuple([ijdS1[0][i], ijdS1[1][i]] + ijdS1[3][i].tolist())))

    include_0 = ijdS0[2][sort_order_0] <= r_cut
    include_1 = ijdS1[2][sort_order_1] <= r_cut

    same = (sum(include_0) == sum(include_1))

    if same:
        for ind in range(len(ijdS0)):
            same &= np.allclose(ijdS0[ind][sort_order_0][include_0], ijdS1[ind][sort_order_1][include_1])

    return same

def test_basic_pos():
    # test minimal skin rebuild functionality, with long range (larger than cell),
    # and atoms at very large periodic images

    r_cut = 6.0

    ################################################################################
    # images in origin cell
    rng = np.random.default_rng(seed=5)
    atoms = Atoms('H5', cell=[5] * 3, scaled_positions=rng.uniform(size=(5, 3)), pbc=[True] * 3)
    dp = 0.001 * rng.normal(size=atoms.positions.shape)

    neighbor_list = NeighborListSkin(neighbor_list_simple, skin=0.1)
    n_rebuild = 0
    for i in range(100):
        ijdS_skin = neighbor_list('ijdS', atoms, r_cut)
        n_rebuild += neighbor_list.n_since_last_build == 0
        ijdS_msp = neighbor_list_simple('ijdS', atoms, r_cut)

        same = _compare_lists(ijdS_skin, ijdS_msp, r_cut)
        assert same

        atoms.positions += dp

    print("n_rebuild", n_rebuild)
    # some rebuilds, but not too many
    assert n_rebuild == 5

    ################################################################################
    # images in distant cells
    rng = np.random.default_rng(seed=5)
    atoms = Atoms('H5', cell=[5] * 3, scaled_positions=rng.uniform(size=(5, 3)), pbc=[True] * 3)
    dp = 0.001 * rng.normal(size=atoms.positions.shape)

    atoms.positions += rng.integers(-10, 10, size=atoms.positions.shape) @ atoms.cell

    neighbor_list = NeighborListSkin(neighbor_list_simple, skin=0.1)
    n_rebuild_imgs = 0
    for i in range(100):
        ijdS_skin = neighbor_list('ijdS', atoms, r_cut)
        n_rebuild_imgs += neighbor_list.n_since_last_build == 0
        ijdS_msp = neighbor_list_simple('ijdS', atoms, r_cut)

        same = _compare_lists(ijdS_skin, ijdS_msp, r_cut)
        assert same

        atoms.positions += dp

    print("n_rebuild_imgs", n_rebuild_imgs)
    # some rebuilds, but not too many
    assert n_rebuild_imgs == n_rebuild

def test_basic_cell():
    # test minimal skin rebuild functionality, with long range (larger than cell),
    # and atoms at very large periodic images
    r_cut = 6.0

    ################################################################################
    # images in origin cell
    rng = np.random.default_rng(seed=5)
    atoms = Atoms('H5', cell=[5] * 3, scaled_positions=rng.uniform(size=(5, 3)), pbc=[True] * 3)
    dc = 0.0001 * rng.normal(size=(3,3))

    neighbor_list = NeighborListSkin(neighbor_list_simple, skin=0.1)
    n_rebuild = 0
    for i in range(100):
        ijdS_skin = neighbor_list('ijdS', atoms, r_cut)
        n_rebuild += neighbor_list.n_since_last_build == 0
        ijdS_msp = neighbor_list_simple('ijdS', atoms, r_cut)
        same = _compare_lists(ijdS_skin, ijdS_msp, r_cut)
        assert same

        atoms.set_cell(atoms.cell @ (np.eye(3) + dc), True)

    print("n_rebuild", n_rebuild)
    # same number of rebuilds
    assert n_rebuild == 7

    ################################################################################
    # images in distant cells
    rng = np.random.default_rng(seed=5)
    atoms = Atoms('H5', cell=[5] * 3, scaled_positions=rng.uniform(size=(5, 3)), pbc=[True] * 3)
    dc = 0.0001 * rng.normal(size=(3,3))

    atoms.positions += rng.integers(-10, 10, size=atoms.positions.shape) @ atoms.cell

    neighbor_list = NeighborListSkin(neighbor_list_simple, skin=0.1)
    n_rebuild_imgs = 0
    for i in range(100):
        ijdS_skin = neighbor_list('ijdS', atoms, r_cut)
        n_rebuild_imgs += (neighbor_list.n_since_last_build == 0)
        ijdS_msp = neighbor_list_simple('ijdS', atoms, r_cut)
        same = _compare_lists(ijdS_skin, ijdS_msp, r_cut)
        assert same

        atoms.set_cell(atoms.cell @ (np.eye(3) + dc), True)

    print("n_rebuild_imgs", n_rebuild_imgs)
    # same number of rebuilds
    assert n_rebuild_imgs == n_rebuild


@pytest.mark.xfail(reason="current skin implementation does not ignore pbc jumps")
def test_pbc_jump():
    atoms = Atoms('H2', positions=[[0.2, 0, 0], [1.99, 0, 0]], cell=[2] * 3, pbc=[True] * 3)

    neighbor_list = NeighborListSkin(neighbor_list_simple, skin=0.1)

    ijS = neighbor_list('ijS', atoms, cutoff=0.25)
    assert neighbor_list.n_since_last_build == 0

    # pbc jump shouldn't trigger a rebuild
    atoms.positions[1] = [0.01, 0, 0]

    ijS = neighbor_list('ijS', atoms, cutoff=0.25)

    # make sure rebuild wasn't triggered
    assert not neighbor_list.n_since_last_build == 0

def test_skin_far_move_in():
    # test that atom that moves from outside to inside through cell shape changes is included
    # https://github.com/vicvaleeva/FourierD3/issues/9
    atoms = Atoms('H', positions=[[0, 0, 0]], cell=[100, 1, 100], pbc=[True] * 3)

    cutoff = 4.9

    eps = 0.001
    s0 = 4 * atoms.cell[1][1] - (4.0 / 5.0) * (cutoff - eps)
    s1 = 5 * atoms.cell[1][1] - cutoff
    skin = (s0 + s1) / 2

    neighbor_list = NeighborListSkin(neighbor_list_simple, skin=skin)

    # initial list
    _ = neighbor_list('ijdS', atoms, cutoff=cutoff)

    # modify cell to shift image (0, 5, 0) within cutoff
    Lp = (cutoff - eps) / 5.0
    atoms.cell[1, 1] = Lp

    ijdS = neighbor_list('ijdS', atoms, cutoff=cutoff)
    # should have rebuilt
    assert neighbor_list.n_since_last_build == 0

    # again, from scratch
    nl_scratch = NeighborListSkin(neighbor_list_simple, skin=skin)
    ijdS_scratch = nl_scratch('ijdS', atoms, cutoff=cutoff)

    same = _compare_lists(ijdS, ijdS_scratch, cutoff)
    assert same
