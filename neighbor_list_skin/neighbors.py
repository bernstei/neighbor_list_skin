from copy import deepcopy
import numpy as np

class NeighborListSkin:
    """Neighbor list calculator that avoids recalculation if atoms have not moved as
    far as a 'skin' distance.  Object constructor takes a neighbor list calculation
    function compatible with `ase.neighbors.neighbor_list` and a skin distance, and
    resulting object can the be called with same arguments as `ase.neighbors.neighbor_list`.

    """

    def __init__(self, neighbor_list, skin=1.0):
        """Initialize for a neighbor list with skin

        Parameters
        ----------
        skin: float, default 1.0
            extra padding to avoid recalculation if no atom has moved more
            than half skin, < 0 to disable
        neighbor_list: function
            ase.neighborlist.neighbor_list compatible function
        """
        # fixed at construction time
        self.neighbor_list = neighbor_list
        self.skin = skin

        self._reset()

    def _reset(self):
        """(re)initialize storage, forcing a reset
        """
        # last rebuild structural info
        self.positions_wrapped = None # wrapped positions at last rebuild
        self.positions_s_wrap = None # vectors that, when subtracted, shift scaled positions to origin image
        self.cell = None
        self.cell_inv = None # cell inverse
        self.numbers = None # atomic numbers
        self.cutoff = None # cutoff used
        self.max_shift = None # max shift along each lattice vector

        # last rebuild neighbor list info
        self.i = None # neighbor list froms
        self.j = None # neighbor list tos
        self.S = None # neighbor list shifts

        # info perhaps useful for calling code
        self.n_since_last_build = None # number of calls since last rebuild, 0 if rebuild just happened

    def __call__(self, quantities, atoms=None, cutoff=None, positions=None, cell=None, pbc=None, numbers=None, cell_origin=None):
        """get a neighbor list using skin to avoid recalculation if atoms haven't moved far

        See ase.neighborlist.neighbor_list or matscipy.neighbours.neighbour_list for details
        """
        if self.skin < 0.0:
            return self.neighbor_list(quantities, atoms=atoms, cutoff=cutoff, positions=positions,
                                      cell=cell, pbc=pbc, numbers=numbers, cell_origin=cell_origin)

        if atoms is None and any([positions is None, cell is None, numbers is None]):
            raise RuntimeError("atoms is None and missing one of [positions, cell, numbers]")
        if atoms is not None and any([positions is not None, cell is not None, numbers is not None]):
            raise RuntimeError("atoms is present and so is at least one of [positions, cell, numbers]")
        if cutoff is None:
            raise RuntimeError("cutoff is required")

        # extract numbers because they are needed for early rebuild test
        numbers = numbers if atoms is None else atoms.numbers
        if self.numbers is None or len(self.numbers) != len(numbers) or any(self.numbers != numbers) or self.cutoff != cutoff:
            # force reset
            self._reset()

        # extract other essential quantities from arguments
        cell = cell if atoms is None else atoms.cell
        positions = positions if atoms is None else atoms.positions

        # calculate scaled positions as well as shifts needed to wrap to origin image
        positions_s = cell.scaled_positions(positions)
        positions_s_wrap = np.floor(positions_s)

        rebuild = True
        if self.positions_wrapped is not None:
            # undo rotations
            # A maps from stored cell to current cell
            # cell = self.cell @ A
            # A = self.cell^-1 @ cell
            A = self.cell_inv @ cell
            U, _, Vt = np.linalg.svd(A)
            R = U @ Vt
            # non-rotation part of A
            P = R.T @ A
            # map stored cell to current cell without rotation
            cell_aligned = self.cell @ P
            # map current positions to "original" image, i.e. using saved wrapping displacement
            positions_aligned = (positions_s - self.positions_s_wrap) @ cell_aligned
            # could try to apply min image convention to this displacement, but not clear whether
            # to use saved or current cell
            delta = positions_aligned - self.positions_wrapped

            displ_cartesian = np.max(np.linalg.norm(delta, axis=1))
            # maximum relative displacement caused by cell shape change,
            delta_cell = cell - self.cell
            displ_cell = self.max_shift @ np.linalg.norm(delta_cell, axis=1)
            rebuild = (2.0 * displ_cartesian + displ_cell >= self.skin)

        if rebuild:
            # store values that will be compared to later
            self.cell = cell.copy()
            self.cell_inv = np.linalg.inv(self.cell)
            self.numbers = numbers.copy()
            self.cutoff = deepcopy(cutoff) # could be a dict
            if isinstance(self.cutoff, dict):
                use_cutoff = {k: v + self.skin for k, v in self.cutoff.items()}
            elif isinstance(self.cutoff, float):
                use_cutoff = self.cutoff + self.skin
            else:
                try:
                    use_cutoff = [c + self.skin / 2.0 for c in self.cutoff]
                except TypeError as exc:
                    raise TypeError(f"cutoff '{self.cutoff}' not a dict, float, or iterable of floats") from exc
            self.positions_s_wrap = positions_s_wrap.astype(int) # locally generated
            self.positions_wrapped = (positions_s - self.positions_s_wrap) @ self.cell

            pbc = pbc if atoms is None else atoms.pbc
            # always need ijS for our own purposes, add if needed
            inds = {}
            quantities_use = str(quantities)
            for quantity in 'ijS':
                if quantity not in quantities_use:
                    quantities_use += quantity
                inds[quantity] = quantities_use.index(quantity)
            # wrap positions so that returned shifts don't include drift to other periodic images,
            # to ensure that max_shift is physically meaningful
            positions_use = positions - positions_s_wrap @ cell
            # actually compute neighbor list
            quants_out = self.neighbor_list(quantities_use, cutoff=use_cutoff, positions=positions_use, cell=cell,
                                            pbc=pbc, numbers=numbers.astype(np.int32), cell_origin=cell_origin)
            # save ijS
            self.i = quants_out[inds['i']].copy()
            self.j = quants_out[inds['j']].copy()
            # store max raw shifts + 1 (for max displacement caused by cell shape change),
            # with shifts that come from _wrapped_ positions (which were passed to nn list)
            # so that irrelevant large displacements are ignored
            if len(self.i) > 0:
                self.max_shift = np.max(np.abs(quants_out[inds['S']]), axis=0) + 1
            else:
                self.max_shift = np.ones(3).astype(int)

            # reconstruct "true" S based on original, unwrapped, positions
            # we want d = p[j] - p[i] + S' @ cell
            # we have S such that d = (p[j] - p_wrap_s[j] @ cell) - (p[i] - p_wrap_s[i] @ cell) + S @ cell
            #                       = p[j] - p[i] + (p_wrap_s[i] - p_wrap_s[j] + S) @ cell
            # need to return S' = S + p_wrap_s[i] - p_wrap_s[j]
            # do this in-place so that quantity returned by function will be S'
            Sp = quants_out[inds['S']]
            Sp += self.positions_s_wrap[quants_out[inds['i']]]
            Sp -= self.positions_s_wrap[quants_out[inds['j']]]
            # store this S' which is consistent with true (non-wrapped) positions, since no-rebuild
            # case will reconstruct relative vector from _non-wrapped_ positions
            self.S = Sp.copy()

            # drop quantities that we added internally
            quants_out = quants_out[:len(quantities)]

            self.n_since_last_build = 0

            return quants_out

        self.n_since_last_build += 1

        # Benchmarking (ase md, skin 1.0) indicates that this part takes most of the time (~80%),
        # with remaining 20% dominated by actual nn list calculation.
        # Of this part, about 2/3 is the calculation of rv, and 1/3 the creation of quants_out

        # recalculate from stored i, j, S
        rv = None
        if 'd' in quantities or 'D' in quantities:
            rv = atoms.positions[self.j] - atoms.positions[self.i] + self.S @ atoms.cell
        quants_out = []
        for q in quantities:
            if q == 'i':
                quants_out.append(self.i.copy())
            elif q == 'j':
                quants_out.append(self.j.copy())
            elif q == 'S':
                quants_out.append(self.S.copy())
            elif q == 'd':
                quants_out.append(np.linalg.norm(rv, axis=1))
            elif q == 'D':
                quants_out.append(rv)

        return tuple(quants_out)
