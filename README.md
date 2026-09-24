# neighbor\_list\_skin

This package defines a wrapper to neighbor-list calculation functions that uses a cached list of
neighbors until atoms have moved far enough to invalidate it.

A single class is defined, `neighbor_list_skin.NeighborListSkin`.  The constructor
takes as arguments a neighbor list function compatible with `ase.neighbors.neighbor_list`
(or `matscipy.neighbours.neighbour_list`) and a skin thickness, by default 1 Angstrom.
The resulting object can then be called as though it's the neighbor list function, with the
same arguments. If atoms have moved less than half the skin the same list of neighbors will
be used to compute the current returned values.  If they have moved farther, the underlying
neighbor list function will be called to calculate a new neighbor list.
