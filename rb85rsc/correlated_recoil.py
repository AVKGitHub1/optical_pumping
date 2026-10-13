"""Population recoil with one shared photon direction for all lattice axes.

For absorption paths a,b, integrate the product of the three displacement
cross products over a single unit vector u, rather than multiplying three
separately averaged kernels. Spherical emission channels remain resolved.
"""
import numpy as np

from . import motion as mo, polarization as pol


MAX_JOINT_STATES = 1024


def emission_sphere(b_local, n_polar, pattern):
    """Positive Gauss-Legendre x uniform-azimuth rule, in local lattice axes."""
    if not isinstance(n_polar, (int, np.integer)) or isinstance(n_polar, bool) or n_polar < 2:
        raise ValueError('correlated recoil quadrature requires an integer >= 2')
    c, w = mo.gauss_legendre(n_polar)
    phi = 2*np.pi*np.arange(2*n_polar)/(2*n_polar)
    cb, ph = np.meshgrid(c, phi, indexing='ij')
    directions = np.stack([np.sqrt(1-cb*cb)*np.cos(ph),
                           np.sqrt(1-cb*cb)*np.sin(ph), cb], axis=-1).reshape(-1, 3)
    # A B-aligned frame defined relative to the lattice preserves a rigid
    # apparatus rotation even at finite quadrature order.
    directions = directions @ pol.field_frame(pol.unit(b_local))
    weights = np.repeat(w*mo.emission_weight(c, 1., pattern)/len(phi), len(phi))
    return directions, weights


def joint_apply(kernel, populations):
    return (populations.reshape(populations.shape[0], -1) @ kernel.T).reshape(populations.shape)


class CorrelatedRecoil:
    """Build dense motional kernels once per site, with bounded work batches.

    Dense kernels scale as (Nx Ny Nz)^2. The explicit state limit prevents an
    unexpectedly large allocation; no truncation or fallback is applied.
    Construction caches are discarded after LocalExperiment setup.
    """

    def __init__(self, sites, b_local, n_polar, check_stop=lambda: None):
        self.sites = sites
        self.shape = tuple(s.N for s in sites)
        self.size = int(np.prod(self.shape))
        if len(sites) != 3 or self.size > MAX_JOINT_STATES:
            raise ValueError(f'correlated recoil supports three axes and at most {MAX_JOINT_STATES} joint motional states; got {self.shape}')
        self.b_local, self.n_polar = b_local, n_polar
        self.check_stop = check_stop
        self.directions, _ = emission_sphere(b_local, n_polar, 'isotropic')
        self._pairs = {}
        self._kernels = {}

    def _axis_pair(self, axis, ka, kb, kemit):
        key = (axis, ka, kb, kemit)
        if key not in self._pairs:
            site = self.sites[axis]
            projection = self.directions[:, axis]
            etas, indices = np.unique(np.r_[ka-kemit*projection, kb-kemit*projection]*site.x0,
                                      return_inverse=True)
            # Bulk overlaps avoid filling each persistent site's displacement
            # cache with thousands of sphere-specific values.
            overlaps = []
            for start in range(0, len(etas), 64):
                self.check_stop()
                batch = etas[start:start+64]
                if hasattr(site, 'displacements'):
                    overlaps.append(site.displacements(batch))
                else:
                    overlaps.append(np.asarray([site.displacement(eta) for eta in batch]))
            d = np.concatenate(overlaps)
            a, b = indices.reshape(2, -1)
            self._pairs[key] = np.real_if_close(d[a]*d[b].conj(), tol=1000)
        return self._pairs[key]

    def kernel(self, ka, kb, kemit, pattern):
        key = (tuple(ka), tuple(kb), kemit, pattern)
        if key not in self._kernels:
            reverse = (tuple(kb), tuple(ka), kemit, pattern)
            if reverse in self._kernels:
                return self._kernels[reverse].conj()
            ks = [self._axis_pair(i, ka[i], kb[i], kemit) for i in range(3)]
            _, weights = emission_sphere(self.b_local, self.n_polar, pattern)
            nx, ny, nz = self.shape
            # First group each axis's (final, initial) pair, then permute to
            # canonical K[nx',ny',nz', nx,ny,nz]. Chunking bounds intermediates.
            grouped = np.zeros((nx*nx*ny*ny, nz*nz), dtype=np.result_type(*ks))
            for start in range(0, len(weights), 64):
                self.check_stop()
                sl = slice(start, start+64)
                xy = (ks[0][sl].reshape(-1, nx*nx, 1)*ks[1][sl].reshape(-1, 1, ny*ny))
                grouped += xy.reshape(-1, nx*nx*ny*ny).T @ (weights[sl, None]*ks[2][sl].reshape(-1, nz*nz))
            kernel = grouped.reshape(nx, nx, ny, ny, nz, nz).transpose(0, 2, 4, 1, 3, 5).reshape(self.size, self.size)
            kernel.setflags(write=False)
            self._kernels[key] = kernel
        return self._kernels[key]


def recoil_description(model):
    if model == 'correlated':
        return ('Emission uses one shared 3D photon direction across lattice axes; '
                'spherical dipole channels remain resolved and recoil coherence transfer is omitted.')
    return 'Emission Cartesian components use independent dipole marginals; recoil coherence transfer omitted.'
