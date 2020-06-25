import numpy as np
import itertools
from tBG.hopping import filter_neig_list
from scipy.linalg.lapack import zheev
import copy
import matplotlib.collections as mc
from scipy.special import comb
from tBG.utils import rotate_on_vec

s0 = np.identity(2)
s1 = np.array([[0, 1],[ 1, 0]])
s2 = np.array([[0, -1j],[1j, 0]])
s3 = np.array([[1, 0],[0, -1]])

tau0 = np.identity(2)
tau1 = np.array([[0, 1],[ 1, 0]])
tau2 = np.array([[0, -1j],[1j, 0]])
tau3 = np.array([[1, 0],[0, -1]])

def plot_tiling(ax, coords, bonds, site_size, bond_width, site_color='black', bond_color='black', alpha=1.0):
    ax.scatter(coords[:,0], coords[:,1], s=site_size, c=site_color)
    line = mc.LineCollection([[coords[i][0], coords[i][1]] for i in bonds], bond_width, colors=bond_color)
    ax.add_collection(line)
    
def get_nth_nearest_neighbors(pmg_struct, nth_nearest=3, dist_prec=3):
    r_cut = 2.0
    while True:
        neigh_list = pmg_struct.get_neighbor_list(r_cut)
        cents, points, offsets, dists = neigh_list
        dists = np.round(dists, dist_prec)
        dist_shells = np.unique(dists)
        ind0 = 0
        if min(dist_shells)==0.:
            ind0 = 1
        try:
            ind1s = np.where(dists<=dist_shells[ind0+nth_nearest-1])[0]
            ind2s = np.where(dists>0.0)[0]
            inds = np.intersect1d(ind1s, ind2s)
            cents = cents[inds]
            points = points[inds]
            offsets = offsets[inds]
            dists = dists[inds]
            break
        except:
            r_cut = r_cut + 2.0
    cents, points, offsets, dists = filter_neig_list([cents, points, offsets, dists])
    dist_shells = np.unique(dists)
    neighs = []
    for dist_ith in dist_shells:
        inds = np.where(dists==dist_ith)[0]
        neighs_ith = np.append([cents[inds]], [points[inds]], axis=0).T
        neighs.append(neighs_ith)
    return dist_shells, neighs

def cos_ntheta(n, cos_theta, sin_theta):
    out = 0.
    for i in range(int(np.floor(n/2))+1):
        out = out + (-1)**i * comb(n,2*i) * cos_theta**(n-2*i) * sin_theta**(2*i)
    return out

def hop_func_prl_124_036803(r1s, r2s, dr, t1, t2, xi,  g, eta):
    fr_jk = np.exp(1-dr/xi)
    dx = r1s[:,0] - r2s[:,0]
    dy = r1s[:,1] - r2s[:,1]
    cos_theta_old = dx/dr
    sin_theta_old = dy/dr
    #cos_theta = 1/np.sqrt(2)*(cos_theta_old-sin_theta_old)
    #sin_theta = 1/np.sqrt(2)*(cos_theta_old+sin_theta_old)
    cos_theta = cos_theta_old
    sin_theta = sin_theta_old
    s3tau1 = np.matmul(np.kron(tau0, s3),np.kron(tau1, s0))
    s0tau2 = np.kron(tau2, s0)
    s0tau3 = np.kron(tau3, s0)
    s3tau1cos = np.kron(cos_theta, s3tau1)
    s0tau2sin = np.kron(sin_theta, s0tau2)
    s0tau3identity = np.kron([1.]*len(r1s), s0tau3)
    hop = -0.5*fr_jk*(1j*t1*(s3tau1cos +s0tau2sin)+t2*s0tau3identity)
    if g:    
        s1tau1 = np.matmul(np.kron(tau0, s1),np.kron(tau1, s0))
        cos_eta_theta = cos_ntheta(eta, cos_theta, sin_theta)
        hop = hop + 0.5*g*fr_jk*np.kron(cos_eta_theta, s1tau1)
    return hop

def onsite_energy(M, t2):
    s0tau3 = np.kron(tau3, s0)
    return (M+2*t2)*s0tau3


class TilingMethods:
    """
    This is the method collection shared by all quasi-periodic tilings
    """
    def cut_tiling(self, func):
        pass

    def cut_circle(self, r):
        rs = np.linalg.norm(self.coords, axis=1)
        inds = np.where(rs<=r)[0]
        self.coords = self.coords[inds]
        self.nsite = len(inds)

    def pymatgen_structure(self):
        from pymatgen.core.structure import Structure as pmg_struct
        xmin, ymin = np.min(self.coords, axis=0)
        xmax, ymax = np.max(self.coords, axis=0)
        coords = copy.deepcopy(self.coords)
        nsite = len(coords)
        coords[:,0] = coords[:,0] - xmin + 10.
        coords[:,1] = coords[:,1] - ymin + 10.
        coords = np.append(coords, [[0]]*nsite, axis=1)
        latt_vec = np.array([[xmax-xmin+20, 0, 0],[0, ymax-ymin+20, 0],  [0, 0, 100]])
        return pmg_struct(latt_vec, ['C']*nsite, coords, coords_are_cartesian=True)

    def hamiltonian(self, hop_func=hop_func_prl_124_036803, t1=1, t2=1, xi=1, M=-1, g=1, eta=2):
        ndim = self.nsite*4
        H = np.zeros([ndim,ndim], dtype=complex)
        def put_value(i,j,val):
            H[4*i:4*(i+1), 4*j:4*(j+1)] = val
        ## put onsite energies
        Eon = onsite_energy(M, t2)
        tmp = [put_value(i,i,Eon) for i in range(self.nsite)]
        del tmp
        ## put hopping energies
        pmg_st = self.pymatgen_structure()
        bond_lengths, pairs_bonds = get_nth_nearest_neighbors(pmg_st)
        for i in range(len(bond_lengths)):
            dr = bond_lengths[i]
            pairs = pairs_bonds[i]
            r1s = self.coords[pairs[:,0]]
            r2s = self.coords[pairs[:,1]]
            hops = hop_func(r1s, r2s, dr, t1, t2, xi, g, eta)
            tmp = [put_value(pairs[k][0], pairs[k][1], hops[:,4*k:4*(k+1)]) for k in range(len(pairs))]
            tmp = [put_value(pairs[k][1], pairs[k][0], np.matrix(hops[:,4*k:4*(k+1)]).H) for k in range(len(pairs))]
            del tmp
        return H
    
    def plot_tiling(self, bond_1th=False, bond_3th=False):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.scatter(self.coords[:,0], self.coords[:,1], s=3)
        pmg_st = self.pymatgen_structure()
        bonds, pairs = get_nth_nearest_neighbors(pmg_st)
        coords = self.coords

        plot_tiling(ax, coords, pairs[1], 1.0, 0.5, site_color='black', bond_color='black', alpha=1.0)
        if bond_1th:
            plot_tiling(ax, coords, pairs[0], 0.0, 0.8, site_color='white', bond_color='yellow', alpha=1.0)
        if bond_3th:
            plot_tiling(ax, coords, pairs[2], 0.0, 0.1, site_color='white', bond_color='red', alpha=1.0)
        ax.axis('equal')
        plt.show()

class AmmanBeenkerTiling(TilingMethods):
    def make_structure(self, N=50, l=1.0):
        from functools import reduce
        i1s = range(-N, N+1)
        j1s = range(-N, N+1)
        grids = tuple(itertools.product(i1s, j1s))
        def get_pts(i1,j1):
            A_dx = (i1+j1)/np.sqrt(2)
            A_dy = (i1-j1)/np.sqrt(2)
            B = 1/np.sqrt(2)-0.5
            # A_dx+B < i2 < A_dx and A_dx < i2 < A_dx-B ----- for dx condition
            i2s = np.concatenate((range(int(np.ceil(A_dx-B)),int(np.floor(A_dx))+1), range(int(np.ceil(A_dx)),int(np.floor(A_dx+B))+1)))
            # A_dy+B < j2 < A_dy and A_dy < j2 < A_dy-B ----- for dy condition
            j2s = np.concatenate((range(int(np.ceil(A_dy-B)),int(np.floor(A_dy))+1), range(int(np.ceil(A_dy)),int(np.floor(A_dy+B))+1)))
            if not (len(i2s) and len(j2s)):
                return np.empty([0,4], dtype=int)
            pts = np.array(list(itertools.product([i1],[j1], i2s, j2s))) 
            pz = pts[:,0]/np.sqrt(2) + pts[:,1]/np.sqrt(2) - pts[:,2]
            pw = pts[:,0]/np.sqrt(2) - pts[:,1]/np.sqrt(2) - pts[:,3]
            ##  for dx+dy condition
            inds = np.where(np.abs(pz)+np.abs(pw)<l*(1-np.sqrt(0.5)))[0]   ## dx+dy
            return pts[inds]
        pts = np.concatenate([get_pts(i[0], i[1]) for i in grids])
        px = pts[:,0]/np.sqrt(2) + pts[:,1]/np.sqrt(2) + pts[:,2]
        py = pts[:,0]/np.sqrt(2) - pts[:,1]/np.sqrt(2) + pts[:,3]
        coords = np.concatenate([[px/2], [py/2]]).T/2.91421356
        coords = np.unique(coords, axis=0) ## remove the repeated coords
        self.coords = rotate_on_vec(45., coords)
        self.nsite = len(self.coords)

class DodecagonalTiling(TilingMethods):
    def make_structure(self):
        pass
        

def diag(H, vec=1):
    vals, vecs, info = zheev(H, vec)
    if info:
        raise ValueError('zheev failed!')
    np.savez_compressed('EIGEN',vals=vals, vecs=vecs)

def plot_eigenvals(fname='EIGEN.npz'):
    data = np.load(fname)
    from matplotlib import pyplot as plt
    plt.scatter(range(len(data['vals'])), data['vals'])
    plt.ylim([-3,3])
    plt.show()

def plot_vec(coords, bonds, fname='EIGEN.npz', nth=0):
    data = np.load(fname)
    vec = data['vecs'][:,nth]
    chg = np.square(np.abs(vec))
    chg_ = np.array([np.sum(chg[4*i:4*(i+1)]) for i in range(int(len(vec)/4))])
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1,1)
    plot_tiling(ax, coords, bonds, chg_*2000, 1.0, site_color='red', bond_color='black', alpha=0.3)
    ax.axis('equal')
    plt.show()
    plt.close()
    
    
if __name__=='__main__':
    abt = AmmanBeenkerTiling()
    abt.make_structure(12)
    #abt.cut_circle(15)
    pmg_st = abt.pymatgen_structure()
    bond_lengths, pairs = get_nth_nearest_neighbors(pmg_st, nth_nearest=3, dist_prec=3)
    bonds = pairs[1]
    H = abt.hamiltonian(hop_func=hop_func_prl_124_036803, t1=1, t2=1, xi=1, M=-1, g=1, eta=2)
    diag(H,1)
    plot_eigenvals() 
    vals = np.load('EIGEN.npz')['vals']
    inds = np.where(abs(vals)<1.e-2)[0]
    for ind in inds:
        plot_vec(abt.coords, bonds, fname='EIGEN.npz', nth=ind)
