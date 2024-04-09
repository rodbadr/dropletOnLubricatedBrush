import hoomd
import hoomd.md
import gsd
import gsd.hoomd
import numpy
import time
import random
import sys
from math import sqrt

strFormat = '{:g}'

##################################################
### Set parameters and read initial conditions ###
##################################################

args = str(sys.argv[1])

paramInd = int(args)

paramSpace = numpy.genfromtxt('ParameterSpaceLiqSlabFull.csv', delimiter=',')

params = paramSpace[paramInd,:] # select specific parameter set

sig_b = 1.0; # Wall LJ length scale parameter
eps_b = 1.0; # Wall LJ potential depth
T = 1.0; # temperature for Langevin Thermostat
k = 20.0; # bond spring constant
r0 = 1.0; # bond equilibrium position if harmonic/maximum extension if FENE
rc = 1.0; # cutoff radius for dpd attraction
rd = 0.8; # cutoff radius for mdpd repulsion
gamma = 4.5 # drag coefficient
alpha = 15/(3.1415*rd**3) # coefficient by which HOOMD repulsion parameter B is different from papers

Nbrush1 = int(params[0]); # number of polymers in each direction. Total number of polymers is Nbrush1*Nbrush2
Nbrush2 = int(params[1]); # number of polymers in each direction. Total number of polymers is Nbrush1*Nbrush2
oliLen = int(params[2]); # number of monomers per oligomer chain
Noli = int(params[3]); # number of oligomers
NmonOli = int(oliLen*Noli); # total number of oligomer monomers
NbondOli = int((oliLen-1)*Noli) # number of oligomer bonds
Ntot = int(NmonOli); # total number of particles in simulation box

brushDist = float(params[4]); # distance between brush polymers

A = float(params[5]); # monomer-monomer interaction
B = float(params[6]); # density dependent repulsion strength. Needs to be the same for all species. (See no-go theorem in many-body dissipative particle dynamics)

mPoly = 1 # mass of polymer monomers

dens = float(params[7]);

nsteps = int(params[8]);
dt = 1e-3
Nmeas = int(params[9]);
meas_period = nsteps/Nmeas

animate = int(params[10]); # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory

sin = '_' + strFormat.format(Nbrush1) + '_' + strFormat.format(Nbrush2) + '_' + strFormat.format(brushDist) + '_' + strFormat.format(oliLen) + '_' + strFormat.format(Noli) + '_' + strFormat.format(-A) + '_' + strFormat.format(B);
sout =  '_' + strFormat.format(oliLen) + '_' + strFormat.format(-A) + '_' + strFormat.format(B);

seed1 = random.randint(1,999999999)
seed2 = random.randint(1,999999999)
seed3 = random.randint(1,999999999)

##################################
### Start HOOMD Initialization ###
##################################

context = hoomd.context.initialize("--mode=gpu");

t = gsd.hoomd.open(name="initialConditions/liqSlabFull" + sin + ".gsd",mode='rb')
tempSnap = t[0]

Lxin = t[0].configuration.box[0];
Lyin = t[0].configuration.box[1];
Lzin = t[0].configuration.box[2];

Lx = Lxin;
Ly = Lyin;
Lz = 2*Lzin;

WallPos = -Lz/2 + 4*sig_b

Ntot = tempSnap.particles.position[:,0].shape[0]

snapshot = hoomd.data.make_snapshot(N=Ntot,box=hoomd.data.boxdim(Lx=Lx, Ly=Ly, Lz=Lz),particle_types=['A','D'],bond_types=['polymer']);

snapshot.bonds.resize( NbondOli );

snapshot.particles.position[:] = tempSnap.particles.position[:]
snapshot.particles.position[:,2] = snapshot.particles.position[:,2] - snapshot.particles.position[:,2].min() + WallPos + 2*sig_b
snapshot.particles.typeid[:]=1;
snapshot.particles.mass[:]=mPoly;
snapshot.bonds.group[0:NbondOli] = tempSnap.bonds.group[0:NbondOli]

for i in range(Noli):

	shift = 0

	for j in range(oliLen-1):

		dz = tempSnap.particles.position[i*oliLen+j,2] - tempSnap.particles.position[i*oliLen+j+1,2]

		if (abs(dz)>Lzin/2):
			shift = 1

	if(shift == 1):

		tempz = numpy.zeros(oliLen)
		tempz[:] = snapshot.particles.position[i*oliLen:(i+1)*oliLen,2]
		tempz2 = numpy.zeros(oliLen)
		tempz2[:] = tempSnap.particles.position[i*oliLen:(i+1)*oliLen,2]
		tempz[tempz2<0] = tempz[tempz2<0] + Lzin
		snapshot.particles.position[i*oliLen:(i+1)*oliLen,2] = tempz[:];


system = hoomd.init.read_snapshot(snapshot);

nl = hoomd.md.nlist.cell();

### dpd attraction and thermostat setup
dpd = hoomd.md.pair.dpd(r_cut=rc, nlist=nl, kT=T, seed=seed1)
dpd.pair_coeff.set('A', 'A', A = A, gamma = gamma)
dpd.pair_coeff.set('A', 'D', A = A, gamma = gamma)
dpd.pair_coeff.set('D', 'D', A = A, gamma = gamma)


### mdpd density dependent repulsion setup
sqd = hoomd.md.pair.square_density(r_cut=rd, nlist=nl)
sqd.pair_coeff.set('A', 'A', B=B/alpha)
sqd.pair_coeff.set('A', 'D', B=B/alpha)
sqd.pair_coeff.set('D', 'D', B=B/alpha)

if(oliLen > 1):
	harmonic = hoomd.md.bond.harmonic()
	harmonic.bond_coeff.set('polymer',k=k,r0=r0)

#define wall surfaces and group them
surface1=hoomd.md.wall.plane(origin=(0.0, 0.0, WallPos), normal=(0.0, 0.0, 1.0), inside=True)
surface2=hoomd.md.wall.plane(origin=(0.0, 0.0, WallPos), normal=(0.0, 0.0, -1.0), inside=True)
walls = hoomd.md.wall.group([surface1,surface2])
#add walls
walllj=hoomd.md.wall.lj(walls, r_cut=sig_b*2.0**(1.0/6.0))
walllj.force_coeff.set('A', sigma=sig_b,epsilon=eps_b)
walllj.force_coeff.set('D', sigma=sig_b,epsilon=eps_b)

hoomd.md.integrate.mode_standard(dt=dt)

nl.reset_exclusions(exclusions = []);

all = hoomd.group.all();
integGroup = hoomd.group.type(name='integrable',type='D')

integrator = hoomd.md.integrate.nve(group=integGroup)


#####################
### equilibration ###
#####################

if (animate == 1):
	anim = hoomd.dump.gsd("trajectories/liqSlabTraj" + sout + ".gsd", group=all, overwrite=True, period=meas_period,dynamic=["momentum"]);

hoomd.run(nsteps);

hoomd.dump.gsd("initialConditions/liqSlab" + sout + ".gsd", group=all, overwrite=True, period=None);
