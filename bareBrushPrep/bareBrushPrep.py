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

args = str(sys.argv[1]) # read command line argument

paramInd = int(args) # change it to integer

paramSpace = numpy.genfromtxt('ParameterSpaceBareBrushPrep.csv', delimiter=',') # read parameter file

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

brushLen = int(params[0]); # number of monomers per brush polymer
Nbrush1 = int(params[1]); # number of polymers in each direction. Total number of polymers is Nbrush1*Nbrush2
Nbrush2 = int(params[2]); # number of polymers in each direction. Total number of polymers is Nbrush1*Nbrush2
NmonBrush = int(brushLen*Nbrush1*Nbrush2); # total number of monomers in brush
Ntot = int(NmonBrush); # total number of particles in simulation box
brushDist = float(params[3])*rc; # distance between brush polymers

### particle types are: A-grafted monomer, B-brush/gel monomers
Amm = float(params[4]); # monomer-monomer interaction
B = float(params[5]); # density dependent repulsion strength. Needs to be the same for all species. (See no-go theorem in many-body dissipative particle dynamics)

mPoly = 1 # mass of polymer monomers. mass of PDMS repeat unit: 74 g/mol
mLiq = 1 # mass of liquid particles. mass of water molecule: 18 g/mol

nsteps = int(params[6])
dt = 1e-3
Nmeas = int(params[7])
meas_period = nsteps/Nmeas
print(meas_period)
print(Nmeas)

animate = int(params[8]); # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory

SimID = int(params[9]);

seed1 = random.randint(1,999999999)
seed2 = random.randint(1,999999999)
seed3 = random.randint(1,999999999)


##################################
### Start HOOMD Initialization ###
##################################

context = hoomd.context.initialize("--mode=gpu");

Lx = brushDist*Nbrush1
Ly = brushDist*Nbrush2
Lz = brushLen + 4*rc

WallPos = -Lz/2 + rc # wall Position

sout = '_' + strFormat.format(brushLen) + '_' + strFormat.format(Nbrush1) + '_' + strFormat.format(Nbrush2) + '_' + strFormat.format(-Amm) + '_' + strFormat.format(brushDist) + '_' + strFormat.format(SimID);
brushHeight = rc*(brushLen - 1) # position of highest monomer in the initial condition for the substrate

zcoords = numpy.linspace(WallPos,WallPos + rc*brushHeight,brushLen) # initial z coordinates of brush monomers

snapshot = hoomd.data.make_snapshot(N=Ntot,box=hoomd.data.boxdim(Lx=Lx, Ly=Ly, Lz=Lz),particle_types=['A', 'B'],bond_types=['polymer']);
### particle types are: A-grafted monomer, B-brush/gel monomers

snapshot.bonds.resize( (brushLen-1)*Nbrush1*Nbrush2); # resize bond array to 2xNbonds

##### initialize polymer brush substrate

tempx = -Lx/2 + brushDist/2

for i in range(Nbrush1):
	tempy = -Ly/2 + brushDist/2
	for j in range(Nbrush2):
		ind = i*Nbrush2 + j ### index of the brush being initialized

		### position initialization
		snapshot.particles.position[ind*brushLen:(ind+1)*brushLen,0] = tempx
		snapshot.particles.position[ind*brushLen:(ind+1)*brushLen,1] = tempy
		snapshot.particles.position[ind*brushLen:(ind+1)*brushLen,2] = zcoords[:]

		### type initialization (0 is grafted, 1 is normal)
		snapshot.particles.typeid[ind*brushLen]=0;
		snapshot.particles.typeid[ind*brushLen+1:(ind+1)*brushLen]=1;

		### mass initialization
		snapshot.particles.mass[ind*brushLen]=mPoly;
		snapshot.particles.mass[ind*brushLen+1:(ind+1)*brushLen]=mPoly;

		### bond initialization
		if(brushLen > 1):
			snapshot.bonds.group[ind*(brushLen-1):(ind+1)*(brushLen-1),0] = numpy.linspace(ind*brushLen,(ind+1)*brushLen - 2,brushLen-1);
			snapshot.bonds.group[ind*(brushLen-1):(ind+1)*(brushLen-1),1] = numpy.linspace(ind*brushLen+1,(ind+1)*brushLen - 2 + 1,brushLen-1);

		tempy = tempy + brushDist

	tempx = tempx + brushDist


system = hoomd.init.read_snapshot(snapshot);

nl = hoomd.md.nlist.cell();

### dpd repulsion and thermostat setup
dpd = hoomd.md.pair.dpd(r_cut=rc, nlist=nl, kT=T, seed=seed1)
dpd.pair_coeff.set('A', 'A', A = Amm, gamma = gamma)
dpd.pair_coeff.set('A', 'B', A = Amm, gamma = gamma)
dpd.pair_coeff.set('B', 'B', A = Amm, gamma = gamma)

### mdpd density dependent attraction setup
sqd = hoomd.md.pair.square_density(r_cut=rd, nlist=nl)
sqd.pair_coeff.set('A', 'A', B=B/alpha)
sqd.pair_coeff.set('A', 'B', B=B/alpha)
sqd.pair_coeff.set('B', 'B', B=B/alpha)

hoomd.md.integrate.mode_standard(dt=dt)

nl.reset_exclusions(exclusions = []);

if(brushLen > 1 or oliLen > 1):
	harmonic = hoomd.md.bond.harmonic()
	harmonic.bond_coeff.set('polymer',k=k,r0=r0)


#define wall surfaces and group them
surface1=hoomd.md.wall.plane(origin=(0.0, 0.0, WallPos), normal=(0.0, 0.0, 1.0), inside=True)
surface2=hoomd.md.wall.plane(origin=(0.0, 0.0, WallPos), normal=(0.0, 0.0, -1.0), inside=True)
walls = hoomd.md.wall.group([surface1,surface2])
#add walls
walllj=hoomd.md.wall.lj(walls, r_cut=sig_b*2.0**(1.0/6.0))
walllj.force_coeff.set('A', sigma=sig_b,epsilon=0.0)
walllj.force_coeff.set('B', sigma=sig_b,epsilon=eps_b)

all = hoomd.group.all();
grafted = hoomd.group.type(name='grafted-monomers', type='A')
brush = hoomd.group.type(name='brush-monomers',type='B')

integrator = hoomd.md.integrate.nve(group=brush)
integrator.randomize_velocities(kT=T, seed=seed3)

# set the velocity of grafted monomers to 0

for p in grafted:
	p.velocity = (0,0,0)

#####################
### Equilibration ###
#####################

if(animate == 1):
	hoomd.dump.gsd("trajectories/bareBrushTraj" + sout + ".gsd", group=all, overwrite=True, period=meas_period,dynamic=["momentum"]);

hoomd.run(nsteps);

hoomd.dump.gsd("initialConditions/bareBrush" + sout + ".gsd", group=all, overwrite=True, period=None);
