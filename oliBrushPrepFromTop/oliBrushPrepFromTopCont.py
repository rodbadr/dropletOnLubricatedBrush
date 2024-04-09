import hoomd
import hoomd.md
import gsd
import gsd.hoomd
import numpy
import time
import random
import math
import sys

##################################################
### Set parameters and read initial conditions ###
##################################################

args = str(sys.argv[1])

paramInd = int(args)

paramSpace = numpy.genfromtxt('ParameterSpaceOliBrush.csv', delimiter=',')

params = paramSpace[paramInd,:] # select specific parameter set

strFormat = '{:g}'


sig_b = 1.0; # Wall LJ length scale parameter
eps_b = 1.0; # Wall LJ potential depth
T = 1.0; # temperature for Langevin Thermostat
k = 20.0; # bond spring constant
r0 = 1.0; # bond equilibrium position if harmonic/maximum extension if FENE
rc = 1.0; # cutoff radius for dpd attraction
rd = 0.8; # cutoff radius for mdpd repulsion
gamma = 4.5 # drag coefficient
alpha = 15/(3.1415*rd**3) # coefficient by which HOOMD repulsion parameter B is different from papers

### particle types are: A-grafted monomer, B-brush/gel monomers, C-oligomer monomers, D-fluid particles
### mdpd interaction parameters in order: [AA,AB,AC,AD,BB,BC,BD,CC,CD,DD]

brushLen = int(params[0]); # number of monomers per brush polymer
Nbrush1 = int(params[1]); # number of polymers in each direction. Total number of polymers is Nbrush1*Nbrush2
Nbrush2 = int(params[2]); # number of polymers in each direction. Total number of polymers is Nbrush1*Nbrush2
oliLen = int(params[3]); # number of monomers per oligomer chain
Noli = int(params[4]); # number of oligomers
brushDist = float(params[5])*rc; # distance between brush polymers


NmonBrush = int(brushLen*Nbrush1*Nbrush2); # total number of monomers in brush
NmonOli = int(oliLen*Noli); # total number of oligomer monomers
NbondBrush = int((brushLen-1)*Nbrush1*Nbrush2) # number of brush bonds
NbondOli = int((oliLen-1)*Noli) # number of oligomer bonds
Ntot = int(NmonBrush + NmonOli); # total number of particles in simulation box


Amm = float(params[6]); # monomer-monomer interaction
B = float(params[7]); # density dependent repulsion strength. Needs to be the same for all species. (See no-go theorem in many-body dissipative particle dynamics)

mPoly = 1 # mass of polymer monomers

nsteps = int(params[8]);
dt = 1e-3
Nmeas = int(params[9]);
meas_period = nsteps/Nmeas;

animate = int(params[10]); # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory

SimID = int(params[11]);


seed1 = random.randint(1,999999999)
seed2 = random.randint(1,999999999)
seed3 = random.randint(1,999999999)


##################################
### Start HOOMD Initialization ###
##################################

context = hoomd.context.initialize("--mode=gpu");

sin = '_' + strFormat.format(brushLen) + '_' + strFormat.format(Nbrush1) + '_' + strFormat.format(Nbrush2) + '_' + strFormat.format(oliLen) +'_' + strFormat.format(Noli) +'_' + strFormat.format(brushDist) + '_' + strFormat.format(-Amm) + '_' + strFormat.format(B) + '_' + strFormat.format(SimID);

snapshot = hoomd.data.gsd_snapshot("initialConditions/oliBrushFinalConf" + sin + ".gsd", frame=0)

Lz = snapshot.box.Lz;

WallPos = -Lz/2 + rc # wall Position

sout = '_' + strFormat.format(brushLen) + '_' + strFormat.format(Nbrush1) + '_' + strFormat.format(Nbrush2) + '_' + strFormat.format(oliLen) +'_' + strFormat.format(Noli) +'_' + strFormat.format(brushDist) + '_' + strFormat.format(-Amm) + '_' + strFormat.format(B) + '_' + strFormat.format(SimID);

#######################

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
integGroup = hoomd.group.type(name='brush-monomers',type='B')

integrator = hoomd.md.integrate.nve(group=integGroup)

# set the velocity of grafted monomers to 0

for p in grafted:
	p.velocity = (0,0,0)

#####################
### Equilibration ###
#####################

if (animate == 1):
	hoomd.dump.gsd("trajectories/oliBrushTraj" + sout + ".gsd", group=all, overwrite=True, period=meas_period,dynamic=["momentum"]);

hoomd.run(nsteps);

hoomd.dump.gsd("initialConditions/oliBrushFinalConf" + sout + ".gsd", group=all, overwrite=True, period=None);
