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

paramSpace = numpy.genfromtxt('ParameterSpaceOliMeltFull.csv', delimiter=',')

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

Nbrush1 = int(params[0]); # number of polymers in each direction. Total number of polymers is Nbrush1*Nbrush2
Nbrush2 = int(params[1]); # number of polymers in each direction. Total number of polymers is Nbrush1*Nbrush2
oliLen = int(params[2]); # number of monomers per oligomer chain
Noli = int(params[3]); # number of oligomers
NmonOli = int(oliLen*Noli); # total number of oligomer monomers
NbondOli = int((oliLen-1)*Noli) # number of oligomer bonds
Ntot = int(NmonOli); # total number of particles in simulation box

brushDist = float(params[4]); # distance between brush polymers

Amm = float(params[5]); # monomer-monomer interaction
B = float(params[6]); # density dependent repulsion strength. Needs to be the same for all species. (See no-go theorem in many-body dissipative particle dynamics)

dens = float(params[7]);

nsteps = int(params[8]);
dt = 1e-3
Nmeas = int(params[9]);
meas_period = nsteps/Nmeas

animate = int(params[10]); # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory


mPoly = 1 # mass of polymer monomers


Lx0 = oliLen + 10

Lx = Nbrush1*brushDist
Ly = Nbrush2*brushDist
Lz = Ntot/Lx/Ly/dens

# print("Lz = " + str(Lz))
#
# exit()

if ( Lz < 2*(rc+0.5) ): # to avoid self interaction across boundaries for low number of chains

	Lz = 2*(rc+0.5)

	Lx = numpy.sqrt(Ntot/dens/Lz)
	Ly = numpy.sqrt(Ntot/dens/Lz)

print(Ntot,Lx,Ly,Lz)

seed1 = random.randint(1,999999999)
seed2 = random.randint(1,999999999)
seed3 = random.randint(1,999999999)


##################################
### Start HOOMD Initialization ###
##################################

context = hoomd.context.initialize("--mode=gpu --notice-level=1");

############################
### Initialize Particles ###
############################

sout = '_' + strFormat.format(Nbrush1) + '_' + strFormat.format(Nbrush2) + '_' + strFormat.format(brushDist) + '_' + strFormat.format(oliLen) + '_' + strFormat.format(Noli) + '_' + strFormat.format(-Amm) + '_' + strFormat.format(B);

snapshot = hoomd.data.make_snapshot(N=Ntot,box=hoomd.data.boxdim(Lx=Lx0, Ly=Ly, Lz=Lz),particle_types=['A', 'B'],bond_types=['polymer']);

snapshot.bonds.resize( NbondOli );

##### initialize oligomers

xcoords2 = numpy.linspace(0,oliLen-1,oliLen) # initial x coordinates of oligomers

for i in range(Noli):

	tempx = random.random()*(Lx0-oliLen) - Lx0/2
	tempy = random.random()*Ly - Ly/2
	tempz = random.random()*Lz - Lz/2
	ind1 = i*oliLen
	ind2 = (i+1)*oliLen
	indBond = i*(oliLen-1)
	### position initialization
	if(oliLen > 1):
		snapshot.particles.position[ind1:ind2,0] = xcoords2[:] + tempx
	else:
		snapshot.particles.position[ind1,0] = tempx

	snapshot.particles.position[ind1:ind2,1] = tempy
	snapshot.particles.position[ind1:ind2,2] = tempz


	### type initialization
	snapshot.particles.typeid[ind1:ind2]=1;

	### mass initialization
	snapshot.particles.mass[ind1:ind2]=mPoly;

	### bond initialization
	if(oliLen > 1):
		snapshot.bonds.group[indBond:indBond+(oliLen-1),0] = numpy.linspace(ind1,ind2 - 2,oliLen-1);
		snapshot.bonds.group[indBond:indBond+(oliLen-1),1] = numpy.linspace(ind1+1,ind2 - 2 + 1,oliLen-1);

snapshot.particles.typeid[:]=1;

snapshot.particles.mass[:]=mPoly;
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


if(oliLen > 1):
	harmonic = hoomd.md.bond.harmonic()
	harmonic.bond_coeff.set('polymer',k=k,r0=r0)

hoomd.md.integrate.mode_standard(dt=dt)

nl.reset_exclusions(exclusions = []);

WallPos = -Lz/2+1.5;


# define wall surfaces and group them
# surface1=hoomd.md.wall.plane(origin=(0.0, 0.0, WallPos), normal=(0.0, 0.0, 1.0), inside=True)
# surface2=hoomd.md.wall.plane(origin=(0.0, 0.0, WallPos), normal=(0.0, 0.0, -1.0), inside=True)
# walls = hoomd.md.wall.group([surface1,surface2])
# # add walls
# walllj=hoomd.md.wall.lj(walls, r_cut=sig_b*2.0**(1.0/6.0))
# walllj.force_coeff.set('A', sigma=sig_b,epsilon=0.0)
# walllj.force_coeff.set('B', sigma=sig_b,epsilon=eps_b)


all = hoomd.group.all();
frozen = hoomd.group.type(name='frozen-monomers', type='A')
integGroup = hoomd.group.type(name='integrable-monomers',type='B')

integrator = hoomd.md.integrate.nve(group=integGroup)
integrator.randomize_velocities(kT=T/2, seed=seed3)
# set the velocity of grafted monomers to 0

# hoomd.md.force.constant(fvec=(0,0,-0.04),group=integGroup)

hoomd.update.box_resize(Lx=hoomd.variant.linear_interp([(0,system.box.Lx),(nsteps-500,Lx)]),
                        Ly=hoomd.variant.linear_interp([(0,system.box.Ly),(nsteps-500,Ly)]),
                        Lz=hoomd.variant.linear_interp([(0,system.box.Lz),(nsteps-500,Lz)]),
                        scale_particles=True)

for p in frozen:
	p.velocity = (0,0,0)

#####################
### Equilibration ###
#####################

if (animate == 1):
	hoomd.dump.gsd("trajectories/oliMeltFullTraj" + sout + ".gsd", group=all, overwrite=True, period=meas_period);

hoomd.run(nsteps);


hoomd.dump.gsd("initialConditions/oliMelt" + sout + ".gsd", group=all, overwrite=True, period=None);
