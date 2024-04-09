import hoomd
import hoomd.md
import gsd
import gsd.hoomd
import numpy
import time
import random
import math
import sys

strFormat = '{:g}'

##################################################
### Set parameters and read initial conditions ###
##################################################

args = str(sys.argv[1])

paramInd = int(args)

paramSpace = numpy.genfromtxt('ParameterSpaceDropOliBrush.csv', delimiter=',')

params = paramSpace[paramInd,:] # select specific parameter set

animate = 1; # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory

sig_b = 1.0; # Wall LJ length scale parameter
eps_b = 1.0; # Wall LJ potential depth
T = 1.0; # temperature for Langevin Thermostat
k = 20.0; # bond spring constant
r0 = 1.0; # bond equilibrium position if harmonic/maximum extension if FENE
rc = 1.0; # cutoff radius for dpd attraction
rd = 0.8; # cutoff radius for mdpd repulsion
gamma = 4.5 # drag coefficient
alpha = 15/(3.1415*rd**3) # coefficient by which HOOMD repulsion parameter B is different from papers
liqDens = 3.0 # density of the liquid for A = -40; B = 40
vapDens = 0.02 # vapor density for A = -40; B = 40

dropRadius = int(params[0])

brushLen = int(params[1]); # number of monomers per brush polymer
Nbrush1 = int(params[2]); # number of polymers in each direction. Total number of polymers is Nbrush1*Nbrush2
Nbrush2 = int(params[3]); # number of polymers in each direction. Total number of polymers is Nbrush1*Nbrush2
oliLen = int(params[4]); # number of monomers per oligomer chain
Noli = int(params[5]); # number of oligomers
brushDist = float(params[6]); # distance between brush polymers

NmonBrush = int(brushLen*Nbrush1*Nbrush2); # total number of monomers in brush
NmonOli = int(oliLen*Noli); # total number of oligomer monomers
NbondBrush = int((brushLen-1)*Nbrush1*Nbrush2) # number of brush bonds
NbondOli = int((oliLen-1)*Noli) # number of oligomer bonds

### particle types are: A-grafted monomer, B-brush/gel monomers, C-oligomer monomers, D-fluid particles
### mdpd interaction parameters in order: [AA,AB,AC,AD,BB,BC,BD,CC,CD,DD]
Amm = float(params[7]); # monomer-monomer interaction
All = float(params[8]); # fluid-fluid interaction
Aml = float(params[9]); # monomer-fluid interaction
B = float(params[10]); # density dependent repulsion strength. Needs to be the same for all species. (See no-go theorem in many-body dissipative particle dynamics)

mPoly = 1 # mass of polymer monomers
mLiq = 1 # mass of liquid particles

bottomEdgePos = float(params[11]); # position of the bottom of the liquid slab

g = float(params[12]); # gravitational constant

nsteps = int(params[13]);
dt = 1e-3
Nmeas = int(params[14]);
meas_period = nsteps/Nmeas

animate = int(params[15]); # 0 - Do note print equilibration trajectory; 1 - print equilibration trajectory

SimID = int(params[16]);

sinB = '_' + strFormat.format(brushLen) + '_' + strFormat.format(Nbrush1) + '_' + strFormat.format(Nbrush2) + '_' + strFormat.format(oliLen) +'_' + strFormat.format(Noli) +'_' + strFormat.format(brushDist) +'_' + strFormat.format(-Amm) + '_' + strFormat.format(B) + '_' + strFormat.format(SimID);

initCondDirB = "../oliBrushPrepFromTop/initialConditions/"

t = gsd.hoomd.open(name=initCondDirB + "oliBrushFinalConf" + sinB + ".gsd",mode='rb')

tempSnap = t[0]

brushHeight =  tempSnap.particles.position[0:NmonBrush,2].max()-tempSnap.particles.position[0:NmonBrush,2].min()# brush thickness in the initial condition for the substrate

LzB = tempSnap.configuration.box[2];

for i in range(Noli):

	shift = 0

	for j in range(oliLen-1):

		dz = tempSnap.particles.position[NmonBrush+i*oliLen+j,2] - tempSnap.particles.position[NmonBrush+i*oliLen+j+1,2]

		if (abs(dz)>LzB/2):
			shift = 1

	if(shift == 1):

		tempz = numpy.zeros(oliLen)
		tempz[:] = tempSnap.particles.position[NmonBrush+i*oliLen:NmonBrush+(i+1)*oliLen,2]
		tempz[tempz<0] = tempz[tempz<0] + LzB
		tempSnap.particles.position[NmonBrush+i*oliLen:NmonBrush+(i+1)*oliLen,2] = tempz[:];


sinP = '_' + strFormat.format(1) + '_' + strFormat.format(-All) + '_' + strFormat.format(B);

initCondDirP = "../slabPrep/initialConditions/"

tPend = gsd.hoomd.open(name=initCondDirP + "liqSlab" + sinP + ".gsd",mode='rb')

tempSnapPend = tPend[0]

LxP = tempSnapPend.configuration.box[0];
LyP = tempSnapPend.configuration.box[1];
LzP = tempSnapPend.configuration.box[2];

Nliq = 0;

LzP = 2*LzP

# pendantWallPos = (-LzP/2 + 4*sig_b)

for i in range(len(tempSnapPend.particles.typeid)):
	xtemp = tempSnapPend.particles.position[i,0]
	ytemp = tempSnapPend.particles.position[i,1]
	ztemp = tempSnapPend.particles.position[i,2] - bottomEdgePos
	rhotemp = numpy.sqrt(xtemp*xtemp + ytemp*ytemp + ztemp*ztemp)

	if ((rhotemp <=  dropRadius) and ( ztemp >  0)):
		Nliq = Nliq+1;

xP = tempSnapPend.particles.position[:,0]
yP = tempSnapPend.particles.position[:,1]
zP = tempSnapPend.particles.position[:,2] - bottomEdgePos

rhoP = numpy.sqrt(xP*xP + yP*yP + zP*zP)

mask = (rhoP <=  dropRadius)
mask2 = ( zP > 0 )# based on the wall position in pendant droplet prep
mask4 = mask & mask2

LiqPos = numpy.zeros((Nliq,3))
LiqPos[:] = tempSnapPend.particles.position[mask4]
LiqTypes = numpy.zeros(Nliq)
LiqTypes[:] = tempSnapPend.particles.typeid[mask4]
LiqTypes[LiqTypes==1]=2
LiqVel = numpy.zeros((Nliq,3))
LiqVel[:] = tempSnapPend.particles.velocity[mask4]

del xP,yP,zP,rhoP,mask,mask2,mask4

Ntot = int(NmonBrush + NmonOli + Nliq)

Lx = brushDist*Nbrush1
Ly = brushDist*Nbrush2
Lz = math.ceil(brushHeight + max(LzP,LzB) + sig_b + 30)

WallPos = -Lz/2 + rc # wall Position

seed1 = random.randint(1,999999999)
seed2 = random.randint(1,999999999)
seed3 = random.randint(1,999999999)

##################################
### Start HOOMD Initialization ###
##################################

context = hoomd.context.initialize("--mode=gpu");

snapshot = hoomd.data.make_snapshot(N=Ntot,box=hoomd.data.boxdim(Lx=Lx, Ly=Ly, Lz=Lz),particle_types=['A', 'B', 'C'],bond_types=['polymer']);

snapshot.bonds.resize( NbondBrush + NbondOli );

snapshot.particles.position[0:NmonBrush+NmonOli] = tempSnap.particles.position[0:NmonBrush+NmonOli]
snapshot.particles.position[0:NmonBrush+NmonOli,2] = snapshot.particles.position[0:NmonBrush+NmonOli,2] - snapshot.particles.position[0:NmonBrush+NmonOli,2].min() + WallPos
snapshot.particles.velocity[0:NmonBrush+NmonOli] = tempSnap.particles.velocity[0:NmonBrush+NmonOli]
snapshot.particles.typeid[0:NmonBrush+NmonOli]=tempSnap.particles.typeid[0:NmonBrush+NmonOli];
snapshot.particles.mass[0:NmonBrush+NmonOli]=mPoly;
snapshot.bonds.group[0:NbondBrush+NbondOli] = tempSnap.bonds.group[0:NbondBrush+NbondOli]

snapshot.particles.position[NmonBrush+NmonOli:NmonBrush+NmonOli + Nliq] = LiqPos
snapshot.particles.position[NmonBrush+NmonOli:NmonBrush+NmonOli + Nliq,2] = snapshot.particles.position[NmonBrush+NmonOli:NmonBrush+NmonOli + Nliq,2] - snapshot.particles.position[NmonBrush+NmonOli:NmonBrush+NmonOli + Nliq,2].min() -Lz/2 + brushHeight
snapshot.particles.velocity[NmonBrush+NmonOli:NmonBrush+NmonOli + Nliq] = LiqVel
snapshot.particles.typeid[NmonBrush+NmonOli:NmonBrush+NmonOli + Nliq]= LiqTypes;
snapshot.particles.mass[NmonBrush+NmonOli:NmonBrush+NmonOli + Nliq]=mLiq;

sout = '_' + strFormat.format(dropRadius) + '_' + strFormat.format(brushLen) + '_' + strFormat.format(Nbrush1) + '_' + strFormat.format(Nbrush2) + '_' + strFormat.format(oliLen) +'_' + strFormat.format(Noli) + '_' + strFormat.format(brushDist) + '_' + strFormat.format(-Amm) + '_' + strFormat.format(-All) + '_' + strFormat.format(-Aml) + '_' + strFormat.format(B) + '_' + strFormat.format(-g) + '_' + strFormat.format(SimID);

del t,tempSnap

system = hoomd.init.read_snapshot(snapshot);

nl = hoomd.md.nlist.tree();

### dpd repulsion and thermostat setup
dpd = hoomd.md.pair.dpd(r_cut=rc, nlist=nl, kT=T, seed=seed1)
dpd.pair_coeff.set('A', 'A', A = Amm, gamma = gamma)
dpd.pair_coeff.set('A', 'B', A = Amm, gamma = gamma)
dpd.pair_coeff.set('A', 'C', A = Aml, gamma = gamma)
dpd.pair_coeff.set('B', 'B', A = Amm, gamma = gamma)
dpd.pair_coeff.set('B', 'C', A = Aml, gamma = gamma)
dpd.pair_coeff.set('C', 'C', A = All, gamma = gamma)

### mdpd density dependent attraction setup
sqd = hoomd.md.pair.square_density(r_cut=rd, nlist=nl)
sqd.pair_coeff.set('A', 'A', B=B/alpha)
sqd.pair_coeff.set('A', 'B', B=B/alpha)
sqd.pair_coeff.set('A', 'C', B=B/alpha)
sqd.pair_coeff.set('B', 'B', B=B/alpha)
sqd.pair_coeff.set('B', 'C', B=B/alpha)
sqd.pair_coeff.set('C', 'C', B=B/alpha)


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
walllj.force_coeff.set('C', sigma=sig_b,epsilon=eps_b)

all = hoomd.group.all();
grafted = hoomd.group.type(name='grafted-monomers', type='A')
chains = hoomd.group.type(name='chains-monomers',type='B')
water = hoomd.group.type(name='oligomer-monomers',type='C')
integGroup = hoomd.group.union(name='temp',a=chains,b=water)

integrator = hoomd.md.integrate.nve(group=integGroup)

# set up constant (gravitational) force
if (abs(g) > 0):
	hoomd.md.force.constant(fvec=(0,0,g*mLiq),group=water)
	hoomd.md.force.constant(fvec=(0,0,g*mPoly),group=chains)

# set the velocity of grafted monomers to 0

for p in grafted:
	p.velocity = (0,0,0)

#####################
### Equilibration ###
#####################

if (animate == 1):
	hoomd.dump.gsd("trajectories/dropOliBrushTraj" + sout + ".gsd", group=all, overwrite=True, period=meas_period,dynamic=["momentum"]);

hoomd.run(nsteps);

hoomd.dump.gsd("initialConditions/dropOliBrushFinalConf" + sout + ".gsd", group=all, overwrite=True, period=None);
