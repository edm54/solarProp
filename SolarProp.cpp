/*

Make global variable for displace counter. Once x_width have been displaced, drop x_width down 
consider array to keep track of displaced
use delta n, but keep track regardless of iteration--make global var
*/


/*
https://stackoverflow.com/questions/11095309/openmp-set-num-threads-is-not-working
Program:  pulse.cpp

This is a 2D soft-particle discrete element simulation.  The simulation is set
up simulating a two-dimensional box with vertical walls subjected to vertical
sinusoidal vibrations.

The program was originally written by Carl Wassgren.  I have modified it to allow for
the input and detection of pressure waves in the direction perpendicular to the
direction of shaking.

This program was then modified to track the temperatures of particles and stimulate heat transfer.
This program is currently set up for the walls to have a temperature of 1.0, and the particles
are initialized with a temperature value of 0. The particles then change temperature when
colliding with the wall or other particles.

This program is dimensionless with the variables made so through a combination of the
mean particle diameter, the mean particle mass and the speed of sound through the bulk
particle material.

This incarnation of the program inputs a pressure wave (pulsed or continuous excitation)
of a given width or frequency and amplitude from the left boundary.  The wave is detected
at the right boundary.

Brillantov, et. al. (1996)
Dissipation in the normal direction is included using the viscoelastic model presented by

*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <string.h>
#include <iostream> 
#include <fstream>
#include<string>
#include <iomanip>
#include <openacc.h>
#include <sys/time.h>
#include <algorithm>
#include <vector> 
#include <unordered_set>

using namespace std;

#define PI (3.14159265)

// gravity
#define g (9.81)

// ratio of overlap to the smallest particle radius at which a warning is printed
#define MAX_OVERLAP (0.03)

// maximum number of contacts per particle
#define CONTACT_MAX (8)

// function to tell the sign of a variable
#define sgn(x) ((x)>=0.0 ? 1.0 : -1.0)

// input data file name
//#define filename_in ("NDpulse.in")

// pressure output file
#define pfilename_out ("pressure.out")

// Experimental parameters output file
#define conditions_out ("conditions.out")

#define alpha (386/(8960*385)) //k/PC 

#define inputD .005

#define Co 4600

//height of heated section
#define heatedLength 20

//#define dtConstant alpha/(inputD*Co)
//#define dtConstant 0.01
#define dtConstant 0.001

//Pretto's number, taken at 1000K air from page 995 of "principles of heat and mass transfer, Bergman"
#define Pr .726

//constant taken taken at 1000K air from page 995 of "principles of heat and mass transfer, Bergman"
#define v .0001219

#define windowHeight 40

#define k_ratio (100/401)

#define airTemp 1


/*
The data structure for all particles in the simulations:
num: the particle identifying number (between 1 and N where N is the total
number of particles
cellx, celly: the in which the particle is located
contact[]: the numbers of the particles that are currently in contact with
this particle
x, y, thetaz: the translational and rotational positions of the particle
xdot, ydot, thetazdot: the trans. and rot. velocities of the particle
Fx, Fy, Tz: the forces and torques acting on the particle
r, m, I: the particle's radius, mass, and moment of inertia
deltas[]: the tangential displacement for each particle in contact with
this particle
*prev, *next: pointers to the previous and next particles in the cell in
which the current particle is located

particleTemperature: This value represents the temperature of the particle, is arbitrarily set to 20 Celcius
youngModulus:The particle's Young's Modulus (E)
poissonRatio: The particle's Poisson's Ratio (V)
thermalConductivity: This value is Ks, kept as a constant since the Ks of copper remains relatively constant despite large temperature change
This value should be adjusted each timestep, especially if using other materials
volume: each particle's volume is stored(not necessary but makes thermal capacity easier)
density:Kg/m^3
normalForces[] stores the normal force between two particles
coincides with contact[] so that for particle[i].contact[j]
the normal force between j and i is stored in particle[i].normalForce[j]

*/
struct state
{
	short int
		num,
		cellx,
		celly,
		contact[CONTACT_MAX + 1];
	double
		x,
		y,
		thetaz,
		xdot,
		ydot,
		thetazdot,
		Fx,
		Fy,
		Tz,
		r,
		m,
		I,
		particleTemperature,
		nextTemperature,
		nextRadius,
		poissonRatio,
		youngModulus,
		thermalConductivity,
		density,
		specificHeat,
		thermalExpansionCoefficient,
		volume,
		qSum, //holds the value for sum(Q*i/r*i^3
		qSum1,//first term
		qSum2,//second term
		qSum1Sum,
		qSum2Sum,
		deltas[CONTACT_MAX + 1],
		yToInsert,
		xToInsert;
	bool rain;
	//normalForces[CONTACT_MAX + 1],
//20 steps in the simulation
	//temperatures[100];
	struct state *prev, *next;
} *particle;

// the structure for the cell pointers
struct element
{
	struct state *next;
} **cell;




/*
Define global variables:
N: total number of particles
x_wall_flag: the flag that tells whether wall boundaries (1) or periodic
boundaries (0) are being used
cellmax_x, cellmax_y: the total number of cells in the environment
wall_num: the total number of wall boundaries
Gamma: the oscillation acceleration amplitude
a: the oscillation amplitude
w: the oscillation radian frequency
gx, gy: the gravity acceleration components
deltat: the simulation time step
t: time
tmax: the time at which the simulation will end
x_width, y_width: the rectangular container dimensions
k_pn, nu_pn: the particle/particle nornal contact spring constant and
dashpot coefficient
k_ps, mu_p: the particle/particle tangential contact spring constant and
coefficient of friction
k_wn, nu_wn: the particle/wall normal contact spring constant and
dashpot coefficient
k_ws, mu_w: the particle/wall tangential contact spring constant and
coefficient of friction
cellsize: the length of the square cell
dmax: the maximum particle diameter
rmin: the minimum particle radius
deltat_samp_recordstate: the time between successive recordings of the
particle states
t_samp_recordstates: the time at which the next recording of the particle
states will be made
*/
int
N,n,
x_wall_flag,
cellmax_x,
cellmax_y,
wall_num,
firstParticle;
double
Gamma,
a,
w,
GammaH,
InputAmp,
InputWidthOrFreq,
gx,
gy,
Fr,
deltat,
t,
t0,
tmax,
x_width,
y_width,
k_pn,
nu_pn,
A_pn,
k_ps,
mu_p,
k_wn,
nu_wn,
k_ws,
mu_w,
k2p,
k2w,
cellsize,
dmax,
dbar,
rmin,
deltat_samp_recordstates,
t_samp_recordstates,
t_psamp_recordstates,
deltat_psamp_recordstates,
Pin,
Pin2,
Pout,
Pout2,
SensorTop,
SensorBottom,
SensorTop2,
SensorBottom2,
SensorSize,
SensorLoc,
SensorLoc2,
FreeSurface,
v0max,
initialTemp, //To used for t*
wallTemp,
qSum1Sum,
qSum2Sum,//T1 used for t*
c0,
xToInsert,
yToInsert,
particlesInserted,
timeAfterInsert;
bool up;
std::vector<int> particlesToInsert;


int main(int argc, char **argv)
{
	void makeQZero1(void);
	void read_data(char Directory[8]);
	void calculate_forces(FILE *outfile);
	void integrate_eqns(void);
	void record_states(FILE *outfile);
	void RecordStatesAll(FILE *outfile, char Directory[8]);
	void LoadStates(char Directory[8]);
	void MovePiston(int ExcitationFlag);  // 1 for pulse, other (i.e. 0) for continuous
	FILE *outfile;
	FILE *poutfile;
	int
		x = 0,
		i = 0;
	extern int N,firstParticle;
	extern double
		t,
		tmax,
		w,
		t_samp_recordstates,
		deltat_samp_recordstates,
		t_psamp_recordstates,
		deltat_psamp_recordstates,
		Pout,
		Pout2,
		Pin,
		y_width,
		FreeSurface,
		InputWidthOrFreq,
		InputAmp,
		initialTemp, //To used for t*
		wallTemp, //T1 used for t*
		SensorSize,
		xToInsert,
		yToInsert;
	double
		//	FreeSurfOld,
		OriginalFrequency,
		OriginalAmp;
	extern struct state *particle;
	const char
		POut[] = "pressure.out",
		OOut[] = "states.out";
	char
		PFileName[20],
		OutFileName[20],
		Directory[8];
	//FILE * outfile;
	clock_t
		start,
		end;
	error_t
		err;
	initialTemp = 0.0;
	wallTemp = 1.0;
	start = clock();
	strcpy(OutFileName, "states.out");
	outfile = fopen(OutFileName, "wb");
	printf("The state outputs file was opened.\n");
	strcpy(PFileName, "pressure.out");
	poutfile = fopen64(PFileName, "wb");
	printf("The pressure output file was opened.\n");

	// goto the subroutine that reads in the input data
	read_data(Directory);
	OriginalFrequency = InputWidthOrFreq;
	OriginalAmp = InputAmp;
	firstParticle = 1;
	particlesInserted = 0;
	xToInsert = 1;
	yToInsert = 0;
	struct timeval start_time, stop_time, elapsed_time, current_time;  // timers
	gettimeofday(&start_time, NULL); // Unix timer
	// start the simulation
	t = 0.0;
	record_states(outfile);
	start = clock();
	tmax = tmax * 2.0*PI / w;
	up = true;
	
	do
	{
		//use force as guide to run temperatures
		calculate_forces(outfile);
		if (t >= t_samp_recordstates)
		{
			end = clock();
			gettimeofday(&current_time, NULL);
			timersub(&current_time, &start_time, &elapsed_time);

			printf("states: cycles = %.3f (%.1f%% -- %.3f Hours) Ymax = %.3f\n",
				w*t / (2.0*PI),
				100.0*t / tmax,
				//(end - start) / (double(CLOCKS_PER_SEC) * 3600.0),
				((elapsed_time.tv_sec + elapsed_time.tv_usec / 1000000.0) / 3600),
				FreeSurface);
			t_samp_recordstates += deltat_samp_recordstates;
			fprintf(poutfile, "%.5e %e %e %e %e %e\n",
				t,
				Pin,
				Pin2,
				Pout,
				Pout2,
				particle[N + 3].xdot);

			//Used to calculate the temperature change of the particles, should run every timestep
			//only saves every several time steps
			//temperatureChange(outfile);
			record_states(outfile);
			//	makeQZero1();
		}
		if (t >= t_psamp_recordstates)
		{
			fprintf(poutfile, "%.5e %e %e %e %e %e\n",
				t,
				Pin,
				Pin2,
				Pout,
				Pout2,
				particle[N + 3].x);
			t_psamp_recordstates += deltat_psamp_recordstates;
		}
		integrate_eqns();
		timeAfterInsert += 1;

	} while (t < tmax);
	calculate_forces(outfile);
	record_states(outfile);
	printf("cycles = %.3f (%.1f%%)\n", w*t / (2.0*PI), 100.0*t / tmax);

	// close the output file
	fclose(outfile);
	fclose(poutfile);
	end = clock();
	printf("This took %f hours.\n", (end - start) / (double(CLOCKS_PER_SEC) * 3600.0));
	gettimeofday(&stop_time, NULL);
	timersub(&stop_time, &start_time, &elapsed_time); // Unix time subtract routine
	printf("Total time was %0.2f hours.\n", (elapsed_time.tv_sec + elapsed_time.tv_usec / 1000000.0) / 3600);
	return 0;
}



void makeQZero(void)
{
	int nthreads, tid;
	///#pragma acc kernels
	#pragma omp parallel for private(tid) num_threads(4)
	for (int i = 0;i <= N;i++)
	{
		particle[i].qSum = 0;
		particle[i].qSum1 = 0;
		particle[i].qSum2 = 0;
	}
}
void makeQZero1(void)
{
	int nthreads, tid;
	///#pragma acc kernels
	#pragma omp parallel for private(tid) num_threads(4)
	for (int i = 0;i <= N;i++)
	{
		particle[i].qSum = 0;
		particle[i].qSum1 = 0;
		particle[i].qSum2 = 0;
	}
}
void calculate_forces(FILE *outfile)
/*
This routine initializes the particle forces with the gravity force and
then determines which particles could be in contact.  It calls other routines
to actually do the calculations for the forces
*/
{
	void x_wall_forces(int i, int j);
	void y_wall_forces(int i, int j);
	void particle_forces(int i, int j, double pj_dx);
	void changeTemperatures(void);
	void makeQZero(void);
	void allParticlesConvection(void);

	int
		i,
		cellx,
		celly,
		x,
		y;
	struct state *particlej;
	extern int
		N,
		cellmax_x,
		cellmax_y,
		x_wall_flag;
	extern double
		gx,
		gy,
		FreeSurface,
		Pin,
		Pin2,
		Pout,
		Pout2,
		SensorSize;
	extern struct state *particle;
	extern struct element **cell;
	Pout = 0.0;
	Pout2 = 0.0;
	Pin = 0.0;
	Pin2 = 0.0;
	FreeSurface = 0.0;
	makeQZero();
	int tid;
	// initialize particle forces with gravity force

	#pragma omp parallel for private(tid) num_threads(4)
	for (i = 1;i <= N;i++)
	{
		particle[i].Fx = gx * particle[i].m;
		particle[i].Fy = gy * particle[i].m;
		particle[i].Tz = 0.0;
		if (particle[i].y > FreeSurface)
			//#pragma omp critical
			#pragma acc critical
			FreeSurface = particle[i].y;
	}
	// check for collision with the bottom wall
	celly = particle[N + 1].celly;
	for (y = celly; y <= celly + 1;y++)
	{
		for (x = 1;x <= cellmax_x - 1;x++)
		{
			if ((particlej = cell[x][y].next) != NULL)
			{
				do
				{
					y_wall_forces(particlej->num, N + 1);
				} while ((particlej = particlej->next) != NULL);
			}
		}
	}

	// check for collision with the top wall
	celly = particle[N + 2].celly;
	for (y = celly - 1; y <= celly;y++)
	{
		for (x = 1;x <= cellmax_x - 1;x++)
		{
			if ((particlej = cell[x][y].next) != NULL)
			{
				do
				{
					y_wall_forces(particlej->num, N + 2);
				} while ((particlej = particlej->next) != NULL);
			}
		}
	}
	if (x_wall_flag == 1)  // wall boundaries
	{
		// check for collisions with the left wall
		cellx = particle[N + 3].cellx;
		for (x = cellx;x <= cellx + 1;x++)
		{
			for (y = particle[N + 1].celly;y <= particle[N + 2].celly;y++)
			{
				if ((particlej = cell[x][y].next) != NULL)
				{
					do
					{
						x_wall_forces(particlej->num, N + 3);
					} while ((particlej = particlej->next) != NULL);
				}
			}
		}

		// check for collisions with the right wall
		cellx = particle[N + 4].cellx;
		for (x = cellx - 1;x <= cellx;x++)
		{
			for (y = particle[N + 1].celly;y <= particle[N + 2].celly;y++)
			{
				if ((particlej = cell[x][y].next) != NULL)
				{
					do
					{
						x_wall_forces(particlej->num, N + 4);
					} while ((particlej = particlej->next) != NULL);
				}
			}
		}
	}

	// check for interparticle collisions
	//was N-1
	int nthreads;
	#pragma omp parallel for private(tid,cellx,celly,particlej,x,y,x_wall_flag) num_threads(4) //stacksize(10000)
	for (int i = 1;i <= N;i++)
	{
		cellx = particle[i].cellx;
		celly = particle[i].celly;
		// Check for collisions with cells around and including current cell
		for (x = cellx - 1;x <= cellx + 1;x++)
		{
			for (y = celly - 1;y <= celly + 1;y++)
			{
				if ((particlej = cell[x][y].next) != NULL)
				{
					do
					{
						if (particlej->num > i)
							particle_forces(particle[i].num, particlej->num, 0.0);
					} while ((particlej = particlej->next) != NULL);
				}
			}
		}
		if (x_wall_flag == 0)  // periodic boundaries
		{
			if (cellx == 1)
			{
				for (y = celly - 1;y <= celly + 1;y++)
				{
					if ((particlej = cell[cellmax_x - 1][y].next) != NULL)
					{
						do
						{
							if (particlej->num > i)
								particle_forces(particle[i].num, particlej->num, -x_width);
						} while ((particlej = particlej->next) != NULL);
					}
				}
			}
			if (cellx == cellmax_x - 1)
			{
				for (y = celly - 1;y <= celly + 1;y++)
				{
					if ((particlej = cell[1][y].next) != NULL)
					{
						do
						{
							if (particlej->num > i)
								particle_forces(particle[i].num, particlej->num, x_width);
						} while ((particlej = particlej->next) != NULL);
					}
				}
			}
		}
	}

	allParticlesConvection();
	//for each particle, we want to find the Q* for each particle interacting with a particle, then adjust heat. Next we want to find dT/dt
	changeTemperatures();
	Pout = Pout / SensorSize;
	Pout2 = Pout2 / SensorSize;
	Pin = Pin / SensorSize;
	Pin2 = Pin2 / SensorSize;
	makeQZero();
}


void allParticlesConvection(void)
{
	float calculateConvection(int i);
	for (int i = 0; i <= N; i++)
	{
		particle[i].qSum += calculateConvection(i);
	}
}


void changeTemperatures(void)
{
	int tid;

	#pragma omp parallel for private(tid) num_threads(4)
	for (int i = 0;i <= N;i++)
	{
		particle[i].particleTemperature += (3 * particle[i].qSum*deltat*dtConstant) / (4 * PI*pow(0.5, 3));

		if (particle[i].particleTemperature > 1 || particle[i].particleTemperature < 0)
			printf("particleTemperature %.3f \n", particle[i].particleTemperature);

		if (particle[i].particleTemperature  > 0)
			printf("particleTemperature %.3f \n", particle[i].particleTemperature);

	}
}
double deltaT(int i, int j)
{
	double
		T1, T2;
	T1 = (particle[i].particleTemperature - initialTemp) / (wallTemp - initialTemp);
	T2 = (particle[j].particleTemperature - initialTemp) / (wallTemp - initialTemp);
	return T2 - T1;
}

//The only change I made was to store the normal force between i and j and particle[i].normalForce, which is used to find heatflux
void x_wall_forces(int i, int j)
/*
This routine first determines if a particle is in contact with either the left
or right vertical wall boundaries and if so, calculates the forces acting on
the particle
*/
{
	int
		k,
		q;
	double
		dx,            	// the particle separation in the x-direction
		overlap,          // amount of particle overlap
		nx, sy,           // direction of vector components
		fnmag, fsmag,     // normal and tangential forces
		dxdot, dydot,     // relative velocities in the x and y directions
		deltaTemp,
		qOne,
		qTwo,
		ndot, sdot;       // relative velocities in the n and s directions
	extern double
		rmin,
		k_wn,
		nu_wn,
		k_ws,
		mu_w,
		k2w,
		deltat,
		Pin,
		Pin2,
		Pout,
		Pout2,
		SensorTop,
		SensorBottom,
		SensorTop2,
		SensorBottom2;
	extern struct state *particle;

	// determine the amount of overlap between particles
	dx = particle[j].x - particle[i].x;
	overlap = particle[i].r - fabs(dx);

	if (overlap >= 0.0)     // checks for contact
	{
		if (overlap / rmin > MAX_OVERLAP)    // checks for too much overlap
		{
			printf("WARNING: t=%.3e sec\t%d/W%d", t, i, j);
			printf("\toverlap/rmin=%.3e\n", overlap / rmin);
		}
		
		qOne = 0;
		qTwo = 0;
		// determine the direction vectors for the contact
		nx = sgn(dx);
		sy = nx;

		// determine the relative contact velocities
		dxdot = particle[j].xdot - particle[i].xdot;
		dydot = particle[j].ydot - particle[i].ydot;
		ndot = dxdot * nx;
		sdot = dydot * sy - particle[i].r*particle[i].thetazdot;

		// determine the normal force (FNMAG)
		fnmag = -k2w * pow(overlap, 1.5) + nu_wn * pow(overlap, 0.5)*ndot;
		particle[i].Fx += fnmag * nx;
		/*
		deltaTemp = deltaT(i, j);
		//ref* is .5
		qOne = (pow(0.5*overlap, 0.5));
		//printf("ndot: %f ", ndot);
		if (ndot < 0)
			qTwo = (pow((1.5*(pow(0.5, (float) 1.5))*A_pn*pow(overlap, 0.5)*-1 * ndot), (float)1 / 3));
		else
			qTwo = (pow((1.5*(pow(0.5, 1.5))*A_pn*pow(overlap, 0.5) * ndot), (float)1 / 3));
		//printf("qTwo: %f \n", qTwo);
		particle[i].qSum1 += (2.0*deltaTemp*qOne);
		particle[i].qSum2 += (2.0*deltaTemp*qTwo);
		particle[i].qSum += (2.0*deltaTemp*(qOne + qTwo));
		//particle[i].qSum += (2.0*deltaTemp*(qOne));

		//qOne = 0;
		//qTwo = 0;
		*/
		// calculate the pressure measured through the bed
		if (j == N + 4)
		{
			if ((particle[i].y < SensorTop) && (particle[i].y > SensorBottom))
				Pout -= fnmag * nx;
			if ((particle[i].y < SensorTop2) && (particle[i].y > SensorBottom2))
				Pout2 -= fnmag * nx;
		}
		// calculate the pressure input into the bed
		if (j == N + 3)
		{
			if ((particle[i].y < SensorTop) && (particle[i].y > SensorBottom))
				Pin += fnmag * nx;
			if ((particle[i].y < SensorTop2) && (particle[i].y > SensorBottom2))
				Pin2 += fnmag * nx;
		}
		// determine if the contact between these two particles is a continuting
		// contact or if it is a new contact
		// go through the contact list
		k = 1;
		while ((particle[i].contact[k] != j) && (k <= CONTACT_MAX))
			k++;
		if (k <= CONTACT_MAX)  // this is an old contact
		{
			q = k;
		}
		else   				  // this is a new contact
		{
			k = 1;
			while (particle[i].contact[k] != 0)
				k++;
			q = k;
			particle[i].contact[q] = j;
			particle[i].deltas[q] = 0.0;
		}
		//This seems to count the number of contacts of each particle
		// update the tangential spring extension for this contact
		particle[i].deltas[q] += sdot * deltat;

		// check to see if there is slip between these particles, if so adjust the
		// spring extension appropriately
		if (mu_w*fabs(fnmag) < fabs(k_ws*particle[i].deltas[q]))
			particle[i].deltas[q] = sgn(k_ws*particle[i].deltas[q])*mu_w*fabs(fnmag) / k_ws;

		// determine the force in the tangential direction
		fsmag = k_ws * particle[i].deltas[q];
		particle[i].Fy += fsmag * sy;
		particle[i].Tz += particle[i].r*fsmag;
	}
	else
	{
		// remove reference to particle j in particle[i].contacts since particle
		// i and particle j are not in contact
		for (k = 1;k <= CONTACT_MAX;k++)
			if (particle[i].contact[k] == j)
			{
				particle[i].contact[k] = 0;
			}
	}
}

void y_wall_forces(int i, int j)

/*
This routine first determines if a particle is in contact with either the
horizontal top or bottom wall boundaries and if so, calculates the forces
acting on the particle.
*/

{
	int
		k,
		q;
	double
		dy,             // particle separation in the y-direction
		overlap,        // amount of particle overlap
		ny, sx,         // direction vector components
		fnmag, fsmag,   // normal and tangential forces
		dxdot, dydot,   // relative velocities in the x and y directions
		deltaTemp, qOne,
		qTwo,
		ndot, sdot;     // relative velocities in the n and s directions
	extern double
		rmin,
		k_wn,
		nu_wn,
		k_ws,
		mu_w,
		k2w,
		deltat;
	extern struct state *particle;

	// determine the amount of overlap between particles
	dy = particle[j].y - particle[i].y;
	overlap = particle[i].r - fabs(dy);
	qOne = 0;
	qTwo = 0;
	if (overlap >= 0.0)    // check for contact
	{
		if (overlap / rmin > MAX_OVERLAP)   // check for too much overlap
		{
			printf("WARNING: t=%.3e sec\t%d/W%d", t, i, j);
			printf("\toverlap/rmin=%.3e\n", overlap / rmin);
		}
		// determine the direction vectors for the contact
		ny = sgn(dy);
		sx = -ny;

		// determine the relative contact velocities
		dxdot = particle[j].xdot - particle[i].xdot;
		dydot = particle[j].ydot - particle[i].ydot;
		ndot = dydot * ny;
		sdot = dxdot * sx - particle[i].r*particle[i].thetazdot;

		// determine the normal force
		fnmag = -k2w * pow(overlap, 1.5) + nu_wn * pow(overlap, 0.5)*ndot;
		particle[i].Fy += fnmag * ny;

		//deltaTemp = deltaT(i, j);
		//ref* is .5

		/*
		qOne = (pow(0.5*overlap, 0.5));
		if (ndot < 0)
			qTwo = (pow(1.5*(pow(0.5, 1.5))*A_pn*pow(overlap, 0.5)*-1 * ndot, (float)1 / 3));

		else
			qTwo = (pow(1.5*(pow(0.5, 1.5))*A_pn*pow(overlap, 0.5) * ndot, (float)1 / 3));
			*/
		//if (2.0*deltaTemp*((pow(0.5*overlap, 0.5)) + qTwo) < 0)
	//		printf("here");


	//	particle[i].qSum1 += 2.0*deltaTemp*(pow(0.5*overlap, 0.5));
		//particle[i].qSum2 += 2.0*deltaTemp*qTwo;
		//particle[i].qSum += 2.0*deltaTemp*((pow(0.5*overlap, 0.5)) + qTwo);

		//qOne = 0;
		//qTwo = 0;

		// determine if the contact between these two particles is a continuing
		// contact or a new contact

		// go through contact list
		k = 1;
		while ((particle[i].contact[k] != j) && (k <= CONTACT_MAX))
			k++;
		if (k <= CONTACT_MAX)           // this is an old contact
		{
			q = k;
		}
		else                         // this is a new contact
		{
			k = 1;
			while (particle[i].contact[k] != 0)
				k++;
			q = k;
			particle[i].contact[q] = j;
			particle[i].deltas[q] = 0.0;
		}
		// update the tangential spring extension for this contact
		particle[i].deltas[q] += sdot * deltat;

		// check to see if there is slip between these particles; if so adjust
		// the spring extension appropriately
		if (mu_w*fabs(fnmag) < fabs(k_ws*particle[i].deltas[q]))
			particle[i].deltas[q] = sgn(k_ws*particle[i].deltas[q])*mu_w*fabs(fnmag) / k_ws;

		// determine the force in the tangential direction
		fsmag = k_ws * particle[i].deltas[q];
		particle[i].Fx += fsmag * sx;
		particle[i].Tz += particle[i].r*fsmag;
	}
	else
	{
		// remove reference to particle j in particle[i].deltas since particle
		// i and particle j are not in contact
		for (k = 1;k <= CONTACT_MAX;k++)
			if (particle[i].contact[k] == j)
			{
				particle[i].contact[k] = 0;
			}
	}
}


void particle_forces(int i, int j, double pj_dx)
/*
This routine first determines if two particles are in contact and if so,
calculates the forces acting on them
*/
{
	int
		k,
		q;
	double
		dx, dy,               // the particle separation in the x and y directions
		dij,                  // the distance between the particle centers
		overlap,              // amount of particle overlap
		nx, ny,               // the direction vectors in the norm. direction
		sx, sy,               // the direction vectors in the tang, direction
		fnmag, fsmag,			 // normal and tangential forces
		dxdot, dydot,			 // relative velocities in the x and y directions
		deltaTemp, qOne,
		qTwo, third,
		ndot, sdot;				 // relative velocities in the n and s directions
	extern double
		rmin,
		k_pn,
		nu_pn,
		k_ps,
		mu_p,
		k2p,
		deltat;
	extern struct state *particle;

	qOne = 0;
	qTwo = 0;
	// determine the amount of overlap between particles
	dx = (particle[j].x + pj_dx) - particle[i].x;         // WHY IS PJ_DX HERE!!!
	dy = particle[j].y - particle[i].y;
	dij = sqrt(dx*dx + dy * dy);
	overlap = (particle[j].r + particle[i].r) - dij;

	//  contact forces
	if (overlap >= 0.0)              // check for contact
	{
		if (overlap / rmin > MAX_OVERLAP)     // check for too much overlap
		{
			printf("WARNING: t=%.3e sec\t%d/%d", t, i, j);
			printf("\toverlap/rmin=%.3e", overlap / rmin);
			printf("Dx: %e, Dy: %e, Y1: %e , Y2: %e X1: %e , X2: %e  \n", dx, dy, particle[j].y, particle[i].y, particle[j].x, particle[i].x);
		}

		// determine the direction vectors for the contact20
		nx = dx / dij;
		ny = dy / dij;
		sx = -ny;
		sy = nx;

		// determine the relative contact velocities
		dxdot = particle[j].xdot - particle[i].xdot;
		dydot = particle[j].ydot - particle[i].ydot;
		ndot = dxdot * nx + dydot * ny;
		sdot = dxdot * sx + dydot * sy -
			particle[j].r*particle[j].thetazdot -
			particle[i].r*particle[i].thetazdot;

		// determine the normal forces
		fnmag = -k2p * pow(overlap, 1.5) + nu_pn * pow(overlap, 0.5)*ndot;
		particle[i].Fx += fnmag * nx;
		particle[i].Fy += fnmag * ny;
		particle[j].Fx -= fnmag * nx;
		particle[j].Fy -= fnmag * ny;


		/*
		deltaTemp = deltaT(i, j);

		//qOne = (pow(0.25*overlap, 0.5));
		if (ndot < 0)
			qTwo = (pow(1.5*(pow(0.25, 1.5))*A_pn*pow(overlap, 0.5)*-1 * ndot, (float)1 / 3));

		else
			qTwo = (pow(1.5*(pow(0.25, 1.5))*A_pn*pow(overlap, 0.5) * ndot, (float)1 / 3));

		particle[i].qSum1 += 2.0*deltaTemp*(pow(0.25*overlap, 0.5));
		particle[i].qSum2 += 2.0*deltaTemp*qTwo;

		particle[j].qSum1 -= 2.0*deltaTemp*(pow(0.25*overlap, 0.5));
		particle[j].qSum2 -= 2.0*deltaTemp*qTwo;
		//if (ndot < 0)
		//{
		particle[i].qSum += 2.0*deltaTemp*((pow(0.25*overlap, 0.5)) + qTwo);
		particle[j].qSum -= 2.0*deltaTemp*((pow(0.25*overlap, 0.5)) + qTwo);
		*/
		//}
		//else {
			//particle[i].qSum += 2.0*deltaTemp*((pow(0.25*overlap, 0.5)) + qTwo);
			//particle[j].qSum -= 2.0*deltaTemp*((pow(0.25*overlap, 0.5)) + qTwo);
		//}

		// go through the contact list
		k = 1;
		while ((particle[i].contact[k] != j) && (k <= CONTACT_MAX))
			k++;
		if (k <= CONTACT_MAX)		// this is an old contact
		{
			q = k;
		}
		else							// this is a new contact
		{
			k = 1;
			while (particle[i].contact[k] != 0)
				k++;
			q = k;
			particle[i].contact[q] = j;
			particle[i].deltas[q] = 0.0;
		}

		// update the tangential spring extension for this contact
		particle[i].deltas[q] += sdot * deltat;

		// check to see if there is slip between these particles; is so adjust
		// the spring extension appropriately
		if (mu_p*fabs(fnmag) < fabs(k_ps*particle[i].deltas[q]))
			particle[i].deltas[q] = sgn(k_ps*particle[i].deltas[q])*mu_p*fabs(fnmag) / k_ps;

		// determine the force in the tangential direction
		fsmag = k_ps * particle[i].deltas[q];
		particle[i].Fx += fsmag * sx;
		particle[i].Fy += fsmag * sy;
		particle[i].Tz += particle[i].r*fsmag;
		particle[j].Fx -= fsmag * sx;
		particle[j].Fy -= fsmag * sy;
		particle[j].Tz += particle[j].r*fsmag;
	}
	else
	{
		// remove reference to particle j in particle[i].contact since particle i
		// and particle j are not in contact
		for (k = 1;k <= CONTACT_MAX;k++)
			if (particle[i].contact[k] == j)
			{
				particle[i].contact[k] = 0;
			}
	}
}

void rainParticles(void)
{
	int index, particlesInserted;
	double yToInsert;
	int size = particlesToInsert.size();
	
	double	deltan = x_width / (double)(n);
	printf("raining %d \n", size);
	particlesInserted = 0;

	yToInsert = y_width - windowHeight;
	particlesInserted = 0;
	firstParticle = particlesToInsert.at(0);
	while (yToInsert < 119&&!particlesToInsert.empty())
	{
		//printf("y to insert : %e \n", yToInsert);
		firstParticle = particlesToInsert.at(0);
		index = particlesToInsert.at(0);
		particle[index].y = yToInsert;
		particle[index].ydot = 0;
		particle[index].xdot = 2 * v0max*(0.5 - (double)rand() / (double)RAND_MAX);
		particle[index].x = deltan * (particlesInserted + 0.5);
		particlesToInsert.erase(particlesToInsert.begin() + 0);
		particle[index].rain = false;
		particle[index].particleTemperature = 0;
		particlesInserted++;
		if (particlesInserted >= n)
		{
			yToInsert+=1.2;
			particlesInserted = 0;
		}
	}

	timeAfterInsert = 0;
	if (particlesToInsert.empty())
		printf("empty \n");
	else
		printf("not empty \n");
	
	if (n > size)
		particlesToInsert.resize(0);
}

void integrate_eqns(void)
/*
This routine integrates the equations of motion for the particles and the
moving boundaries.
*/
{
	void remove_from_cell(int x, int y, int i);
	
	void add_to_cell(int x, int y, int i);
	double calculateConvection(int i);
	void rainParticles();
	double maxY;
	int
		i,
		cellx,
		celly;
	extern int
		N,
		wall_num,
		x_wall_flag;
	extern double
		t,
		t0,
		tmax,
		deltat,
		a,
		w,
		x_width,
		y_width,
		cellsize,
		SensorTop,
		SensorBottom,
		SensorTop2,
		SensorBottom2,
		SensorSize,
		SensorLoc,
		SensorLoc2,
		xToInsert,
		yToInsert;
	extern struct state *particle;
	extern struct element **cell;

	// update time
	t += deltat;

	// update wall boundary positions and velocities
	for (i = N + 1;i <= N + wall_num; i++)
		particle[i].ydot = a * w*cos(w*t);
	particle[N + 1].y = a * sin(w*t);
	particle[N + 2].y = particle[N + 1].y + y_width;

	SensorTop = particle[N + 1].y + SensorLoc + SensorSize / 2;
	SensorBottom = SensorTop - SensorSize;
	SensorTop2 = particle[N + 1].y + SensorLoc2 + SensorSize / 2;
	SensorBottom2 = SensorTop2 - SensorSize;
	int particleF = 1;
	
	int n = (int)(x_width / (1.05*dmax));
	double	deltan = x_width / (double)(n);
	int xcounter = 1;
	particlesInserted = 0;
	maxY = 0;
	// update particle positions and velocities
	for (i = 1;i <= N;i++)
	{
		// this is a "leap frog" integration scheme
		particle[i].xdot += (particle[i].Fx / particle[i].m)*deltat;
		particle[i].ydot += (particle[i].Fy / particle[i].m)*deltat;
		particle[i].thetazdot += (particle[i].Tz / particle[i].I)*deltat;
		particle[i].y += particle[i].ydot*deltat;
		particle[i].x += particle[i].xdot*deltat;
		particle[i].qSum += calculateConvection(i);
		if (particle[i].y > maxY)
			maxY = particle[i].y;

		if (particle[i].y <= y_width - 2*windowHeight &&!particle[i].rain)
		{
			particlesToInsert.push_back(i);
			particle[i].rain = true;
		}		
		else {
			particle[i].y += particle[i].ydot*deltat;
		}
		
		particle[i].thetaz += particle[i].thetazdot*deltat;
		// keep rotational position between 0 and 2*PI
		if (particle[i].thetaz > 2.0*PI)
			particle[i].thetaz = particle[i].thetaz - 2.0*PI*(int)((particle[i].thetaz) / (2.0*PI));

		// if using periodic boundaries, check to see if the particle has wrapped
		// around in the horizontal direction
		if (x_wall_flag == 0)
		{
			if (particle[i].x < 0.0)
				particle[i].x += x_width;
			if (particle[i].x > x_width)
				particle[i].x -= x_width;
		}

		// check to see if the particle has changed cells
		cellx = (int)(particle[i].x / cellsize) + 1;
		celly = (int)((particle[i].y - particle[N + 1].y) / cellsize) + 1;
		if ((particle[i].cellx != cellx) || (particle[i].celly != celly))
		{
			remove_from_cell(particle[i].cellx, particle[i].celly, i);
			add_to_cell(cellx, celly, i);
		}

	}

	if (maxY <= y_width-windowHeight-1.05&& particlesToInsert.size() >= n)
	{
		printf("Free Surface: %f, maxY: %f \n", FreeSurface, maxY);
		rainParticles();
	}
}


double calculateConvection(int particleIndex)
{
	double calculateVelo(int);
	double getC(double);
	double getM(double);
	double
		nu,
		heatTransferCoef,
		reynoldsNumber, C, M, q;

	if (particle[particleIndex].y < y_width - windowHeight && particle[particleIndex].y > y_width - 2 * windowHeight) {
		//d is assumed to be 1
		reynoldsNumber = calculateVelo(particleIndex) / v;
		C = getC(reynoldsNumber);
		M = getM(reynoldsNumber);
		nu = C * pow(reynoldsNumber, M)*pow(Pr, (float)1 / 3);
		q = nu * k_ratio * 2 * PI*0.5*(airTemp - particle[particleIndex].particleTemperature);
	}	
	else {
		q = 0;
	}
	return q;
}

//from the table 2 on page 458, used to caluclate nusselts number
double getM(double reynoldsNumber)
{
	if (reynoldsNumber <= 4)
		return .330;
	else if (reynoldsNumber <= 40)
		return .385;
	else if (reynoldsNumber <= 4000)
		return .466;
	else if (reynoldsNumber <= 40000)
		return .618;
	else
		return .805;
}

//from table 7.2 on page 458 used to calc Nu
double getC(double reynoldsNumber)
{
	if (reynoldsNumber <= 4)
		return .989;
	else if (reynoldsNumber <= 40)
		return .911;
	else if (reynoldsNumber <= 4000)
		return .683;
	else if (reynoldsNumber <= 40000)
		return .193;
	else
		return .027;
}

double calculateVelo(int i)
{
	double velo;
	velo=pow(pow(particle[i].ydot,2)+pow(particle[i].xdot,2) , (float).5);
	return velo;

}


void read_data(char Directory[8])
/*
This routine reads in the input data file and calls a routine to initialize
the simulation
*/
{
	void initialize_states(void);

	int
		i, j; 							// temporary integers
	double
		f,								// cyclic frequency {Hz}
		theta_g,						// the gravity vector angle from the vertical	
		//A_wn, A_pn,					// particle/particle and particle/wall viscous coefficients
		A_wn, 				// particle/particle and particle/wall viscous coefficients
		ddev,							// the fractional deviation from the mean diameter
		dmin,							// the minimum particle diameter
										//overlap=0.01,				// the expected maximum ratio of the overlap to the
										// 	mean particle radius for a collision with
										//xd_r=1000.0,				//		a relative impact velocity equal to xd_r*mean
										// 	particle radius*1 sec
		tau_tp,                 // translational and rotational periods for a particle
		tau_rp,               	//		in contact with six other fixed particle contacts
		tau_H,
		rbar,							// mean particle radius	
		Ibar,							// mean particle moment of inertia
		samps_per_cycle,			// number of samples to record per cycle
		samps_per_second,		// number of samples per second to read the pressure sensors
		Poissonp,
		k1p,
		k1w,
		F_typ,

		//particleTemperature,	//The temperature of the particle

		Vmax;
	FILE *infile;					// input file pointer
	FILE *condout;
	extern int
		N,
		x_wall_flag,
		wall_num,
		cellmax_x,
		cellmax_y;
	extern double
		Gamma,
		x_width,
		y_width,
		k_pn,
		nu_pn,
		k_ps,
		mu_p,
		k_wn,
		nu_wn,
		k_ws,
		k2p,
		k2w,
		mu_w,
		dmax,
		deltat,
		a,
		w,
		InputAmp,
		InputWidthOrFreq,
		Fr,
		gx,
		gy,
		tmax,
		t_samp_recordstates,
		deltat_samp_recordstates,
		t_psamp_recordstates,
		deltat_psamp_recordstates,
		cellsize,
		SensorTop,
		SensorBottom,
		SensorTop2,
		SensorBottom2,
		SensorSize,
		SensorLoc,
		SensorLoc2,
		A_pn,
		initialWallTemp,
		InitialParticleTemp,
		qSum2Sum,
		qSum1Sum,
		v0max;
	extern struct state *particle;
	extern struct element **cell;

	const char
		COut[] = "conditions.out",
		In[] = "conditions.in";

	char
		CFileName[25],
		IFileName[20];

	error_t
		err;

	strcpy(IFileName, "conditions.in");
	infile = fopen64(IFileName, "rb");
	printf("The input file was opened.\n");

	strcpy(CFileName, "conditions.out");
	condout = fopen64(CFileName, "wb");
	printf("The output file was opened.\n");

	// read in the input file data

	fscanf(infile, "%lg %lg %lg", &Gamma, &InputAmp, &Fr);
	fscanf(infile, "%lg %lg %lg %lg %lg", &tmax, &f, &InputWidthOrFreq, &samps_per_cycle,
		&samps_per_second);
	fscanf(infile, "%lg %lg %lg %d", &x_width, &y_width, &theta_g, &x_wall_flag);
	fscanf(infile, "%lg %lg %lg", &SensorSize, &SensorLoc, &SensorLoc2);
	fscanf(infile, "%d", &N);
	fscanf(infile, "%lg %lg %lg %lg", &A_pn, &k_ps, &mu_p, &Poissonp);
	fscanf(infile, "%lg %lg %lg", &A_wn, &k_ws, &mu_w);
	fscanf(infile, "%lg %lg", &ddev, &v0max);
	fclose(infile);


	wall_num = 4;
	qSum2Sum = 0;
	qSum1Sum = 0;
//	N *= (int)x_width;
	N = 1000;
	// make room for *particle
	printf("%d bytes required for *particle.\n", (N + wall_num + 1) * sizeof(struct state));
	if ((particle = (struct state *) calloc(N + wall_num + 1, sizeof(struct state))) == NULL)
	{
		printf("Not enough storage for *particle.\n");
		fclose(condout);
		exit(1);
	}

	rbar = 0.5;
	Ibar = (2.0 / 5.0)*1.0*rbar*rbar;
	dmax = 1.0 + ddev;
	dmin = (1.0 - ddev);
	rmin = 0.5;

	/*
	randomly choose the particle radii and then determine the particle mass and
	moment of inertia; NOTE: mass and moment of inertia are based on that of
	a sphere
	*/

	srand((unsigned)time(NULL));   		//  initialize the random number generator
	for (i = 1;i <= N;i++)
	{
		particle[i].r = 0.5*(1.0 + 2.0*ddev*(0.5 - (double)rand() / (double)RAND_MAX));
		particle[i].m = 8 * particle[i].r*particle[i].r*particle[i].r;
		particle[i].I = (2.0 / 5.0)*particle[i].m*particle[i].r*particle[i].r;
		particle[i].num = i;
		particle[i].particleTemperature = 0.0;
	}

	// determine the gravity acceleration components
	gx = Fr * sin(theta_g*PI / 180.0);
	printf("Gx: %.5e ", gx);
	gy = -Fr * cos(theta_g*PI / 180.0);
	F_typ = -N * gy / (2.0*x_width);
	// determine particle/particle spring constants and dashpot coefficient
	k1p = 2.0;
	k2p = 2 / (PI*(1 - Poissonp * Poissonp));
	k_pn = pow(k2p, 2.0 / 3.0)*pow(F_typ, 1.0 / 3.0);;
	nu_pn = 3 * A_pn / (PI*(1 - Poissonp * Poissonp));
	k_ps = k_ps * k_pn;
	// determine the particle/wall spring constants and dashpot coefficients
	k1w = 1.0;
	k2w = 2.0*sqrt(2.0) / (PI*(1 - Poissonp * Poissonp));  //  assumes wall & particle of same material
	k_wn = pow(k2w, 2.0 / 3.0)*pow(F_typ, 1.0 / 3.0);
	nu_wn = 3 * sqrt(2.0)*A_wn / (PI*(1 - Poissonp * Poissonp));
	k_ws = k_ws * k_wn;
	// determine the oscillation radian frequency and amplitude from the cyclic
	// frequency, f, and the acceleration amplitude, Gamma
	w = 2.0*PI*f;
	a = Fr * Gamma / (w*w);
	// determine the simulation time step
	Vmax = a * w;
	if (v0max*sqrt(2.0) > Vmax)
		Vmax = v0max * sqrt(2.0);
	// contact time based on Hertzian contact
	if (k1p*k2p > k1w*k2w)
		tau_H = (2.9432 / pow(Vmax, 0.2))*pow(5 / (4 * k1p*k2p), 0.4);
	else
		tau_H = (2.9432 / pow(Vmax, 0.2))*pow(5 / (4 * k1w*k2w), 0.4);
	// translational period for a particle in contact with six other (fixed)
	// particles with no damping
	tau_tp = 2.0*PI*sqrt(1 / (3.0*(k_pn + k_ps)));
	// translational period for a particle in contact with six other (fixed)
	// particles with no damping
	tau_rp = 2.0*PI*sqrt(Ibar / (6.0*k_ps*rbar*rbar));
	// choose the smallest period and make the simulation time step one-tenth
	// that period
	if (tau_tp < tau_rp)
		deltat = tau_tp;
	else
		deltat = tau_rp;
	if (tau_H < deltat)
		deltat = tau_H / 10.0;
	else
		deltat = deltat / 10.0;


	// determine the initial top and bottom of the sensor
	SensorTop = SensorLoc + SensorSize / 2;
	SensorBottom = SensorTop - SensorSize;
	SensorTop2 = SensorLoc2 + SensorSize / 2;
	SensorBottom2 = SensorTop2 - SensorSize;

	// determine the next time a recording of the output states will be made and
	// the time between successive recordings
	deltat_samp_recordstates = (1 / f) / samps_per_cycle;
	t_samp_recordstates = 0.0;		//14.9*deltat_samp_recordstates;

	t_psamp_recordstates = 0.0;
	deltat_psamp_recordstates = 1 / samps_per_second;

	// make room for the cell list pointers; NOTE: the cellsize is just larger
	// than the maximum particle diameter,dmax
	cellmax_x = (int)(x_width / dmax) + 1;
	cellsize = x_width / (double)(cellmax_x - 1);
	cellmax_y = (int)(y_width / cellsize) + 1;
	printf("%d bytes required for **cell.\n", (cellmax_x + 1) * sizeof(struct element *)*
		(cellmax_y + 1) * sizeof(struct element));
	if ((cell = (struct element **) calloc(cellmax_x + 1, sizeof(struct element *))) == NULL)
	{
		printf("Can't assign cell row pointers.\n");
		exit(1);
	}
	for (i = 0;i <= cellmax_x;i++)
	{
		if ((cell[i] = (struct element *) calloc(cellmax_y + 1, sizeof(struct element))) == NULL)
		{
			printf("Not enough storage for cell columns.\n");
			exit(1);
		}
	}

	// initialize cell list
	for (j = 0;j <= cellmax_y;j++)
		for (i = 0;i <= cellmax_x;i++)
			cell[i][j].next = NULL;

	// call routine to determine the initial conditions for the simulation
	initialize_states();

	// print the input data to the screen
	printf("\nfr=%.3e VGamma=%.2f InputAmp=%.3e InputWidthOrFreq=%.2f\n", Fr, Gamma,
		InputAmp, InputWidthOrFreq);
	printf("tmax=%.3e  f=%.3e  theta_g=%.3e deg\n", tmax, f, theta_g);
	printf("samps_per_cycle=%.3e Sampling Rate=%.3e\n", samps_per_cycle, samps_per_second);
	printf("tau_tp=%.3e s tau_rp=%.3e s tau_H=%.3e s deltat=%.3e s\n", tau_tp, tau_rp,
		tau_H, deltat);
	printf("x_width=%.1f\n", x_width);
	printf("y_width=%.1f\n", y_width);
	printf("theta_g=%.3e deg  x_wall_flag=%d\n", theta_g * 180 / PI, x_wall_flag);
	printf("Sensor Size=%.1f  Sensor Locations=%.1f , %.1f\n", SensorSize,
		SensorLoc, SensorLoc2);
	printf("N=%d\n", N);
	printf("K2p=%.3e K2w=%.3e\n", k2p, k2w);
	printf("k_pn=%.3e  nu_pn=%.3e N/(m/s) A_pn=%.3f\n", k_pn, nu_pn, A_pn);
	printf("k_ps=%.3e  mu_p=%.3f\n", k_ps, mu_p);
	printf("k_wn=%.3e  nu_wn=%.3e N/(m/s) A_wn=%.3f\n", k_wn, nu_wn, A_wn);
	printf("k_ws=%.3e  mu_w=%.3f\n", k_ws, mu_w);
	printf("ddev=%.3e\n\n", ddev);

	// print the input data to the conditions file
	fprintf(condout, "%e %e\n", Gamma, InputAmp);
	fprintf(condout, "%e %e\n", f, InputWidthOrFreq);
	fprintf(condout, "%e %e\n", tmax, theta_g);
	fprintf(condout, "%e %e\n", samps_per_cycle, samps_per_second);
	fprintf(condout, "%e %e\n", tau_tp, tau_rp);
	fprintf(condout, "%e %e\n", tau_H, deltat);
	fprintf(condout, "%e %e\n", x_width, y_width);
	fprintf(condout, "%e %e\n", SensorSize, SensorLoc);
	fprintf(condout, "%e %e\n", SensorLoc2, Poissonp);
	fprintf(condout, "%d %d\n", N, x_wall_flag);
	fprintf(condout, "%e %e\n", k_pn, nu_pn);
	fprintf(condout, "%e %e\n", k_ps, k_ws);
	fprintf(condout, "%f %f\n", A_pn, mu_p);
	fprintf(condout, "%e %e\n", k_wn, nu_wn);
	fprintf(condout, "%f %f\n", A_wn, mu_w);
	fprintf(condout, "%e %e\n", gx, gy);
	fprintf(condout, "%e %f\n", Fr, ddev);
	for (i = 1;i <= N;i++)
		fprintf(condout, "%e %e\n", particle[i].r, particle[i].m);
	fclose(condout);
}


void initialize_states(void)
/*
This routine determines the initial conditions for the simulations.  The wall
conditions are initialized and then the particle conditions are assigned.  Note
that particles are placed in the container at prescribed positions (in a grid)
but with random initial velocities.
Particles are assigned 20 degrees
Wall assigned 100 degrees
*/
{
	void add_to_cell(int x, int y, int i);

	int
		i,
		xcounter,
		cellx,
		celly;
	double
		deltan;
	extern int N, n;
	extern double
		x_width,
		y_width,
		cellsize,
		dmax,
		v0max;
	extern struct state *particle;
	extern struct element **cell;

	// initialize wall states
	//initialize wall temperature to be 920 (no unit)


	for (i = N + 1;i <= N + wall_num;i++)
	{
		particle[i].x = 0.0;
		particle[i].y = 0.0;
		particle[i].thetaz = 0.0;
		particle[i].xdot = 0.0;
		particle[i].ydot = 0.0;
		particle[i].thetazdot = 0.0;
		particle[i].r = 0.0;
		particle[i].num = i;
		particle[i].particleTemperature = 1.0;
	}
	//Left wall is heated
	//particle[N + 3].particleTemperature = 1.0;
	//n+3 is left
	//n+4 is right
	//particle[N + 3].particleTemperature = 1000;
	particle[N + 1].y = 0.0;
	particle[N + 2].y = y_width;
	particle[N + 3].x = 0.0;
	particle[N + 4].x = x_width;

	// put the walls in the proper cells
	for (i = N + 1;i <= N + 2;i++)
	{
		particle[i].cellx = 0;
		particle[i].celly = (int)((particle[i].y - particle[N + 1].y) / cellsize) + 1;
	}
	for (i = N + 3;i <= N + 4;i++)
	{
		particle[i].cellx = (int)(particle[i].x / cellsize) + 1;
		particle[i].celly = 0;
	}

	// intialize particle states
	srand((unsigned)time(NULL));
	n = (int)(x_width / (1.05*dmax));
	deltan = x_width / (double)(n);
	xcounter = 1;
	for (i = 1;i <= N;i++)
	{
		particle[i].x = deltan * (xcounter - 0.5);
		particle[i].y = deltan * ((double)((int)((i - 1) / n) + 1));
		particle[i].thetaz = 0.0;
		particle[i].xdot = 2 * v0max*(0.5 - (double)rand() / (double)RAND_MAX);
		particle[i].ydot = 2 * v0max*(0.5 - (double)rand() / (double)RAND_MAX);
		particle[i].thetazdot = 0.0;
		// put the particles in the proper cells
		cellx = (int)(particle[i].x / cellsize) + 1;
		celly = (int)((particle[i].y - particle[N + 1].y) / cellsize) + 1;
		add_to_cell(cellx, celly, i);

		xcounter++;
		if (xcounter > n)
			xcounter = 1;
	}
	
}
//no change
void remove_from_cell(int x, int y, int i)
/*
This routine removes a particle from a cell
*/
{
	extern struct state *particle;
	extern struct element **cell;

	if (particle[i].prev == NULL)
		cell[x][y].next = particle[i].next;
	else
		(particle[i].prev)->next = particle[i].next;
	if (particle[i].next != NULL)
		(particle[i].next)->prev = particle[i].prev;
}
//no change
void add_to_cell(int x, int y, int i)
/*
This routine adds a particle to a cell
*/
{
	extern struct state *particle;
	extern struct element **cell;

	if (cell[x][y].next != NULL)
		(cell[x][y].next)->prev = &(particle[i]);
	particle[i].next = cell[x][y].next;
	particle[i].prev = NULL;
	cell[x][y].next = &(particle[i]);
	particle[i].cellx = x;
	particle[i].celly = y;
}
void record_states(FILE *outfile)
/*
This routine prints the particle state data to an output file
*/
{
	int
		i;
	extern int
		N;
	extern double
		t;
	extern struct state *particle;

	//fprintf(outfile,"* %.5e\n",t);
	for (i = 1;i <= N;i++)    //(i=1;i<=N;i++)
	{
		fprintf(outfile, "%.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e %.5e \n",
			particle[i].x,
			particle[i].y,
			particle[i].xdot,
			particle[i].ydot,
			particle[i].Fx,
			particle[i].Fy,
			particle[i].particleTemperature,
			particle[i].qSum,
			particle[i].qSum1,
			particle[i].qSum2);

		if (particle[i].particleTemperature > 1)
			printf("Temperature: %.8e %d\n", particle[i].particleTemperature, i);

	}
}
void RecordStatesAll(FILE *outfile, char Directory[8])
/*
This routine prints all of the particle state data (including the walls) to an output
file so that a new simulation can be started from this point in time forward
*/
{
	int
		i;
	extern int
		N;
	extern struct state *particle;
	char
		StateFileName[20];
	const char
		SOut[] = "states.out";
	FILE *soutfile;
	error_t
		err;
	// open the output data files
	strcpy(StateFileName, "\0");
	strcat(StateFileName, "\\");
	strcat(StateFileName, SOut);
	soutfile = fopen64(StateFileName, "wb");
	for (i = 1;i <= N + 4;i++)
	{
		fprintf(soutfile, "%d %d %d %d %d %d %d %d %d %d %d %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n",
			particle[i].cellx, particle[i].celly, particle[i].contact[1], particle[i].contact[2],
			particle[i].contact[3], particle[i].contact[4], particle[i].contact[5],
			particle[i].contact[6], particle[i].contact[7], particle[i].contact[8],
			particle[i].contact[9], particle[i].x, particle[i].y, particle[i].thetaz,
			particle[i].xdot, particle[i].ydot,
			particle[i].thetazdot, particle[i].r, particle[i].m, particle[i].I,
			particle[i].deltas[1], particle[i].deltas[2], particle[i].deltas[3],
			particle[i].deltas[4], particle[i].deltas[5], particle[i].deltas[6],
			particle[i].deltas[7], particle[i].deltas[8], particle[i].deltas[9]);
	}
}
void LoadStates(char Directory[8])
/*
This routine loads in all necessary particle and wall state data to continue
the experiment from a previously settled case.
*/
{
	void add_to_cell(int x, int y, int i);
	extern int N;
	extern struct state *particle;
	extern struct element **cell;
	int
		k,
		j,
		cellx,
		Lcelly,
		contact[9];
	double
		x,
		y,
		thetaz,
		xdot,
		ydot,
		thetazdot,
		r,
		m,
		I,
		deltas[9];
	char
		StateFileName[20];
	const char
		SOut[] = "states.out";
	FILE *soutfile;
	error_t
		err;
	
	// open the output data files
	strcpy(StateFileName, "\0");
	//StateFileName = Directory & "\\" & SOut;
	strcat(StateFileName, Directory);
	strcat(StateFileName, "\\");
	strcat(StateFileName, SOut);
	soutfile = fopen64(StateFileName, "rb");
	printf("Loading states....");
	// intialize particle and wall states
	for (k = 1;k <= N + 4;k++)
	{
		fscanf(soutfile, "%d %d %d %d %d %d %d %d %d %d %d %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg\n",
			&cellx, &Lcelly, &contact[1], &contact[2], &contact[3], &contact[4], &contact[5],
			&contact[6], &contact[7], &contact[8], &contact[9],
			&x, &y, &thetaz, &xdot, &ydot, &thetazdot, &r, &m, &I, &deltas[1], &deltas[2],
			&deltas[3], &deltas[4], &deltas[5], &deltas[6], &deltas[7], &deltas[8], &deltas[9]);
		particle[k].num = k;
		particle[k].cellx = cellx;
		particle[k].celly = Lcelly;
		for (j = 1;j < 9;j++)
			particle[k].contact[j] = contact[j];
		particle[k].x = x;
		particle[k].y = y;
		particle[k].thetaz = thetaz;
		particle[k].xdot = xdot;
		particle[k].ydot = ydot;
		particle[k].thetazdot = thetazdot;
		particle[k].r = r;
		particle[k].m = m;
		particle[k].I = I;
		particle[k].rain = false;
		for (j = 1;j <= 8;j++)
			particle[k].deltas[j] = deltas[j];

		if (k <= N) add_to_cell(cellx, Lcelly, k);
	}
	fclose(soutfile);
	printf("Done\n");
}

void MovePiston(int ExcitationFlag)
/*
This routine updates the location of the piston.
*/
{
	extern int
		N;
	extern double
		t,
		t0,
		deltat,
		InputAmp,
		InputWidthOrFreq,
		cellsize;
	extern struct state *particle;
	if (ExcitationFlag == 1)			// pulse input
	{
		if (t <= (t0 + InputWidthOrFreq))
			particle[N + 3].xdot = (InputAmp*(t - t0) / InputWidthOrFreq);
		else
			if ((t > (t0 + InputWidthOrFreq)) && (t < (t0 + 2 * InputWidthOrFreq)))
				particle[N + 3].xdot = (InputAmp - InputAmp * (t - t0 - InputWidthOrFreq) / InputWidthOrFreq);
			else
				particle[N + 3].xdot = 0;
		particle[N + 3].x += particle[N + 3].xdot*deltat; // Maybe not right- rework with constant acceleration, a
	}
	else			// continuous input
	{
		particle[N + 3].x = InputAmp * sin(InputWidthOrFreq * 2 * PI*(t - t0));
		particle[N + 3].xdot = InputAmp * InputWidthOrFreq * 2 * PI*cos(InputWidthOrFreq * 2 * PI*(t - t0));
	}
}
