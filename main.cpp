/*
Perceptual Crossing Simulation
Gabriel J. Severino

Initial implementation done by Eduardo J. Izquierdo 2022

*/

#include "TSearch.h"
#include "PerceptualCrosser.h"
#include "CTRNN.h"
#include "random.h"
#include <dirent.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <regex>
#include <regex>
#include <set>
#include <dirent.h>  // For opendir, readdir, closedir
#include <utility>   // For std::pair
#include <filesystem>
#include <iomanip>

#define PRINTOFILE

// Task params
const double StepSize = 0.01; 
const double RunDuration = 800.0; 
const double TransDuration = 400.0; 

const double RunDurationMap = 1600.0; 
const double TransDurationMap = 1500.0;

const double Fixed1 = 150.0;
const double Fixed2 = 450.0;
const double Shadow = 48.0; 
const double SpaceSize = 600.0;
const double HalfSpace = 300;
const double SenseRange = 2.0; 
const double CloseEnoughRange = 2.0; 
const int STEPPOS1 = 50;
// const int STEPPOS2 = 25;

// EA params
const int POPSIZE = 96;
const int GENS = 100;     // 1000
const double MUTVAR = 0.05;			// ~ 1/VectSize for N=3
const double CROSSPROB = 0.0;
const double EXPECTED = 1.1;
const double ELITISM = 0.02;

// Nervous system params
const int N = 3; 
const double WR = 8.0;
const double SR = 8.0;
const double BR = 8.0;
const double TMIN = 1.0;
const double TMAX = 10.0;

int	VectSize = N*N + 2*N + N;

// ================================================
// A. FUNCTIONS FOR EVOLVING A SUCCESFUL CIRCUIT
// ================================================

// ------------------------------------
// Genotype-Phenotype Mapping Function
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), -BR, BR);
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			phen(k) = MapSearchParameter(gen(k), -WR, WR);
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), -SR, SR);
		k++;
	}
}

// ------------------------------------
// Fitness Functions
// ------------------------------------

double FitnessFunction1(TVector<double> &genotype, RandomState &rs)
{
	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double totalfit = 0.0, totaldist = 0.0, dist = 0.0;
	double totaltrials = 0, totaltime = 0.0;
	double shadow1, shadow2;

	for (int fixedFlag = 0; fixedFlag <= 1; fixedFlag += 1){
		for (int shadowFlag = 0; shadowFlag <= 1; shadowFlag += 1){
			if ((fixedFlag != 0) or (shadowFlag != 0)){
				for (double x1 = 0.0; x1 < SpaceSize; x1 += STEPPOS1){
					for (double x2 = 0.0; x2 < SpaceSize - x1; x2 += STEPPOS1) {

						// Set agents positions
						Agent1.Reset(x1);
						Agent2.Reset(x2);

						// Run the sim
						totaldist = 0.0;
						totaltime = 0;

						for (double time = 0; time < RunDuration; time += StepSize)
						{
							// Update shadow positions
							shadow1 = Agent1.pos + Shadow;
							if (shadow1 >= SpaceSize)
								shadow1 = shadow1 - SpaceSize;
							if (shadow1 < 0.0)
								shadow1 = SpaceSize + shadow1;

							// Notice the other shadow is a reflection (not a rotation, i.e., shadow2 = Agent2.pos - Shadow;).
							shadow2 = Agent2.pos + Shadow;
							if (shadow2 >= SpaceSize)
								shadow2 = shadow2 - SpaceSize;
							if (shadow2 < 0.0)
								shadow2 = SpaceSize + shadow2;

							// Sense
							if (shadowFlag == 0)
							{
								if (fixedFlag == 0){
									cout << "Errorrr" << endl;
								}
								else{
									Agent1.Sense(Agent2.pos, 999999999, Fixed2);
									Agent2.Sense(Agent1.pos, 999999999, Fixed1);
								}
							}
							else 
							{
								if (fixedFlag == 0){
									Agent1.Sense(Agent2.pos, shadow2, 999999999);
									Agent2.Sense(Agent1.pos, shadow1, 999999999);
								}
								else{
									Agent1.Sense(Agent2.pos, shadow2, Fixed2);
									Agent2.Sense(Agent1.pos, shadow1, Fixed1);
								}
							}

							// Move
							Agent1.Step(StepSize);
							Agent2.Step(StepSize);

							// Measure distance between them (after transients)
							if (time > TransDuration)
							{
								dist = fabs(Agent2.pos - Agent1.pos);
								if (dist > HalfSpace)
									dist =  SpaceSize - dist;
								if (dist < CloseEnoughRange)
									dist = CloseEnoughRange;
								totaldist += dist;
								totaltime += 1;
							}
						}
						totalfit += 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
						totaltrials += 1;
					}
				}
			}
		}
	}
	return totalfit/totaltrials;
}
// Second stage function for rewarding oscillatory agents
double FitnessFunction2(TVector<double> &genotype, RandomState &rs)
{
	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double totalfit = 0.0, totaldist = 0.0, dist = 0.0;
	double totaltrials = 0, totaltime = 0.0;
	double shadow1, shadow2;
	double totalcross = 0;
	int crosscounter;

	for (int fixedFlag = 0; fixedFlag <= 1; fixedFlag += 1){
		for (int shadowFlag = 0; shadowFlag <= 1; shadowFlag += 1){
			if ((fixedFlag != 0) or (shadowFlag != 0)){
				for (double x1 = 0.0; x1 < SpaceSize; x1 += STEPPOS1){
					for (double x2 = 0.0; x2 < SpaceSize - x1; x2 += STEPPOS1) {

						// Set agents positions
						Agent1.Reset(x1);
						Agent2.Reset(x2);

						// Run the sim
						totaldist = 0.0;
						totaltime = 0;
						crosscounter = 0;

						for (double time = 0; time < RunDuration; time += StepSize)
						{
							// Update shadow positions
							shadow1 = Agent1.pos + Shadow;
							if (shadow1 >= SpaceSize)
								shadow1 = shadow1 - SpaceSize;
							if (shadow1 < 0.0)
								shadow1 = SpaceSize + shadow1;

							// shadow2 = Agent2.pos - Shadow;
							shadow2 = Agent2.pos + Shadow;
							if (shadow2 >= SpaceSize)
								shadow2 = shadow2 - SpaceSize;
							if (shadow2 < 0.0)
								shadow2 = SpaceSize + shadow2;

							// Sense
							if (shadowFlag == 0)
							{
								if (fixedFlag == 0){
									cout << "Errorrr" << endl;
								}
								else{
									Agent1.Sense(Agent2.pos, 999999999, Fixed2);
									Agent2.Sense(Agent1.pos, 999999999, Fixed1);
								}
							}
							else 
							{
								if (fixedFlag == 0){
									Agent1.Sense(Agent2.pos, shadow2, 999999999);
									Agent2.Sense(Agent1.pos, shadow1, 999999999);
								}
								else{
									Agent1.Sense(Agent2.pos, shadow2, Fixed2);
									Agent2.Sense(Agent1.pos, shadow1, Fixed1);
								}
							}

							// Move
							Agent1.Step(StepSize);
							Agent2.Step(StepSize);

							// Measure distance between them (after transients)
							if (time > TransDuration)
							{
								dist = fabs(Agent2.pos - Agent1.pos);
								if (dist > HalfSpace)
									dist =  SpaceSize - dist;
								if (dist < CloseEnoughRange)
									dist = CloseEnoughRange;
								totaldist += dist;
								totaltime += 1;

								// Measure number of times the agents cross paths
								if (((Agent1.pastpos < Agent2.pastpos) && (Agent1.pos >= Agent2.pos)) || ((Agent1.pastpos > Agent2.pastpos) && (Agent1.pos <= Agent2.pos)))
								{
									crosscounter += 1;
								}
							}
						}

						totalfit += 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
						totaltrials += 1;
						totalcross += crosscounter/TransDuration;
					}
				}	
			}
		}
	}
	if (totalfit/totaltrials > 0.99){
		return 1.0 + (totalcross/totaltrials);
	}
	else{
		return totalfit/totaltrials;
	}
}

// ================================================
// B. FUNCTIONS FOR ANALYZING A SUCCESFUL CIRCUIT
// ================================================


double PairTest(TVector<double> &genotype1, TVector<double> &genotype2)
{
	// Map genootype to phenotype
	TVector<double> phenotype1;
	phenotype1.SetBounds(1, VectSize);
	GenPhenMapping(genotype1, phenotype1);

	// Map genotype to phenotype
	TVector<double> phenotype2;
	phenotype2.SetBounds(1, VectSize);
	GenPhenMapping(genotype2, phenotype2);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype1(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype2(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype1(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype2(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype1(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype2(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype1(k));
		Agent2.SetSensorWeight(i,phenotype2(k));
		k++;
	}

	ofstream outfile;
	outfile.open("out2.dat");

	double totaltrials = 0,totalfittrans = 0.0,totaldisttrans = 0.0, dist = 0.0;
	int totaltimetrans;
	double shadow1, shadow2;

	for (double x1 = 0; x1 < SpaceSize; x1 += 25.0) {
		for (double x2 = 0; x2 < SpaceSize; x2 += 25.0) {

			// Set agents positions
			Agent1.Reset(x1);
			Agent2.Reset(x2);

			// Run the sim
			totaldisttrans = 0.0;
			totaltimetrans = 0;

			for (double time = 0; time < RunDurationMap; time += StepSize)
			{
				// Update shadow positions
				shadow1 = Agent1.pos + Shadow;
				if (shadow1 >= SpaceSize)
					shadow1 = shadow1 - SpaceSize;
				if (shadow1 < 0.0)
					shadow1 = SpaceSize + shadow1;

				// XXX shadow2 = Agent2.pos - Shadow;
				shadow2 = Agent2.pos + Shadow;
				if (shadow2 >= SpaceSize)
					shadow2 = shadow2 - SpaceSize;
				if (shadow2 < 0.0)
					shadow2 = SpaceSize + shadow2;

				// Sense
				Agent1.Sense(Agent2.pos, shadow2, Fixed2);
				Agent2.Sense(Agent1.pos, shadow1, Fixed1);

				// Move
				Agent1.Step(StepSize);
				Agent2.Step(StepSize);

				// Measure distance between them
				dist = fabs(Agent2.pos - Agent1.pos);
				if (dist > HalfSpace)
					dist =  SpaceSize - dist;

				// Measure distance also for the fitness calc
				if (time > TransDurationMap)
				{
					if (dist < CloseEnoughRange)
						dist = CloseEnoughRange;
					totaldisttrans += dist;
					totaltimetrans += 1;
				}

			}
			outfile << x1 << " " << x2 << " " << 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
			totalfittrans += 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			totaltrials += 1;
		}
	}
	outfile.close();
	return totalfittrans/totaltrials;
}

double AgentArena(TVector<double> &genotype1, TVector<double> &genotype2, double a1pos, double a2pos)
{

	std::string afilename = "Agent_Positions.dat";
	std::string sfilename = "sensor_output.dat";
	std::string N1filename = "Agent_1_NS.dat";
	std::string N2filename = "Agent_2_NS.dat";

	std::ofstream afile(afilename.c_str());
	std::ofstream sfile(sfilename.c_str());
	std::ofstream N1file(N1filename.c_str());
	std::ofstream N2file(N2filename.c_str());


	// Map genootype to phenotype
	TVector<double> phenotype1;
	phenotype1.SetBounds(1, VectSize);
	GenPhenMapping(genotype1, phenotype1);

	// Map genotype to phenotype
	TVector<double> phenotype2;
	phenotype2.SetBounds(1, VectSize);
	GenPhenMapping(genotype2, phenotype2);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype1(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype2(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype1(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype2(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype1(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype2(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype1(k));
		Agent2.SetSensorWeight(i,phenotype2(k));
		k++;
	}

	double shadow1, shadow2;

			// Set agents positions
			Agent1.Reset(a1pos);
			Agent2.Reset(a2pos);

	for (double time = 0; time < RunDuration; time += StepSize)
				{
					// Update shadow positions
					shadow1 = Agent1.pos + Shadow;
					if (shadow1 >= SpaceSize)
						shadow1 = shadow1 - SpaceSize;
					if (shadow1 < 0.0)
						shadow1 = SpaceSize + shadow1;

					// XXX shadow2 = Agent2.pos - Shadow;
					shadow2 = Agent2.pos + Shadow;
					if (shadow2 >= SpaceSize)
						shadow2 = shadow2 - SpaceSize;
					if (shadow2 < 0.0)
						shadow2 = SpaceSize + shadow2;

					// Sense
					Agent1.Sense(Agent2.pos, shadow2, Fixed2);
					Agent2.Sense(Agent1.pos, shadow1, Fixed1);

					// Move
					Agent1.Step(StepSize);
					Agent2.Step(StepSize);

					// Save

					afile << Agent1.pos << " " << Agent2.pos << endl;//" " << shadow1 << " " << shadow2 << " " << Fixed1 << " " << Fixed2 << " " << endl;
					sfile << Agent1.sensor << " " << Agent2.sensor << " " << endl;

					// n1file << Agent1.sensor << " " << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(3) << endl;
					// n2file << Agent1.NervousSystem.NeuronOutput(2) - Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(3) << " " << Agent1.sensor << endl;
			
					N1file << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(2) << " " << Agent1.NervousSystem.NeuronOutput(3) << endl; //" " << Agent1.NervousSystem.NeuronOutput(4) << " " <<Agent1.NervousSystem.NeuronOutput(5) << " " << endl;
					N2file << Agent2.NervousSystem.NeuronOutput(1) << " " << Agent2.NervousSystem.NeuronOutput(2) << " " << Agent2.NervousSystem.NeuronOutput(3) << endl; //" " << Agent1.NervousSystem.NeuronOutput(4) << " " <<Agent1.NervousSystem.NeuronOutput(5) << " " << endl;
				}
}

// ------------------------------------
// 3. Decoy analysis
// ------------------------------------
void DecoyMap(TVector<double> &genotype, double frequency)
{
	// Start output file
	ofstream fitfile("dm_fit_freq_"+to_string(frequency)+".dat");
	ofstream fittransfile("dm_fittran_freq_"+to_string(frequency)+".dat");
	ofstream distfile("dm_dist_freq_"+to_string(frequency)+".dat");
	ofstream crossfile("dm_cross_freq_"+to_string(frequency)+".dat");

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	double pos, pastpos;

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double totalfit = 0.0, totalfittrans = 0.0, totaldist = 0.0, totaldisttrans = 0.0, dist = 0.0;
	int totaltrials = 0, totaltime = 0, totaltimetrans;
	double shadow1, shadow2;
	double totalcross = 0;
	int crosscounter;
	double avgtotaldist = 0.0, avgtotaldisttrans = 0.0, avgcrosses = 0.0;
	int totalshadows = 0;

	for (double amplitude = 0; amplitude <= 4.0; amplitude += 0.01) {
		for (double velocity = -2; velocity <= 2.0; velocity += 0.01) {

			// Set agents positions
			Agent1.Reset(0.0);
			pos = 300.0;

			// Run the sim
			totaldist = 0.0;
			totaltime = 0;
			totaldisttrans = 0.0;
			totaltimetrans = 0;
			crosscounter = 0;

			for (double time = 0; time < RunDuration; time += StepSize)
			{
				// Move decoy
				pastpos = pos;
				pos += StepSize * (velocity + (amplitude * sin(frequency*time)));
				// Wrap-around Environment
				if (pos >= SpaceSize)
					pos = pos - SpaceSize;
				if (pos < 0.0)
					pos = SpaceSize + pos;

				// Sense
				Agent1.Sense(pos, 999999999, 999999999);

				// Move
				Agent1.Step(StepSize);

				// Measure number of times the agents cross paths
				if (((Agent1.pastpos < pastpos) && (Agent1.pos >= pos)) || ((Agent1.pastpos > pastpos) && (Agent1.pos <= pos)))
				{
					crosscounter += 1;
				}

				// Measure distance between them
				dist = fabs(pos - Agent1.pos);
				if (dist > HalfSpace)
					dist =  SpaceSize - dist;
				totaldist += dist;
				totaltime += 1;

				// Measure distance also for the fitness calc
				if (time > TransDuration)
				{
					if (dist < CloseEnoughRange)
						dist = CloseEnoughRange;
					totaldisttrans += dist;
					totaltimetrans += 1;
				}

			}

			// Save the results
			fitfile << amplitude << " " << velocity << " " << 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
			fittransfile << amplitude << " " << velocity << " " << 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
			distfile <<  amplitude << " " << velocity << " " << Agent1.pos << endl;
			crossfile << amplitude << " " << velocity << " " << crosscounter << endl;
			totalfit += 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			totalfittrans += 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			totaltrials += 1;
		}
	}
	fitfile.close();
	fittransfile.close();
	distfile.close();
	crossfile.close();
	cout << "Robust performance: " << totalfit/totaltrials << " " << totalfittrans/totaltrials << endl;
}

void DecoyMapFixedVel(TVector<double> &genotype)
{
	// Start output file
	ofstream fittransfile("dm_fittran_vel0.dat");
	ofstream crossfile("dm_cross_vel0.dat");

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);


	// Create the agents
	PerceptualCrosser Agent1(1,N);
	double pos, pastpos;

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double totalfittrans = 0.0, totaldisttrans = 0.0, dist = 0.0;
	int totaltrials = 0, totaltimetrans;
	double shadow1, shadow2;
	double totalcross = 0;
	int crosscounter;

	for (double frequency = 0.0; frequency <= 1.0; frequency += 0.005) {
		for (double amplitude = 0.0; amplitude <= 1.0; amplitude += 0.005) {

			// Set agents positions
			Agent1.Reset(0.0);
			pos = 100.0;

			// Run the sim
			totaldisttrans = 0.0;
			totaltimetrans = 0;
			crosscounter = 0;

			for (double time = 0; time < RunDurationMap; time += StepSize)
			{
				// Move decoy
				pastpos = pos;
				pos = 100 + amplitude * sin(frequency*time);
				// pos = 300;
				// // Wrap-around Environment
				// if (pos >= SpaceSize)
				// 	pos = pos - SpaceSize;
				// if (pos < 0.0)
				// 	pos = SpaceSize + pos;

				// Sense
				Agent1.Sense(pos, 999999999, 999999999);

				// Move
				Agent1.Step(StepSize);

				// Measure distance between them
				dist = fabs(pos - Agent1.pos);
				if (dist > HalfSpace)
					dist =  SpaceSize - dist;

				// Measure distance also for the fitness calc
				if (time > TransDurationMap)
				{
					if (dist < CloseEnoughRange)
						dist = CloseEnoughRange;
					totaldisttrans += dist;
					totaltimetrans += 1;

					// Measure number of times the agents cross paths
					if (((Agent1.pastpos < pastpos) && (Agent1.pos >= pos)) || ((Agent1.pastpos > pastpos) && (Agent1.pos <= pos)))
					{
						crosscounter += 1;
					}					
				}

			}
			double fit;
			// Save the results
			fit = 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			if (fit < 0.97)
				fit = 0.5;
			fittransfile << amplitude << " " << frequency << " " << fit << endl;
			crossfile << amplitude << " " << frequency << " " << crosscounter << endl;

		}
	}
	fittransfile.close();
	crossfile.close();
}

double Handedness(TVector<double> &genotype)
{
	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double shadow1, shadow2;
	double x1 = 300.0; 

	// Set agents positions
	Agent1.Reset(x1);

	// Run the sim
	for (double time = 0; time < 200; time += StepSize)
	{
		Agent1.Step(StepSize);
	}
	cout << Agent1.NervousSystem.NeuronOutput(2) << " " << Agent1.NervousSystem.NeuronOutput(1) << endl;
	if ((Agent1.NervousSystem.NeuronOutput(2) - Agent1.NervousSystem.NeuronOutput(1)) > 0){
		return 1.0;
	}
	else{
		return -1.0;
	}
}

void NeuralTracesGeno(TVector<double> &genotype, double a1pos, double a2pos, const std::string &filename)
{

	std::string afilename = "TraceData/" + filename + "Agent_Positions.dat";
	std::string sfilename = "TraceData/" + filename + "sensor_output.dat";
	std::string N1filename = "TraceData/" + filename + "Agent_1_NS.dat";
	std::string N2filename = "TraceData/" + filename + "Agent_2_NS.dat";

	std::ofstream afile(afilename.c_str());
	std::ofstream sfile(sfilename.c_str());
	std::ofstream N1file(N1filename.c_str());
	std::ofstream N2file(N2filename.c_str());

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double shadow1, shadow2;

			// Set agents positions
			Agent1.Reset(a1pos);
			Agent2.Reset(a2pos);

	for (double time = 0; time < RunDuration; time += StepSize){


					double direction = 1;

					Agent1.SetDirection(direction);

					// Update shadow positions
					shadow1 = Agent1.pos + Shadow;
					if (shadow1 >= SpaceSize)
						shadow1 = shadow1 - SpaceSize;
					if (shadow1 < 0.0)
						shadow1 = SpaceSize + shadow1;

					// XXX shadow2 = Agent2.pos - Shadow;
					shadow2 = Agent2.pos + Shadow;
					if (shadow2 >= SpaceSize)
						shadow2 = shadow2 - SpaceSize;
					if (shadow2 < 0.0)
						shadow2 = SpaceSize + shadow2;

					// Sense
					// Agent1.Sense(Agent2.pos, shadow2, Fixed2);
					Agent1.Sense(Agent2.pos, 999999999, 999999999); // Only other agent
					// Agent1.Sense(999999999, 999999999, Fixed2); // Only fixed object 
					// Agent1.Sense(999999999, shadow2, 999999999); // Only shadow 

					
					// Agent2.Sense(Agent1.pos, shadow1, Fixed1);
					Agent2.Sense(Agent1.pos, 999999999, 999999999);
					// Can't sense Shadow:
					// Agent1.Sense(Agent2.pos, 999999999, Fixed2);
					// Agent2.Sense(Agent1.pos, 999999999, Fixed1);


					// Move
					Agent1.Step(StepSize);
					Agent2.Step(StepSize);

		// if (time > 150.0 && time < 250) {
		// Save
		afile << Agent1.pos << " " << Agent2.pos << endl;//" " << shadow1 << " " << shadow2 << " " << Fixed1 << " " << Fixed2 << " " << endl;

		// For Position neural traces plots. 
		// afile << Agent1.pos << " " << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(2) << " " << Agent1.NervousSystem.NeuronOutput(3) << endl;

	
		// afile << time << " " << Agent1.pos << " " << Agent1.NervousSystem.NeuronOutput(1) << " " << endl;
		// sfile << time << " " << Agent2.pos << " " << Agent2.NervousSystem.NeuronOutput(1) << " " << endl;
		sfile << Agent1.sensor << " " << Agent2.sensor << " " << endl;

		//For Position Neural traces plots. 
		// sfile << Agent2.pos << " " << Agent2.NervousSystem.NeuronOutput(1) << " " << Agent2.NervousSystem.NeuronOutput(2) << " " << Agent2.NervousSystem.NeuronOutput(3) << endl;

		// n1file << Agent1.sensor << " " << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(3) << endl;
		// n2file << Agent1.NervousSystem.NeuronOutput(2) - Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(3) << " " << Agent1.sensor << endl;

		N1file << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(2) << " " << Agent1.NervousSystem.NeuronOutput(3) << " " << direction *  2 * ( Agent1.NervousSystem.NeuronOutput(2) - Agent1.NervousSystem.NeuronOutput(1)) << endl;//<< " " << Agent1.NervousSystem.NeuronOutput(4) << " " <<Agent1.NervousSystem.NeuronOutput(5) << " " << endl;
		N2file << Agent2.NervousSystem.NeuronOutput(1) << " " << Agent2.NervousSystem.NeuronOutput(2) << " " << Agent2.NervousSystem.NeuronOutput(3) << " " << 2* ( Agent2.NervousSystem.NeuronOutput(2) - Agent2.NervousSystem.NeuronOutput(1)) <<endl;
		// N2file << Agent2.NervousSystem.NeuronOutput(1) << " " << Agent2.NervousSystem.NeuronOutput(2) << " " << Agent2.NervousSystem.NeuronOutput(3) << " " << 2*(Agent2.NervousSystem.NeuronOutput(1) - Agent2.NervousSystem.NeuronOutput(2)) << endl;//<< " " << Agent1.NervousSystem.NeuronOutput(4) << " " <<Agent1.NervousSystem.NeuronOutput(5) << " " << endl;
		// sfile << Agent1.NervousSystem.NeuronOutput(2) - Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(3) << " " << endl;
		// sfile << Agent2.NervousSystem.NeuronOutput(2) - Agent2.NervousSystem.NeuronOutput(1) << " " << Agent2.NervousSystem.NeuronOutput(3) << " " << endl;

		// } bracket for if statement that takes the data only for a certain time period.
	}
	// cout << Agent1.sensorweights << endl;
	afile.close();
	sfile.close();
	N1file.close();
	N2file.close();
}

void NeuralTraj(TVector<double> &genotype, double a1pos, double a2pos, const std::string &filename)
{

	std::string sfilename = "TraceData/" + filename + "NeuralTrajectories.dat";

	std::ofstream sfile(sfilename.c_str());

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double shadow1, shadow2;

			// Set agents positions
			Agent1.Reset(a1pos);
			Agent2.Reset(a2pos);

	for (double time = 0; time < RunDuration; time += StepSize)
				{
					// Update shadow positions
					shadow1 = Agent1.pos + Shadow;
					if (shadow1 >= SpaceSize)
						shadow1 = shadow1 - SpaceSize;
					if (shadow1 < 0.0)
						shadow1 = SpaceSize + shadow1;

					// XXX shadow2 = Agent2.pos - Shadow;
					shadow2 = Agent2.pos + Shadow;
					if (shadow2 >= SpaceSize)
						shadow2 = shadow2 - SpaceSize;
					if (shadow2 < 0.0)
						shadow2 = SpaceSize + shadow2;

					// Sense
					// Agent1.Sense(Agent2.pos, shadow2, Fixed2);
					Agent1.Sense(Agent2.pos, 999999999, 999999999); // Only other agent
					// Agent1.Sense(999999999, 999999999, Fixed2); // Only fixed object 
					// Agent1.Sense(999999999, shadow2, 999999999); // Only shadow 
					Agent2.Sense(Agent1.pos, shadow1, Fixed1);

					// Move
					Agent1.Step(StepSize);
					Agent2.Step(StepSize);

		// Save
		// Agent1.pos << " " << Agent2.pos << " " << shadow1 << " " << shadow2 << " " << Fixed1 << " " << Fixed2 << " " << endl;
		// sfile << Agent1.sensor << " " << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(3) << " " << endl;
		// sfile << Agent2.sensor << " " << Agent2.NervousSystem.NeuronOutput(1) << " " << Agent2.NervousSystem.NeuronOutput(3) << " " << endl;
		// n1file << Agent1.sensor << " " << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(3) << endl;
		sfile << Agent1.NervousSystem.NeuronOutput(2) - Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(3) << " " << endl;

//  << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(2) << " " << Agent1.NervousSystem.NeuronOutput(3) << " " << Agent1.NervousSystem.NeuronOutput(4) << " " <<Agent1.NervousSystem.NeuronOutput(5) << " " << endl;
	//  << Agent2.NervousSystem.NeuronOutput(1) << " " << Agent2.NervousSystem.NeuronOutput(2) << " " << Agent2.NervousSystem.NeuronOutput(3) << " " << Agent1.NervousSystem.NeuronOutput(4) << " " <<Agent1.NervousSystem.NeuronOutput(5) << " " << endl;
	}
	sfile.close();
}

void movingObject(TVector<double> &genotype, double a1pos, double a2pos, const std::string &filename)
{
    std::string afilename = "TraceData/" + filename + "Agent_Positions.dat";
    std::string sfilename = "TraceData/" + filename + "sensor_output.dat";
    std::string N1filename = "TraceData/" + filename + "Agent_1_NS.dat";

    std::ofstream afile(afilename.c_str());
    std::ofstream sfile(sfilename.c_str());
    std::ofstream N1file(N1filename.c_str());

 	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

// Create the agents
	PerceptualCrosser Agent1(-1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

    // Define a simple moving object (Agent2)
    double agent2Speed = 1.5; 

    // Set initial positions of Agent1 and Agent2
    Agent1.Reset(a1pos);
	// Agent2.Reset(0.0);

    double Agent2Pos = a2pos; 
	// for(agent2Speed = 0.0; agent2Speed <= 3.0; agent2Speed += 0.05){
		for (double time = 0; time < RunDurationMap; time += StepSize)
		{
		Agent1.Sense(Agent2.pos, Agent2Pos, 999999999);
		// Agent1.Sense(999999999, Agent2Pos, 999999999);
		// Agent2.Sense(Agent1.pos, 999999999, 999999999);


        // Update positions
        Agent1.Step(StepSize);
		// Agent2.Step(StepSize);

// Moving object:

        // Agent2Pos += agent2Speed * StepSize;
        // if (Agent2Pos >= SpaceSize)
        //     Agent2Pos -= SpaceSize;
        // if (Agent2Pos < 0.0)
        //     Agent2Pos += SpaceSize;

// Oscillatory object: 

		double amplitude = .50;
		double frequency = 2.0;

		Agent2Pos = 100 + amplitude * sin(frequency*time); 


        // Save positions and neural outputs of Agent1
        afile << Agent1.pos << " " << Agent2Pos << endl;
		// afile << Agent1.pos << " " << Agent2.pos << " " << Agent2Pos << endl;
        sfile << Agent1.sensor << " " << endl;
        N1file << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(2) << " " << Agent1.NervousSystem.NeuronOutput(3) << endl;
		// N1file << Agent1.NervousSystem.NeuronOutput(2) - Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(3) << " " << Agent1.sensor << endl;
    }

    afile.close();
    sfile.close();
    N1file.close();
}

void OscillatoryDecoy(TVector<double> &genotype, const std::string &filename)
{

	std::string ffilename = "TraceData/" + filename + "decoy_fit.dat";
	std::string cfilename = "TraceData/" + filename + "decoy_cross.dat";

	std::ofstream fittransfile(ffilename.c_str(), std::ofstream::trunc);
	std::ofstream crossfile(cfilename.c_str(), std::ofstream::trunc);

// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	// Decoy Params 
	double pos, pastpos;

	// evaluation params 
	double totalfittrans = 0.0, totaldisttrans = 0.0, dist = 0.0;
	int totaltrials = 0, totaltimetrans;
	double shadow1, shadow2;
	double totalcross = 0;
	int crosscounter;

	for (double frequency = 0.0; frequency <= 2.0; frequency += 0.01) { // 0.005
		for (double amplitude = 0.0; amplitude <= 2.0; amplitude += 0.01) {

			// Set agents positions
			Agent1.Reset(300);
			// Agent2.Reset(250);
			// Set Decoy position
			pos = 400.0;

			// Run the sim
			totaldisttrans = 0.0;
			totaltimetrans = 0;
			crosscounter = 0;

			for (double time = 0; time < RunDurationMap; time += StepSize)
			{
				// Move decoy
				pastpos = pos;
				pos = 100 + amplitude * sin(frequency*time);
				// pos = 300;
				// // Wrap-around Environment
				// if (pos >= SpaceSize)
				// 	pos = pos - SpaceSize;
				// if (pos < 0.0)
				// 	pos = SpaceSize + pos;

				// Sense
				Agent1.Sense(Agent2.pos, pos, 999999999);

				// Notice that the other agent cannot sense the decoy 
				// Agent2.Sense(Agent1.pos, 999999999, 999999999);

				// Move
				Agent1.Step(StepSize);
				// Agent2.Step(StepSize);

				// Measure distance between them
				dist = fabs(pos - Agent1.pos);
				if (dist > HalfSpace)
					dist =  SpaceSize - dist;

				// Measure distance also for the fitness calc
				if (time > TransDurationMap)
				{
					if (dist < CloseEnoughRange)
						dist = CloseEnoughRange;
					totaldisttrans += dist;
					totaltimetrans += 1;

					// Measure number of times the agents cross paths
					if (((Agent1.pastpos < pastpos) && (Agent1.pos >= pos)) || ((Agent1.pastpos > pastpos) && (Agent1.pos <= pos)))
					{
						crosscounter += 1;
					}					
				}

			}
			double fit;
			// Save the results
			fit = 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));

			fittransfile << amplitude << " " << frequency << " " << fit << endl;
			crossfile << amplitude << " " << frequency << " " << crosscounter << endl;

		}
	}
	fittransfile.close();
	crossfile.close();
}

void MovingDecoyAnalysis(TVector<double> &genotype)
{
    // Start output file
    ofstream fittransfile("moving_decoy_fit.dat");
    ofstream crossfile("moving_decoy_cross.dat");

    // Map genotype to phenotype
    TVector<double> phenotype;
    phenotype.SetBounds(1, VectSize);
    GenPhenMapping(genotype, phenotype);

    // Create the agents
    PerceptualCrosser Agent1(1, N);
    PerceptualCrosser Agent2(-1, N);

    // Instantiate the nervous systems
    Agent1.NervousSystem.SetCircuitSize(N);
    Agent2.NervousSystem.SetCircuitSize(N);
    int k = 1;

    // Time-constants
    for (int i = 1; i <= N; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i, phenotype(k));
        Agent2.NervousSystem.SetNeuronTimeConstant(i, phenotype(k));
        k++;
    }
    // Biases
    for (int i = 1; i <= N; i++) {
        Agent1.NervousSystem.SetNeuronBias(i, phenotype(k));
        Agent2.NervousSystem.SetNeuronBias(i, phenotype(k));
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i, j, phenotype(k));
            Agent2.NervousSystem.SetConnectionWeight(i, j, phenotype(k));
            k++;
        }
    }
    // Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent1.SetSensorWeight(i, phenotype(k));
        Agent2.SetSensorWeight(i, phenotype(k));
        k++;
    }

    double pos, pastpos;

    // evaluation params 
    double totalfittrans = 0.0, totaldisttrans = 0.0, dist = 0.0;
    int totaltrials = 0, totaltimetrans;
    double shadow1, shadow2;
    double totalcross = 0;
    int crosscounter;

    for (double agent2Speed = -2.00; agent2Speed <= 2.00; agent2Speed += 0.01) {
       
	    if (agent2Speed < 0.0){Agent1.SetDirection(1);} // Flip agent direction to be opposite to decoy 
		else{Agent1.SetDirection(-1);}
		

        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << agent2Speed;
        std::string speedStr = ss.str();

        // Open file for each speed
        ofstream N1file("NS_Traces/Agent_1_NS_" + speedStr + ".dat");

        // Set agents positions
        Agent1.Reset(300);
        pos = 400.0;

        // Run the sim
        totaldisttrans = 0.0;
        totaltimetrans = 0;
        crosscounter = 0;

        for (double time = 0; time < RunDurationMap; time += StepSize) {
            // Move decoy
            pastpos = pos;
            pos += agent2Speed * StepSize;
            if (pos >= SpaceSize)
                pos -= SpaceSize;
            if (pos < 0.0)
                pos += SpaceSize;

            // Sense
            Agent1.Sense(pos, 999999999, 999999999);

            // Move
            Agent1.Step(StepSize);

            // Save the nervous system outputs
			//2D Plots 
            // N1file << Agent1.NervousSystem.NeuronOutput(2) - Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(3) << " " << endl;
			// 3D Plots
			N1file << Agent1.sensor << " " << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(2) << " " << endl;
            // Measure distance between them
            dist = fabs(pos - Agent1.pos);
            if (dist > HalfSpace)
                dist = SpaceSize - dist;

            // Measure distance also for the fitness calc
            if (time > TransDurationMap) {
                if (dist < CloseEnoughRange)
                    dist = CloseEnoughRange;
                totaldisttrans += dist;
                totaltimetrans += 1;

                // Measure number of times the agents cross paths
                if (((Agent1.pastpos < pastpos) && (Agent1.pos >= pos)) || ((Agent1.pastpos > pastpos) && (Agent1.pos <= pos))) {
                    crosscounter += 1;
                }
            }
        }

        double fit;
        // Save the results
        fit = 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
        if (fit < 0.9) { fit = 0.0; }
        int crosses = crosscounter;
        if (crosses < 4) { crosses = 0; }
        fittransfile << agent2Speed << " " << fit << endl;
        crossfile << agent2Speed << " " << crosses << endl;

        N1file.close();
    }
	
    fittransfile.close();
    crossfile.close();
}

void TimescaleAnalysis(TVector<double> &genotype)
{
    // Start output file
    ofstream fittransfile("timescale_fit.dat");
    ofstream crossfile("timescale_cross.dat");

    // Map genotype to phenotype
    TVector<double> phenotype;
    phenotype.SetBounds(1, VectSize);
    GenPhenMapping(genotype, phenotype);

    // Create the agents
    PerceptualCrosser Agent1(1, N);
    PerceptualCrosser Agent2(-1, N);

    // Instantiate the nervous systems
    Agent1.NervousSystem.SetCircuitSize(N);
    Agent2.NervousSystem.SetCircuitSize(N);
    int k = 1;

    // Time-constants
    for (int i = 1; i <= N; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i, phenotype(k));
        Agent2.NervousSystem.SetNeuronTimeConstant(i, phenotype(k));
        k++;
    }
    // Biases
    for (int i = 1; i <= N; i++) {
        Agent1.NervousSystem.SetNeuronBias(i, phenotype(k));
        Agent2.NervousSystem.SetNeuronBias(i, phenotype(k));
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i, j, phenotype(k));
            Agent2.NervousSystem.SetConnectionWeight(i, j, phenotype(k));
            k++;
        }
    }
    // Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent1.SetSensorWeight(i, phenotype(k));
        Agent2.SetSensorWeight(i, phenotype(k));
        k++;
    }

    double pos = 0.0, pastpos = 0.0;

    // evaluation params 
    double totalfittrans = 0.0, totaldisttrans = 0.0, dist = 0.0;
    int totaltrials = 0, totaltimetrans;
    double shadow1, shadow2;
    double totalcross = 0;
    int crosscounter;

    
    for (double timescaleStep = 0.0001; timescaleStep <= 0.5; timescaleStep += 0.0001) {
		// for (double timescaleStep = 0.0001; timescaleStep <= 0.14; timescaleStep *= 1.01){
        
        Agent1.Reset(300);
        Agent2.Reset(400);

        
        totaldisttrans = 0.0;
        totaltimetrans = 0;
        crosscounter = 0;

        for (double time = 0; time < RunDurationMap; time += StepSize) {

            // Sense
            Agent1.Sense(Agent2.pos, 999999999, 999999999);
            Agent2.Sense(Agent1.pos, 999999999, 999999999);

            // Move
            Agent1.Step(StepSize);
            Agent2.Step(timescaleStep);

            // Measure distance between them
            dist = fabs(Agent2.pos - Agent1.pos);
            if (dist > HalfSpace)
                dist = SpaceSize - dist;

            // Measure distance also for the fitness calc
            if (time > TransDurationMap) {
                if (dist < CloseEnoughRange)
                    dist = CloseEnoughRange;
                totaldisttrans += dist;
                totaltimetrans += 1;

                // Measure number of times the agents cross paths
                if (((Agent1.pastpos < Agent2.pastpos) && (Agent1.pos >= Agent2.pos)) || ((Agent1.pastpos > Agent2.pastpos) && (Agent1.pos <= Agent2.pos))) {
                    crosscounter += 1;
                }
            }
        }

        double fit;
       
        if (totaltimetrans > 0) {
            fit = 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
            if (fit < 0.95) {fit = 0.0;}
        } else {
            fit = 0.0;
        }

        int crosses = crosscounter;
        if (crosses < 4) {crosses = 0;}
        fittransfile << timescaleStep << " " << fit << endl;
        crossfile << timescaleStep << " " << crosses << endl;
    }
    fittransfile.close();
    crossfile.close();
}

void VelAnalysis(TVector<double> &genotype)
{
    // Start output file
    ofstream fittransfile("TraceData/vel_fit.dat");
    ofstream crossfile("TraceData/vel_cross.dat");

    // Map genotype to phenotype
    TVector<double> phenotype;
    phenotype.SetBounds(1, VectSize);
    GenPhenMapping(genotype, phenotype);

    // Create the agents
    PerceptualCrosser Agent1(1, N);
    PerceptualCrosser Agent2(-1, N);

    // Instantiate the nervous systems
    Agent1.NervousSystem.SetCircuitSize(N);
    Agent2.NervousSystem.SetCircuitSize(N);
    int k = 1;

    // Time-constants
    for (int i = 1; i <= N; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i, phenotype(k));
        Agent2.NervousSystem.SetNeuronTimeConstant(i, phenotype(k));
        k++;
    }
    // Biases
    for (int i = 1; i <= N; i++) {
        Agent1.NervousSystem.SetNeuronBias(i, phenotype(k));
        Agent2.NervousSystem.SetNeuronBias(i, phenotype(k));
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i, j, phenotype(k));
            Agent2.NervousSystem.SetConnectionWeight(i, j, phenotype(k));
            k++;
        }
    }
    // Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent1.SetSensorWeight(i, phenotype(k));
        Agent2.SetSensorWeight(i, phenotype(k));
        k++;
    }

    double pos = 0.0, pastpos = 0.0;

    // evaluation params 
    double totalfittrans = 0.0, totaldisttrans = 0.0, dist = 0.0;
    int totaltrials = 0, totaltimetrans;
    double shadow1, shadow2;
    double totalcross = 0;
    int crosscounter;

    
    for (double vel = 0.8; vel <= 1.3; vel += 0.001) {
        Agent1.SetDirection(vel);    
        // Set agents positions
        Agent1.Reset(400);
        Agent2.Reset(300);

        // Track the resting velocity observed during the trial
        double resting_velocity = 0.0;

        // Run the sim
        totaldisttrans = 0.0;
        totaltimetrans = 0;
        crosscounter = 0;

        for (double time = 0; time < RunDurationMap; time += StepSize) {

            double current_velocity = Agent1.Velocity();
            if (time > 50 && time < 55) {
                resting_velocity = current_velocity;
            }
            
            // Sense
            Agent1.Sense(Agent2.pos, 999999999, 999999999);
            Agent2.Sense(Agent1.pos, 999999999, 999999999);

            // Move
            Agent1.Step(StepSize);
            Agent2.Step(StepSize);

            // Measure distance between them
            dist = fabs(Agent2.pos - Agent1.pos);
            if (dist > HalfSpace)
                dist = SpaceSize - dist;

            // Measure distance also for the fitness calc
            if (time > TransDurationMap) {
                if (dist < CloseEnoughRange)
                    dist = CloseEnoughRange;
                totaldisttrans += dist;
                totaltimetrans += 1;

                // Measure number of times the agents cross paths
                if (((Agent1.pastpos < Agent2.pastpos) && (Agent1.pos >= Agent2.pos)) || ((Agent1.pastpos > Agent2.pastpos) && (Agent1.pos <= Agent2.pos))) {
                    crosscounter += 1;
                }
            }
        }

        double fit;
        if (totaltimetrans > 0) {
            fit = 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
            if (fit < 0.95) {fit = 0.0;}
        } else {
            fit = 0.0;
        }

        int crosses = crosscounter;
        if (crosses < 4) {crosses = 0;}

        fittransfile << resting_velocity << " " << fit << endl;
		crossfile << vel << " " << resting_velocity << endl;
    }
    
    fittransfile.close();
    crossfile.close();
}

void FeedbackAnalysisTrace(TVector<double> &genotype, double a1pos, double a2pos, const std::string &filename)
{
    std::string afilename = "TraceData/" + filename + "Agent_Positions.dat";
    std::string sfilename = "TraceData/" + filename + "sensor_output.dat";
    std::string N1filename = "TraceData/" + filename + "Agent_1_NS.dat";

    std::ofstream afile(afilename.c_str());
    std::ofstream sfile(sfilename.c_str());
    std::ofstream N1file(N1filename.c_str());

    // Map genotype to phenotype
    TVector<double> phenotype;
    phenotype.SetBounds(1, VectSize);
    GenPhenMapping(genotype, phenotype);

    // Create the agents
    PerceptualCrosser Agent1(1, N);
    PerceptualCrosser Agent2(-1, N);

    // Instantiate the nervous systems
    Agent1.NervousSystem.SetCircuitSize(N);
    Agent2.NervousSystem.SetCircuitSize(N);
    int k = 1;

    // Time-constants
    for (int i = 1; i <= N; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i, phenotype(k));
        Agent2.NervousSystem.SetNeuronTimeConstant(i, phenotype(k));
        k++;
    }
    // Biases
    for (int i = 1; i <= N; i++) {
        Agent1.NervousSystem.SetNeuronBias(i, phenotype(k));
        Agent2.NervousSystem.SetNeuronBias(i, phenotype(k));
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i, j, phenotype(k));
            Agent2.NervousSystem.SetConnectionWeight(i, j, phenotype(k));
            k++;
        }
    }
    // Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent1.SetSensorWeight(i, phenotype(k));
        Agent2.SetSensorWeight(i, phenotype(k));
        k++;
    }

    Agent1.Reset(a1pos);

	double vel = -1.36;
	double feedback = 1.0; // should be between -2.63 and 2.63? // seems that feedback in opposite direction will always result in stable interaction. 
	double delayDuration = 1.0;  // Duration of the delay before returning to base velocity
	double InteractionThreshold = 1.0; 

    double objectPos = a2pos;
    double objectVel = vel;
    const double baseVel = vel;

    bool interacting = false;
    double interactionTimer = 0.0;  // Timer to track interaction delay

    for (double time = 0; time < RunDuration; time += StepSize)
    {
        Agent1.Sense(999999999, objectPos, 999999999);
        Agent1.Step(StepSize);

        // Check if within sensory range
        if (abs(Agent1.pos - objectPos) < InteractionThreshold) {
           
		    // Start the interaction timer
            interacting = true;
            interactionTimer = 0.0;  // Reset the timer
            objectVel = baseVel + feedback;  // Apply feedback velocity
        }

        if (interacting) {
            interactionTimer += StepSize;

            // Check if the delay duration has passed
            if (interactionTimer >= delayDuration) {
                objectVel = baseVel;  // Return to base velocity after delay
                interacting = false;  // Reset interaction state
            }
        }

        // Update object position
        objectPos += objectVel * StepSize;

        if (objectPos >= SpaceSize)
            objectPos -= SpaceSize;
        if (objectPos < 0.0)
            objectPos += SpaceSize;

        afile << Agent1.pos << " " << objectPos << std::endl;
        sfile << Agent1.sensor << " " << objectVel << " " << Agent1.Velocity() << std::endl;
        N1file << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(2) << " " << Agent1.NervousSystem.NeuronOutput(3) << std::endl;
    }

    afile.close();
    sfile.close();
    N1file.close();
}

void FeedbackAnalysis(TVector<double> &genotype, const std::string &filename)
{


    std::string ffilename = "TraceData/" + filename + "feedback_fit.dat";
    std::string cfilename = "TraceData/" + filename + "feedback_cross.dat";

    std::ofstream fitoutfile(ffilename.c_str());
    std::ofstream crossoutfile(cfilename.c_str());

    // Map genotype to phenotype
    TVector<double> phenotype;
    phenotype.SetBounds(1, VectSize);
    GenPhenMapping(genotype, phenotype);

    // Create the agents
    PerceptualCrosser Agent1(1, N);
    PerceptualCrosser Agent2(-1, N);

    // Instantiate the nervous systems
    Agent1.NervousSystem.SetCircuitSize(N);
    Agent2.NervousSystem.SetCircuitSize(N);
    int k = 1;

    // Time-constants
    for (int i = 1; i <= N; i++) {
        Agent1.NervousSystem.SetNeuronTimeConstant(i, phenotype(k));
        Agent2.NervousSystem.SetNeuronTimeConstant(i, phenotype(k));
        k++;
    }
    // Biases
    for (int i = 1; i <= N; i++) {
        Agent1.NervousSystem.SetNeuronBias(i, phenotype(k));
        Agent2.NervousSystem.SetNeuronBias(i, phenotype(k));
        k++;
    }
    // Weights
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            Agent1.NervousSystem.SetConnectionWeight(i, j, phenotype(k));
            Agent2.NervousSystem.SetConnectionWeight(i, j, phenotype(k));
            k++;
        }
    }
    // Sensor Weights
    for (int i = 1; i <= N; i++) {
        Agent1.SetSensorWeight(i, phenotype(k));
        Agent2.SetSensorWeight(i, phenotype(k));
        k++;
    }

	// evaluation params 
	double totalfittrans = 0.0, totaldisttrans = 0.0, dist = 0.0;
	int totaltrials = 0, totaltimetrans;
	double shadow1, shadow2;
	double totalcross = 0;
	int crosscounter;



	double pastpos;
    // const double baseVel = -1.36;
    const double InteractionThreshold = 1.0;

    // Iterate over feedback and return durations
    for (double feedback = 0.7; feedback <= 1.0; feedback += 0.001) {
        for (double delay = 3.0; delay <= 4.0; delay += 0.001) {
        // for (double vel = -2.0; vel <= 2.0; vel += 0.1) {   
			
			// if (vel < 0.0){Agent1.SetDirection(1);} // Flip agent direction to be opposite to decoy 
			// else{Agent1.SetDirection(-1);}
				double vel = -1.36;

            Agent1.Reset(100);
            double objectPos = 500;
            double objectVel = vel;   //baseVel;
			double baseVel = vel;
            bool interacting = false;
			
			
			double delayDuration = delay;   //delay;  // Duration of the delay before returning to base velocity
			double interactionTimer = 0.0;  // Timer to track interaction delay

            double totaldist = 0.0;
            int totaltimesteps = 0;
            int crosscounter = 0;

			totaldisttrans = 0.0;
			totaltimetrans = 0;
			crosscounter = 0;

            // Run the simulation
            for (double time = 0; time < RunDuration; time += StepSize)
            {

				pastpos = objectPos;

                Agent1.Sense(999999999, objectPos, 999999999);
                Agent1.Step(StepSize);

				// Check if within sensory range
				if (abs(Agent1.pos - objectPos) < InteractionThreshold) {
					// Start the interaction timer
					interacting = true;
					interactionTimer = 0.0;  // Reset the timer
					objectVel = baseVel + feedback;  // Apply feedback velocity
				}

				if (interacting) {
					interactionTimer += StepSize;

					// Check if the delay duration has passed
					if (interactionTimer >= delayDuration) {
						objectVel = baseVel;  // Return to base velocity after delay
						interacting = false;  // Reset interaction state
					}
				}

                // Update object position
                objectPos += objectVel * StepSize;

                // Wrap-around environment
                if (objectPos >= SpaceSize)
                    objectPos -= SpaceSize;
                if (objectPos < 0.0)
                    objectPos += SpaceSize;

                // Calculate distance and crossings
                double dist = fabs(objectPos - Agent1.pos);
                if (dist > HalfSpace)
                    dist = SpaceSize - dist;

				// Measure distance also for the fitness calc
				if (time > TransDuration)
				{
					if (dist < CloseEnoughRange)
						dist = CloseEnoughRange;
					totaldisttrans += dist;
					totaltimetrans += 1;

					// Measure number of times the agents cross paths
					if (((Agent1.pastpos < pastpos) && (Agent1.pos >= objectPos)) || ((Agent1.pastpos > pastpos) && (Agent1.pos <= objectPos)))
					{
						crosscounter += 1;
					}					
				}

			}
			double fit;
			fit = 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));

			double normalizedCrossings = static_cast<double>(crosscounter) / totaltimetrans;
			
            fitoutfile << feedback << " " << delay << " " << fit << endl;
            crossoutfile << feedback << " " << delay<< " " << crosscounter << endl;
			// fitoutfile << feedback << " " << vel << " " << fit << endl;
            // crossoutfile << feedback << " " << vel << " " << crosscounter << endl;
        }
    }
    fitoutfile.close();
    crossoutfile.close();
}

// ================================================
// C. ADDITIONAL EVOLUTIONARY FUNCTIONS
// ================================================
int TerminationFunction(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	if (BestPerf > 0.99) return 1;
	else return 0;
}

// ------------------------------------
// Display functions
// ------------------------------------
void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	cout << Generation << " " << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
}

void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	ofstream BestIndividualFile;
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	BestIndividualFile.open("best.gen.dat");
	BestIndividualFile << bestVector << endl;
	BestIndividualFile.close();

	// Also show the best individual in the Circuit Model form
	BestIndividualFile.open("best.ns.dat");
	GenPhenMapping(bestVector, phenotype);
	PerceptualCrosser Agent(1,N);

	// Instantiate the nervous system
	Agent.NervousSystem.SetCircuitSize(N);
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		Agent.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
		// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent.SetSensorWeight(i,phenotype(k));
		k++;
	}
	BestIndividualFile << Agent.NervousSystem << endl;
	BestIndividualFile << Agent.sensorweights << "\n" << endl;
	BestIndividualFile.close();
}

std::vector<std::string> getAllGenotypesFromDirectory(const std::string& directoryPath) {
    std::vector<std::string> genotypes;
    DIR* dirp = opendir(directoryPath.c_str());
    struct dirent * dp;
    std::regex pattern("Round(10|[1-9])_best\\.gen_(100|[0-9][0-9]?)_N3\\.dat"); // matches Round1-10_best.gen_0-100_N2.dat
    while ((dp = readdir(dirp)) != NULL) {
        std::string filename(dp->d_name);
        if(std::regex_match(filename, pattern)) { // check if the file matches the pattern
            genotypes.push_back(directoryPath + "/" + filename);
        }
    }
    closedir(dirp);
    return genotypes;
}

// ------------------------------------
// The main program
// ------------------------------------
int main (int argc, const char* argv[]) 
{

// ================================================
//                   EVOLUTION
// ================================================

	// long randomseed = static_cast<long>(time(NULL));
	// if (argc == 2)
	// 	randomseed += atoi(argv[1]);

	// TSearch s(VectSize);
	
	// #ifdef PRINTOFILE

	// ofstream file;
	// file.open("evol.dat");
	// cout.rdbuf(file.rdbuf());
	
	// // save the seed to a file
	// ofstream seedfile;
	// seedfile.open ("seed.dat");
	// seedfile << randomseed << endl;
	// seedfile.close();
	
	// #endif
	
	// // Configure the search
	// s.SetRandomSeed(randomseed);
	// s.SetSearchResultsDisplayFunction(ResultsDisplay);
	// s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
	// s.SetSelectionMode(RANK_BASED);
	// s.SetReproductionMode(GENETIC_ALGORITHM);
	// s.SetPopulationSize(POPSIZE);
	// s.SetMaxGenerations(GENS);
	// s.SetCrossoverProbability(CROSSPROB);
	// s.SetCrossoverMode(UNIFORM);
	// s.SetMutationVariance(MUTVAR);
	// s.SetMaxExpectedOffspring(EXPECTED);
	// s.SetElitistFraction(ELITISM);
	// s.SetSearchConstraint(1);
	
	// /* Stage 1 */
	// s.SetSearchTerminationFunction(TerminationFunction);
	// s.SetEvaluationFunction(FitnessFunction1); 
	// s.ExecuteSearch();
	// /* Stage 2 */
	// s.SetSearchTerminationFunction(NULL);
	// s.SetEvaluationFunction(FitnessFunction2);
	// s.ExecuteSearch();	

// ================================================
//                   ANALYSIS
// ================================================
	// ifstream genefile;
	// genefile.open("best.gen.dat");
	// TVector<double> genotype(1, VectSize);
	// genefile >> genotype;

	// Behavioral Examples by type: 

	// Minimal rd 1 A 85
	// Oscillator rd 2 A 3
	// Drifter rd rd 3 A 83

// Opening genotyes from 1000s EVOS file:

	// std::string user = "/Users/gjseveri/";
	std::string user = "/Users/gabriel_severino/";

	std::string round = "1";
	std::string agentNumber = "85";	

	ifstream genefile1;
	genefile1.open(user + "Desktop/Projects/01_Research_Projects/BBE/1000EVOS/BestPerfGeno/Round" + round + "_best.gen_" + agentNumber + "_N3.dat");
	TVector<double> genotype1(1, VectSize);
	genefile1 >> genotype1;

	ifstream genefile2;
	genefile2.open(user + "Desktop/Projects/01_Research_Projects/BBE/1000EVOS/BestPerfGeno/Round" + round + "_best.gen_" + agentNumber + "_N3.dat");
	TVector<double> genotype2(1, VectSize);
	genefile2 >> genotype2;
	
	// Saving Traces

	NeuralTracesGeno(genotype1, 100, 500, "01minimal_"); 

	// NeuralTracesGeno(genotype1, 500, 100, "01minimal_");     // oscillatory haha 
	// NeuralTraj(genotype1, 300, 250, "01minimal_traj_");

	// Analyses 

	// VelAnalysis(genotype1);
	// TimescaleAnalysis(genotype1);
	// movingObject(genotype1, 300.0, 400.0, "01minimal_");
	// FeedbackAnalysis(genotype1, "01minimal_");
	// FeedbackAnalysisTrace(genotype1, 100.0, 500.0, "01minimal_");
	// OscillatoryDecoy(genotype1, "01minimal_");
	// MovingDecoyAnalysis(genotype1);

//////// Traces in BULK: =========================================================================

/////// FOR ALL GENOTYPES IN FILE:

//     std::vector<std::string> genotypes = getAllGenotypesFromDirectory("/Users/gabriel_severino/Desktop/Projects/01_Research_Projects/BBE/1000EVOS/MinAgents");

// for (const auto& genotypeFile : genotypes) {
//     std::ifstream genefile(genotypeFile);
//     if (!genefile.is_open()) {
//         continue;  // Skip this file if it fails to open
//     }

//     TVector<double> genotype(1, VectSize);
//     genefile >> genotype;

//     if (genefile.fail()) {
//         continue;  // Skip to the next file if there was an error reading
//     }

//     // Analysis functions
//     OscillatoryDecoy(genotype, genotypeFile);
// 	// MovingDecoyAnalysis(genotype);
// 	// FeedbackAnalysis(genotype, genotypeFile);

// }




//     for (const auto& genotypeFile : genotypes) {
//         ifstream genefile;
//         genefile.open(genotypeFile);
//         TVector<double> genotype(1, VectSize);
//         genefile >> genotype;

//         // Call the BehavioralTraces function for each genotype
// 		// NeuralTracesGeno(genotype, 170, 500, genotypeFile);
// 		OscillatoryDecoy(genotype, genotypeFile);
// }


// For Non-Clonal Analyses: 


// AGENT ARENA:
// 	ifstream genefile1;
// 	genefile1.open("genotype1.dat");
// 	TVector<double> genotype1(1, VectSize);
// 	genefile1 >> genotype1;

// 	ifstream genefile2;
// 	genefile2.open("genotype2.dat");
// 	TVector<double> genotype2(1, VectSize);
// 	genefile2 >> genotype2;


// AgentArena(genotype1, genotype2, 350.0, 250.0);



//////////// 


//for making nervous systems 

	// ofstream nerv;
	// TVector<double> phenotype;
	// phenotype.SetBounds(1, VectSize);

	// // Also show the best individual in the Circuit Model form
	// nerv.open("nervsystem.dat");
	// GenPhenMapping(genotype1, phenotype);

	// nerv << phenotype << endl;
	// nerv.close();

	return 0;
}
