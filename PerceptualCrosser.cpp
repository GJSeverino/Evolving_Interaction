#include "PerceptualCrosser.h"
#include "random.h"

// Constants
const double SpaceSize = 600;
const double HalfSpace = 300;
const double SenseRange = 2.0; 

// *******
// Control
// *******

// Init the agent
void PerceptualCrosser::Set(double direction, int networksize)
{
	size = networksize;
	gain = 2.0; 
	sensorweights.SetBounds(1, size);
	sensorweights.FillContents(0.0);
	pos = 0.0;
	pastpos = 0.0;
	dir = direction;
	sensor = 0.0;
}

// Reset the state of the agent
void PerceptualCrosser::Reset(double initpos)
{
	pos = initpos;
	pastpos = initpos;
	sensor = 0.0;
	NervousSystem.RandomizeCircuitState(0.0,0.0);
}

// Sense (New extension -- Continouos)
void PerceptualCrosser::Sense(double Other, double Shadow, double Fixed)
{
	// Sense
	double distA, distS, distO, mindist;

	// Distance to other agent
	if (Other == 999999999){ // If Other is 999999999, then there's actually no other agent
		distA =  999999999;
	}	
	else{	
		distA = fabs(Other - pos);
		if (distA > HalfSpace)
			distA =  SpaceSize - distA;
	}

	// Distance to shadow
	if (Shadow == 999999999){ // If Shadow is 999999999, then there's actually no shadow
		distS =  999999999;
	}
	else{
		distS = fabs(Shadow - pos);
		if (distS > HalfSpace)
			distS =  SpaceSize - distS;
	}

	// Distance to fixed object
	if (Fixed == 999999999){ // If Fixed is 999999999, then there's actually no fixed object
		distO =  999999999;
	}
	else{
		distO = fabs(Fixed - pos);
		if (distO > HalfSpace)
			distO =  SpaceSize - distO;
	}

	// Figure out which of the distances is the smallest
	if ((distA <= distS) && (distA <= distO))
	{
		mindist = distA;
	}
	else
	{
		if ((distS <= distA) && (distS <= distO))
		{
			mindist = distS;
		}		
		else
		{
			mindist = distO;
		}
	}

	sensor = 1/(1 + exp(8 * (mindist - 1)));

	// // Cut off sensor value after sense range
	// if (mindist > SenseRange)
	// {
	// 	sensor = 0.0;
	// }
	// else
	// {
	// 	// Transfer the distance with a continuous sigmoidal function 
	// 	// With this sigma function, the sensor moves from 0.000335 (when dist is 2) to 0.99966 (when dist is 0)
	// 	sensor = 1/(1 + exp(8 * (mindist - 1)));
	// }
}


// Step
void PerceptualCrosser::Step(double StepSize)
{
    // Remember past position
    pastpos = pos;

	// Set sensor to external input
	for (int i = 1; i <= size; i++)
		NervousSystem.SetNeuronExternalInput(i, sensor*sensorweights[i]);

	// Update the nervous system
	NervousSystem.EulerStep(StepSize);

	// Update the body position
	pos += StepSize * dir * gain * (NervousSystem.NeuronOutput(2) - NervousSystem.NeuronOutput(1));

	// Wrap-around Environment
	if (pos >= SpaceSize)
		pos = pos - SpaceSize;
	if (pos < 0.0)
		pos = SpaceSize + pos;
}
