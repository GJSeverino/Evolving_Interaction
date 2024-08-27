#pragma once

#include "CTRNN.h"

// The PerceptualCrosser class declaration

class PerceptualCrosser {
	public:
		// The constructor
		PerceptualCrosser(double direction, int networksize)
		{
			Set(direction, networksize);
		};
		// The destructor
		~PerceptualCrosser() {};

		// Accessors
		double Position(void) {return pos;};
		double Direction(void) {return dir;};
		double Velocity(void) {return dir * gain * (NervousSystem.NeuronOutput(2) - NervousSystem.NeuronOutput(1));}
		void SetPosition(double newpos) {pos = newpos;};
		void SetDirection(double newdir) {dir = newdir;};
		void SetSensorWeight(int to, double value) {sensorweights[to] = value;};
		void SetSensorState(double state) {sensor = state;};

		// Control
        void Set(double direction, int networksize);
		void Reset(double initpos);
		void Sense(double other, double shadow, double fixed);
		void Step(double StepSize);

		int size;
		double pos, dir, gain, sensor, pastpos;
		TVector<double> sensorweights;
		CTRNN NervousSystem;
};
