/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;

	// Create normal distributions for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// Set the number of particles
	num_particles = 100;

	// Initialize particles
	for(int i = 0; i < num_particles; ++i) {
		Particle particle;

		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.id = i;
		particle.weight = 1.0;

		particles.push_back(particle);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	// Create normal distributions for x, y and theta
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	// Predict each particle by motion models
	for(int i = 0; i < num_particles; ++i) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		if(yaw_rate) {	// yaw rate is non-zero
			double theta_f = theta + yaw_rate * delta_t;
			particles[i].x = x + velocity / yaw_rate * ( sin(theta_f) - sin(theta) ) + dist_x(gen);
			particles[i].y = y + velocity / yaw_rate * ( cos(theta) - cos(theta_f) ) + dist_y(gen);
			particles[i].theta = theta_f + dist_theta(gen);
		} else {	// yaw rate is zero
			particles[i].x = x + velocity * cos(theta) * delta_t + dist_x(gen);
			particles[i].y = y + velocity * sin(theta) * delta_t + dist_y(gen);
			particles[i].theta = theta + dist_theta(gen);
		}
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double k_std_xy = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
	double k_std_xx = 1 / (2 * std_landmark[0] * std_landmark[0]);
	double k_std_yy = 1 / (2 * std_landmark[1] * std_landmark[1]);
	double weight_sum = 0;

	for(int i = 0; i < num_particles; ++i) {
		double xp = particles[i].x;
		double yp = particles[i].y;
		double theta = particles[i].theta;
		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;
		double prob = 1;

		// Transformation, data association, and weight update for each particle and each observation
		for(int n = 0; n < observations.size(); ++n) {
			// Transform landmark observations from the car coordinate to the map coordinate
			LandmarkObs obs_t;
			double xc = observations[n].x;
			double yc = observations[n].y;

			obs_t.x = cos(theta) * xc - sin(theta) * yc + xp;
			obs_t.y = sin(theta) * xc + cos(theta) * yc + yp;

			// Associate transformed landmark observations to landmark IDs
			double dist_min = 500;
			for(int m = 0; m < map_landmarks.landmark_list.size(); ++m){
				double xm = map_landmarks.landmark_list[m].x_f;
				double ym = map_landmarks.landmark_list[m].y_f;
				double dist_obs = dist(obs_t.x, obs_t.y, xm, ym);

				if(dist_min > dist_obs) {
					dist_min = dist_obs;
					obs_t.id = map_landmarks.landmark_list[m].id_i;
				}
			}

			// Save associations
			associations.push_back(obs_t.id);
			sense_x.push_back(obs_t.x);
			sense_y.push_back(obs_t.y);

			// Update weight
			int index = associations[n] - 1; // landmark IDs start from 1
			double mu_x = map_landmarks.landmark_list[index].x_f;
			double mu_y = map_landmarks.landmark_list[index].y_f;
			double dx = sense_x[n] - mu_x;
			double dy = sense_y[n] - mu_y;

			prob *= k_std_xy * exp(-k_std_xx * dx * dx - k_std_yy * dy * dy);
		}

		// Add associations to particle when a landmark is found
		if(observations.size()) {
			particles[i].associations = associations;
			particles[i].sense_x = sense_x;
			particles[i].sense_y = sense_y;
		}

		particles[i].weight = prob;
		weight_sum += prob;
	}

	// Normalize weight
	for(int i = 0; i < num_particles; ++i) {
		particles[i].weight *= 1.0/weight_sum;
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;

	// Create discrete distribution with weights
	discrete_distribution<> d(weights.begin(), weights.end());

	// Resample
	std::vector<Particle> particles_new;
	for(int i = 0; i < num_particles; ++i) {
		particles_new.push_back(particles[d(gen)]);
	}

	particles = particles_new;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
