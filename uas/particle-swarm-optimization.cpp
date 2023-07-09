#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// Define the fitness function to be optimized
double fitnessFunction(const std::vector<double>& position) {
    // Example fitness function: Sphere function
    double sum = 0.0;
    for (double x : position) {
        sum += x * x;
    }
    return sum;
}

class Particle {
public:
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> bestPosition;
    double bestFitness;

    Particle(int numDimensions) {
        position.resize(numDimensions);
        velocity.resize(numDimensions);
        bestPosition.resize(numDimensions);
        bestFitness = std::numeric_limits<double>::max();
    }
};

class PSO {
private:
    int numParticles;
    int numDimensions;
    std::vector<Particle> particles;
    std::vector<double> globalBestPosition;
    double globalBestFitness;
    double inertiaWeight;
    double cognitiveWeight;
    double socialWeight;
    double velocityLimit;
    int numIterations;

public:
    PSO(int numParticles, int numDimensions, double inertiaWeight, double cognitiveWeight, double socialWeight,
        double velocityLimit, int numIterations)
        : numParticles(numParticles), numDimensions(numDimensions), inertiaWeight(inertiaWeight),
          cognitiveWeight(cognitiveWeight), socialWeight(socialWeight), velocityLimit(velocityLimit),
          numIterations(numIterations) {
        particles.resize(numParticles, Particle(numDimensions));
        globalBestPosition.resize(numDimensions);
        globalBestFitness = std::numeric_limits<double>::max();
    }

    void initializeParticles() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < numParticles; ++i) {
            for (int j = 0; j < numDimensions; ++j) {
                particles[i].position[j] = dist(gen);
                particles[i].velocity[j] = dist(gen);
                particles[i].bestPosition[j] = particles[i].position[j];
            }

            double fitness = fitnessFunction(particles[i].position);
            particles[i].bestFitness = fitness;

            if (fitness < globalBestFitness) {
                globalBestFitness = fitness;
                globalBestPosition = particles[i].position;
            }
        }
    }

    void updateParticles() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (int i = 0; i < numParticles; ++i) {
            for (int j = 0; j < numDimensions; ++j) {
                // Update velocity
                double r1 = dist(gen);
                double r2 = dist(gen);
                particles[i].velocity[j] =
                    inertiaWeight * particles[i].velocity[j] +
                    cognitiveWeight * r1 * (particles[i].bestPosition[j] - particles[i].position[j]) +
                    socialWeight * r2 * (globalBestPosition[j] - particles[i].position[j]);

                // Apply velocity limits
                particles[i].velocity[j] = std::min(std::max(particles[i].velocity[j], -velocityLimit), velocityLimit);

                // Update position
                particles[i].position[j] += particles[i].velocity[j];

                // Clamp position within a range if necessary
                // Example: particles[i].position[j] = std::min(std::max(particles[i].position[j], minValue), maxValue);
            }

            double fitness = fitnessFunction(particles[i].position);

            if (fitness < particles[i].bestFitness) {
                particles[i].bestFitness = fitness;
                particles[i].bestPosition = particles[i].position;
            }

            if (fitness < globalBestFitness) {
                globalBestFitness = fitness;
                globalBestPosition = particles[i].position;
            }
        }
    }

    std::vector<double> optimize() {
        initializeParticles();

        for (int iteration = 0; iteration < numIterations; ++iteration) {
            updateParticles();
        }

        return globalBestPosition;
    }
};

int main() {
    // Example usage
    int numParticles = 30;
    int numDimensions = 2;
    double inertiaWeight = 0.5;
    double cognitiveWeight = 1.0;
    double socialWeight = 1.0;
    double velocityLimit = 0.1;
    int numIterations = 100;

    PSO pso(numParticles, numDimensions, inertiaWeight, cognitiveWeight, socialWeight, velocityLimit, numIterations);

    std::vector<double> bestPosition = pso.optimize();

    std::cout << "Optimized solution: ";
    for (double x : bestPosition) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    return 0;
}
