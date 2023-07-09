#include <iostream>
using namespace std;

// Define the genetic algorithm parameters.
const int populationSize = 100;
const int maxGenerations = 100;
const double crossoverProbability = 0.8;
const double mutationProbability = 0.01;

// Function to evaluate the fitness of an individual.
int evaluateFitness(int individual[]) {
  // The fitness of an individual is the number of 1s in the individual.
  int fitness = 0;
  for (int i = 0; i < 10; i++) {
    if (individual[i] == 1) {
      fitness++;
    }
  }

  return fitness;
}

// Function to select a parent from the population.
int selectParent(double fitnesses[]) {
  // The roulette wheel selection algorithm is used to select a parent from the population.
  double totalFitness = 0;
  for (int i = 0; i < populationSize; i++) {
    totalFitness += fitnesses[i];
  }

  double randomNumber = rand() / (RAND_MAX + 1.0);
  int parentIndex = 0;
  double cumulativeFitness = 0;
  while (cumulativeFitness < randomNumber) {
    cumulativeFitness += fitnesses[parentIndex];
    parentIndex++;
  }

  return parentIndex;
}

// Function to perform the crossover operation on two parents to create offspring.
void crossover(int parent1[], int parent2[], int offspring[]) {
  // The single-point crossover algorithm is used to create offspring.
  int crossoverPoint = rand() % 10;
  for (int i = 0; i < crossoverPoint; i++) {
    offspring[i] = parent1[i];
  }
  for (int i = crossoverPoint; i < 10; i++) {
    offspring[i] = parent2[i];
  }
}

int main() {
  // Initialize the population.
  int population[populationSize][10];
  for (int i = 0; i < populationSize; i++) {
    for (int j = 0; j < 10; j++) {
      population[i][j] = rand() % 2;
    }
  }

  // Evaluate the fitness of each individual in the population.
  double fitnesses[populationSize];
  for (int i = 0; i < populationSize; i++) {
    fitnesses[i] = evaluateFitness(population[i]);
  }

  // Repeat the following steps until a termination condition is met:
  int generation = 0;
  while (generation < maxGenerations) {
    // Select parents from the population for reproduction based on their fitness.
    int parents[populationSize / 2][2];
    for (int i = 0; i < populationSize / 2; i++) {
      int parent1Index = selectParent(fitnesses);
      int parent2Index = selectParent(fitnesses);
      parents[i][0] = parent1Index; // Store the parent indices, not the individuals themselves
      parents[i][1] = parent2Index;
    }

    // Apply genetic operators (crossover and mutation) to create offspring from the selected parents.
    int offspring[populationSize / 2][10];
    for (int i = 0; i < populationSize / 2; i++) {
      crossover(population[parents[i][0]], population[parents[i][1]], offspring[i]);
    }

    // Evaluate the fitness of the offspring.
    double offspringFitnesses[populationSize / 2];
    for (int i = 0; i < populationSize / 2; i++) {
      offspringFitnesses[i] = evaluateFitness(offspring[i]);
    }

    // Select individuals from the population (parents and offspring) for the next generation based on their fitness.
    for (int i = 0; i < populationSize; i++) {
      int bestIndex = 0;
      double bestFitness = fitnesses[0];
      for (int j = 1; j < populationSize / 2; j++) {
        if (offspringFitnesses[j] > bestFitness) {
          bestIndex = j;
          bestFitness = offspringFitnesses[j];
        }
      }
      for (int k = 0; k < 10; k++) {
        population[i][k] = offspring[bestIndex][k];
      }
      fitnesses[i] = offspringFitnesses[bestIndex];
    }

    generation++;
  }

  // Return the best individual as the result.
  int bestIndividualIndex = 0;
  double bestFitness = fitnesses[0];
  for (int i = 1; i < populationSize; i++) {
    if (fitnesses[i] > bestFitness) {
      bestIndividualIndex = i;
      bestFitness = fitnesses[i];
    }
  }

  // Print the best individual.
  cout << "Best Individual: ";
  for (int i = 0; i < 10; i++) {
    cout << population[bestIndividualIndex][i] << " ";
  }
  cout << endl;

  return 0;
}
