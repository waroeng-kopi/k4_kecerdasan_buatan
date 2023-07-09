#include <iostream>
#include <vector>

using namespace std;

// Define the linguistic variables and their membership functions.
enum LinguisticVariable {
  Low,
  Medium,
  High
};

struct MembershipFunction {
  double a;
  double b;
};

vector<MembershipFunction> lowMembershipFunctions = {
  {0.0, 0.1},
  {0.1, 0.5},
  {0.5, 1.0}
};

vector<MembershipFunction> mediumMembershipFunctions = {
  {0.2, 0.4},
  {0.4, 0.7},
  {0.7, 1.0}
};

vector<MembershipFunction> highMembershipFunctions = {
  {0.3, 0.6},
  {0.6, 1.0}
};

// Initialize the fuzzy rule base with appropriate fuzzy rules.
const vector<vector<int>> fuzzyRules = {
  {Low, High},
  {Medium, Medium},
  {High, Low}
};

// Repeat the following steps for each input value:
double fuzzifyInput(double input) {
  double lowMembership = 0.0;
  double mediumMembership = 0.0;
  double highMembership = 0.0;

  for (const auto& membershipFunction : lowMembershipFunctions) {
    if (input >= membershipFunction.a && input <= membershipFunction.b) {
      lowMembership = 1.0 - (input - membershipFunction.a) / (membershipFunction.b - membershipFunction.a);
      break;
    }
  }

  for (const auto& membershipFunction : mediumMembershipFunctions) {
    if (input >= membershipFunction.a && input <= membershipFunction.b) {
      mediumMembership = 1.0 - (input - membershipFunction.a) / (membershipFunction.b - membershipFunction.a);
      break;
    }
  }

  for (const auto& membershipFunction : highMembershipFunctions) {
    if (input >= membershipFunction.a && input <= membershipFunction.b) {
      highMembership = 1.0 - (input - membershipFunction.a) / (membershipFunction.b - membershipFunction.a);
      break;
    }
  }

  return lowMembership + mediumMembership + highMembership;
}

double applyRule(double lowMembership, double mediumMembership, double highMembership, int rule) {
  switch (rule) {
  case Low:
    return lowMembership;
  case Medium:
    return mediumMembership;
  case High:
    return highMembership;
  default:
    return 0.0;
  }
}

double aggregateRules(double lowActivation, double mediumActivation, double highActivation) {
  return lowActivation + mediumActivation + highActivation;
}

double defuzzify(double fuzzyOutput) {
  return fuzzyOutput * 0.5 + 0.5;
}

int main() {
  double input = 0.9;
  double lowMembership = fuzzifyInput(input);
  double mediumMembership = fuzzifyInput(input);
  double highMembership = fuzzifyInput(input);

  double lowActivation = applyRule(lowMembership, mediumMembership, highMembership, 0);
  double mediumActivation = applyRule(lowMembership, mediumMembership, highMembership, 1);
  double highActivation = applyRule(lowMembership, mediumMembership, highMembership, 2);

  double fuzzyOutput = aggregateRules(lowActivation, mediumActivation, highActivation);
  double crispOutput = defuzzify(fuzzyOutput);

  cout << "Input: " << input << endl;
  cout << "Low membership: " << lowMembership << endl;
  cout << "Medium membership: " << mediumMembership << endl;
  cout << "High membership: " << highMembership << endl;
  cout << "Low activation: " << lowActivation << endl;
  cout << "Medium activation: " << mediumActivation << endl;
  cout << "High activation: " << highActivation << endl;
  cout << "Fuzzy output: " << fuzzyOutput << endl;
  cout << "Crisp output: " << crispOutput << endl;

  return 0;
}