#include "ann.h"


using namespace std;

NeuralNetwork::NeuralNetwork(int numInputs, int numHidden) {


  hidden.inputs = numInputs; hidden.numNodes = numHidden;
  hidden.weights = new double*[hidden.numNodes];
  for (int i = 0; i < hidden.numNodes; i++) 
    hidden.weights[i] = new double[hidden.inputs];
  hidden.outputs = new double[hidden.numNodes];
  hidden.summed = new double[hidden.numNodes];
  hidden.primed = new double[hidden.numNodes];

  output.inputs = hidden.numNodes; output.numNodes = 7;
  output.weights = new double*[output.numNodes];
  for (int i = 0; i < output.numNodes; i++)
    output.weights[i] = new double[output.inputs];
  output.outputs = new double[output.numNodes];
  output.summed = new double[output.numNodes];
  output.primed = new double[output.numNodes];

  
  inputs = new double[numInputs];

}

void NeuralNetwork::init(double alpha) {

  learnRate = alpha;

  //initializing weights to 0.5 each. TODO: make random initial weights
  
  for (int i = 0; i < hidden.numNodes; i++) {
    for (int j = 0; j < hidden.inputs; j++) {
      hidden.weights[i][j] = (double)std::rand() / (double)RAND_MAX;
      hidden.weights[i][j] /= (double)hidden.numNodes;
    }
    hidden.outputs[i] = 0;
    hidden.summed[i] = 0;
    hidden.primed[i] = 0;
  }

  for (int i = 0; i < output.numNodes; i++) {
    for (int j = 0; j < output.inputs; j++) {
      output.weights[i][j] = (double)std::rand() / (double)RAND_MAX;
      output.weights[i][j] /= (double)output.numNodes;
    }
    output.outputs[i] = 0;
    hidden.summed[i] = 0;
    hidden.primed[i] = 0;
  }
}

double NeuralNetwork::sigma(double weightedSum) {
  //this function is sigma(x) = 1/(1+e^(-x))

  weightedSum *= -1;
  double answer = 1 + exp(weightedSum);
  answer = 1/answer;

  return answer;
}

double NeuralNetwork::sigmaPrime(double weightedSum) {
  double sig = sigma(weightedSum);

  return sig*(1-sig);
}

vector<double> NeuralNetwork::forwardProp(const vector<double>& lineIn) {
  
  if (lineIn.size() != 54) {
    cerr << "line in not correct size\n";
    exit(1);
  } 
  //putting data in input nodes
  for (int i = 0; i < lineIn.size(); i++) {
    inputs[i] = lineIn[i];
  }

  //calculating hidden node outputs.
  for (int node = 0; node < hidden.numNodes; node++) {
    double sum = 0;
    for (int in = 0; in < hidden.inputs; in++) {
      sum += (inputs[in] * hidden.weights[node][in]);
    }
    hidden.outputs[node] = sigma(sum);
    hidden.summed[node] = sum;
    hidden.primed[node] = sigmaPrime(sum);
  }
  
  //now for output node calculations
  for (int node = 0; node < output.numNodes; node++) {
    double sum = 0;
    for (int in = 0; in < output.inputs; in++) {
      sum += (hidden.outputs[in] * output.weights[node][in]);
    }
    output.outputs[node] = sigma(sum);
    output.summed[node] = sum;
    output.primed[node] = sigmaPrime(sum);
  }
  vector<double> answer(output.numNodes, 0);
  for (int i = 0; i < answer.size(); i++)
    answer[i] = output.outputs[i];
  return answer;
}

vector<double> NeuralNetwork::findErrorVector(vector<double> output, int trainer) {

  vector<double> error(output.size(), 0);
  for (int i = 0; i < error.size(); i++) {
    if ((trainer-1) == i)
      error[i] = 1.0 - output[i];
    else
      error[i] = 0.0 - output[i];
  }
  return error;
}


void NeuralNetwork::backProp(const vector<double>& lineIn, int trainer) {

  vector<double> out = forwardProp(lineIn);
  vector<double> err = findErrorVector(out, trainer);
  
  //so now we adjust the error de = err * sigma'
  for (int i = 0; i < err.size(); i++) {
    err[i] *= output.primed[i];
  }

  //now we propagate error;
  vector<double> delta_j(hidden.numNodes, 0);
  for (int j = 0; j < delta_j.size(); j++) {
    for (int i = 0; i < err.size(); i++) {
      delta_j[j] += output.weights[i][j]*err[i];
    }
    delta_j[j] *= hidden.primed[j];
  }

  //now to adjust hidden -> output weights
  //W_ji <- W_ji + learn*out_j*err_i
  for (int outNode = 0; outNode < output.numNodes; outNode++) {
    for (int hidNode = 0; hidNode < hidden.numNodes; hidNode++) {
      double old_weight = output.weights[outNode][hidNode];
      double new_weight = learnRate * hidden.outputs[hidNode] * err[outNode];
      new_weight += old_weight;
      output.weights[outNode][hidNode] = new_weight;
    }
  }

  //now we adjust input -> hidden weights
  //this is much the same as before
  for (int hidNode = 0; hidNode < hidden.numNodes; hidNode++) {
    for (int inNode = 0; inNode < hidden.inputs; inNode++) {
      double old_weight = hidden.weights[hidNode][inNode];
      double new_weight = learnRate * lineIn[inNode] * delta_j[hidNode];
      new_weight += old_weight;
      hidden.weights[hidNode][inNode] = new_weight;
    }
  }
}

int main() {

  Data *d = new Data();

  cout << "reading data\n";
  d->readData();
  cout << "data read\nnow making net\n";
  NeuralNetwork *net = new NeuralNetwork(NUM_ATTRIBUTES, 120);
  double alpha = 0.05; //this is the learning rate
  net->init(alpha);
  cout << "net initialized with random weights\n";
  

  //this is trainingish
  double correct = 0;
  for (int number = 0; number < 20; number++) {
    for (int i = 0; i < 10000; i++) {
      const vector<double>& lineIn = d->getData()[i];
      net->backProp(lineIn, d->getCover(i));
    }
    for (int i = NUM_EXAMPLES - 100000; i < NUM_EXAMPLES; i++) {
      const vector<double>& lineIn = d->getData()[i];
      vector<double> thing = net->forwardProp(lineIn);
      int max = 0;
      for (int j = 0; j < thing.size(); j++){
	if (j != 0 && (thing[j] > thing[max]))
	  max = j;
      }
      
      if (max == d->getCover(i)) correct++;
    }  
    cout << "we got " << 100*correct/(double)(NUM_EXAMPLES-100000) << "% correct\n";
    
  }
}
