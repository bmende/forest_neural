#include "stats.h"


using namespace std;


Stats::Stats() {

  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {

      statistics[i][j] = 0;

    }
  }
}



void Stats::print() {
    cout << "-----------------------------\nStats:\n-----------------------------" << endl;

  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      
      std::cout << statistics[i][j] << "\t";
    }
    std::cout << std::endl;
  }
    
    cout << "-----------------------------" << endl;
    
    double totalCorrect = 0;
    for (int i = 0; i < 7; i++)
    {
        double typeCorrect = statistics[i][i];
        double typeIncorrect = 0;
        totalCorrect += typeCorrect;
        
        cout << i+1 << ":" << endl;
        for (int j = 0; j < 7; j++)
        {
            if (j == i)
                continue;
            
            typeIncorrect += statistics[j][i];
        }
        double typeAccuracy = typeCorrect + typeIncorrect > 0 ? 100*typeCorrect/(typeCorrect + typeIncorrect) : 100;
        cout << typeAccuracy << "%  accuracy (" << typeCorrect << "/" << typeCorrect + typeIncorrect << ")" << endl;
        cout << "-----------------------------" << endl;
    }
    
    double totalAccuracy = 100*totalCorrect / (double)NUM_TEST;
    cout << "total accuracy: " << totalAccuracy << "% (" << totalCorrect << "/" << NUM_TEST << ")\n" << endl;
}

void Stats::writeToFile(string path)
{
    // ensure we are writing to a .csv file
    if (!path.substr(path.size()-3).compare(".csv"))
    {
        path = path.append(".csv");
    }
    
    ofstream outputStream;
    
    // check if file exists
    ifstream existingFile(path.c_str());
    
    if (existingFile.good())
    {
        // if it does, append to it
        outputStream.open(path.c_str(), ios::app);
    }
    else
    {
        // if it doesn't just create it and start writing
        outputStream.open(path.c_str());
        
        // column headers
        outputStream << "EPOCH, # HIDDEN, MSE, 1 ACC, 2 ACC, 3 ACC, 4 ACC, 5 ACC, 6 ACC, 7 ACC, AVG ACC, TOTAL ACC" << endl;
    }
    
    outputStream.precision(5);
    outputStream << "?, ?, ?, ";
    
    double totalCorrect = 0;
    for (int i = 0; i < 7; i++)
    {
        double typeCorrect = statistics[i][i];
        double typeIncorrect = 0;
        totalCorrect += typeCorrect;
        
        for (int j = 0; j < 7; j++)
        {
            if (j == i)
                continue;
            
            typeIncorrect += statistics[j][i];
        }
        double typeAccuracy = typeCorrect + typeIncorrect > 0 ? 100*typeCorrect/(typeCorrect + typeIncorrect) : 100;
        outputStream << typeAccuracy << ", ";
    }
    
    double totalAccuracy = 100*totalCorrect / (double)NUM_TEST;
    outputStream << totalAccuracy << endl;
    
    outputStream.close();
}

int main() {

  Data *d = new Data();
  Stats *s = new Stats();

  cout << "reading data\n";
  d->readData();
  cout << "data read\nnow making net\n";
  NeuralNetwork *net = new NeuralNetwork(NUM_ATTRIBUTES, 10);
  double alpha = 0.1; //this is the learning rate
  net->init(alpha);
  cout << "net initialized with random weights\n";
  
  const vector<int> train = d->getTraining();
  const vector<int> val = d->getValidation();
  const vector<int> test = d->getTest();

  //this is trainingish
  double meanErr = 10000, prevMeanErr = meanErr;
  int epoch = 0;
  double tolerance = 0.1; // I think it should be one std dev, but I havent calculated this yet.
  while (meanErr > 0.8) {
    for (int i = 0; i < NUM_TRAIN; i++) {
      const vector<double>& lineIn = d->getData()[train[i]];
      net->backProp(lineIn, d->getCover(train[i]));
    }
    
    prevMeanErr = meanErr;
    meanErr = 0;
    for (int i = 0; i < NUM_VALIDATE; i++) {
      const vector<double>& lineIn = d->getData()[val[i]];
      vector<double> thing = net->forwardProp(lineIn);
      vector<double> error = net->findErrorVector(thing, d->getCover(val[i]));
      double tempError = 0;
      for (int j = 0; j < error.size(); j++){
	tempError += pow(error[j], 2);
      }
      // tempError /= error.size();
      meanErr += tempError;
    }  
    meanErr /= NUM_VALIDATE;
    cout << epoch << " " << meanErr <<endl;
    epoch++;
  }

//  string blah ("weights55-10.dat");
//  net->readWeightsToFile(blah);
    
//  net->readWeightsFromFile("weights.dat");

  //now we test
  double correct = 0;
  for (int i = 0; i < NUM_TEST; i++) {
    const vector<double>& lineIn = d->getData()[test[i]];
    vector<double> result = net->forwardProp(lineIn);
    //now to compare
    int maxIndex = 0;
    double maxVal = -1;
    for (int j = 0; j < 7; j++) {
      if (result[j] > maxVal){
        maxVal = result[j];
        maxIndex = j;
      }
    }
    int actualTree = d->getCover(test[i]);
    s->statistics[maxIndex][actualTree - 1]++;
    if (maxIndex == actualTree-1) correct++;
  }

  s->print();
  s->writeToFile("stats.csv");
}
