#include "data.h"


using namespace std;


Data::Data() {

  //maybe something will go here
}

void Data::readData() {

  ifstream dataFile("covtype.data");
  
  string line;
  int outer = 0, inner = 0;
  double max[54], min[54];
  for (int i = 0; i <= 54; i++) {
    max[i] = 0;
    min[i] = 100000;
  }
  
  while (std::getline(dataFile, line)) {
    
    istringstream stream(line);
    char value[100];
    inner = 0;
   
    while (stream.getline(value, 100, ',')) {
      
      double number = std::atof(value);
      
      if (inner < 55) {
	dataArray[outer][inner] = number;
	if (number > max[inner])
	  max[inner] = number;
	else if (number < min[inner])
	  min[inner] = number;
      }
      else
	coverType[outer] = number;

      inner++;
    }
    outer++;
  }

  for (int i = 0; i < 581012; i++) {
    for (int j = 0; j < 54; j++) {
      dataArray[i][j] = (dataArray[i][j] - min[j])/(max[j] - min[j]);
      if (dataArray[i][j] > 1 || dataArray[i][j] < 0) {
	
      }
    }
  }
  cout << endl;

  cout << "Outer: " << outer << ", Inner: " << inner << endl;
  cout << "I opened the file!" << endl;

  dataFile.close();

}


int main() {

  Data *d = new Data();

  d->readData();

}
