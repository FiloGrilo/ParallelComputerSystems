#include <stdlib.h>
#include <time.h>

#define ITERATIONS 10000
#define ARRAY_SIZE 32768

int main(int argc, char** argv) {
  srand(time(NULL));

  int data[ARRAY_SIZE];
  
  for(int i=0; i<ARRAY_SIZE; i++) {
    data[i] = rand() % 256;
  }

  for(int i=0; i<ITERATIONS; i++) {
    int sum = 0;
    for(int i=0; i<ARRAY_SIZE; i++) {
      if(data[i] >= 128) {
        sum += data[i];
      }
    }
  }

  return 0;
}