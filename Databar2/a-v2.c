#include <stdlib.h>
#include <time.h>

#define ITERATIONS 10000
#define ARRAY_SIZE 32768

int compare(const void* ap, const void* bp) {
  int a = *(int*)ap;
  int b = *(int*)bp;
  if(a < b) return -1;
  else if(a == b) return 0;
  else return 1;
}

int main(int argc, char** argv) {
  srand(time(NULL));

  int data[ARRAY_SIZE];
  
  for(int i=0; i<ARRAY_SIZE; i++) {
    data[i] = rand() % 256;
  }

  qsort(data, ARRAY_SIZE, sizeof(int), compare);

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