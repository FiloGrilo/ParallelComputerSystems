#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <stdlib.h>

#define INTERVALS 10000000

double a[INTERVALS], b[INTERVALS];

int main(int argc, char **argv)
{
  double *to = b;
  double *from = a;
  int    time_steps = 100;

  char* env_var = getenv("OMP_NUM_THREADS");
  int omp_num_threads;
  sscanf(env_var, "%d", &omp_num_threads);
  /* Set up initial and boundary conditions. */
  from[0] = 1.0;
  from[INTERVALS - 1] = 0.0;
  to[0] = from[0];
  to[INTERVALS - 1] = from[INTERVALS - 1];
  
  #pragma omp parallel
  {
   #pragma omp for nowait schedule(static, (INTERVALS/2)/omp_num_threads)
   for(long i = 1; i < INTERVALS; i++)
    from[i] = 0.0;
    from[i+1] = 0.0;
   // from[i+2] = 0.0;
   // from[i+3] = 0.0;
  }
  printf("Number of intervals: %ld. Number of time steps: %d\n", INTERVALS, time_steps);

  /* Apply stencil iteratively. */
  while(time_steps-- > 0)
  {
   #pragma omp parallel
   {
    #pragma omp for schedule(static, (INTERVALS/omp_num_threads))
    for(long i = 1; i < (INTERVALS - 1); i++)
     to[i] = from[i] + 0.1*(from[i - 1] - 2*from[i] + from[i + 1]);
   }
    { 
     double* tmp = from;
     from = to;
     to = tmp;
    }
  }
  for(long i = 2; i < 30; i += 2)
   printf("Interval %ld: %f\n", i, to[i]);
  return 0;
}                
