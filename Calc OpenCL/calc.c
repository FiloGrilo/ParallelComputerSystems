#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"
#include <time.h>

//pick up device type from compiler command line or from
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

extern double wtime();       // returns time since some fixed past point (wtime.c)
extern int output_device_info(cl_device_id );

#define INTERVALS 10000000

//------------------------------------------------------------------------------
//
// kernel:  calc


const char *KernelSource = "\n" \
"__kernel void fillArray(                                              \n" \
"   __global double* from,                                             \n" \
"   const double intervals)                                         \n" \
"{                                                                     \n" \
"   int i = get_global_id(0)+1;                                        \n" \
"   if(i < intervals-1)                                                  \n" \
"     from[i] = 0.0;                                                   \n" \
"}                                                                     \n" \
"                                                                      \n" \
"__kernel void calc(                                           \n" \
"   __global double* to,                                  \n" \
"   __global double* from,                                             \n" \
"   const double intervals,			    	             	\n" \
"   int time_steps)                                          \n" \
"{                                                                     \n" \
"   int i = get_global_id(0)+1;                                           \n" \
"   while(time_steps-- > 0)                                                \n" \
"   {                                                                      \n" \
"     if( i < (intervals - 1))                                            \n" \
"      to[i] = from[i] + 0.1*(from[i - 1] - 2*from[i] + from[i + 1]);      \n" \
"     __global double* tmp = from;                                               \n" \
"     from = to;                                                 \n" \
"     to = tmp;                                                 \n" \
"                                                        \n" \
"   }                                                      \n" \
"}                                                         \n" \
"                                                         \n" \
"                                                         \n" \
"                                                         \n" \
"                                                         \n" \
"                                                         \n";

//------------------------------------------------------------------------------


int main(int argc, char **argv)
{
  int    time_steps = 100;
  double intervals = INTERVALS;
  
  /* Set up initial and boundary conditions. */
  double*       to = (double*) calloc(INTERVALS, sizeof(double));       // to vector
  double*       from = (double*) calloc(INTERVALS, sizeof(double));       // from vector
  from[0] = 1.0;
  from[INTERVALS - 1] = 0.0;
  to[0] = from[0];
  to[INTERVALS - 1] = from[INTERVALS - 1];

  /* Initialize OpenCL variables */
  int err;               // error code returned from OpenCL calls

  size_t global;                  // global domain size

  cl_device_id     device_id;     // compute device id
  cl_context       context;       // compute context
  cl_command_queue commands;      // compute command queue
  cl_program       program;       // compute program
  cl_kernel        ko_fill, ko_calc;       // compute kernel

  cl_mem d_to;                     // device memory used for the input  to vector
  cl_mem d_from;                     // device memory used for the input  from vector
  
  printf("Number of intervals: %ld. Number of time steps: %d\n", INTERVALS, time_steps);

  // Set up platform and GPU device

  cl_uint numPlatforms;

  // Find number of platforms
  err = clGetPlatformIDs(0, NULL, &numPlatforms);
  checkError(err, "Finding platforms");
  if (numPlatforms == 0)
  {
      printf("Found 0 platforms!\n");
      return EXIT_FAILURE;
  }

  // Get all platforms
  cl_platform_id Platform[numPlatforms];
  err = clGetPlatformIDs(numPlatforms, Platform, NULL);
  checkError(err, "Getting platforms");

  // Secure a device
  for (int i = 0; i < numPlatforms; i++)
  {
    err = clGetDeviceIDs(Platform[i], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err == CL_SUCCESS)
    {
       break;
    }
  }

  if (device_id == NULL)
    checkError(err, "Finding a device");

  err = output_device_info(device_id);
  checkError(err, "Printing device output");

  // Create a compute context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  checkError(err, "Creating context");

  // Create a command queue
  commands = clCreateCommandQueue(context, device_id, 0, &err);
  checkError(err, "Creating command queue");

  // Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
  checkError(err, "Creating program");

  // Build the program
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    size_t len;
    char buffer[2048];
        
    printf("Error: Failed to build program executable!\n%s\n", err_code(err));
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    return EXIT_FAILURE;
  }

  // Create the compute kernel from the program
  ko_fill = clCreateKernel(program, "fillArray", &err);
  checkError(err, "Creating kernel for fillArray");

  ko_calc = clCreateKernel(program, "calc", &err);
  checkError(err, "Creating kernel for calc");

  // Create the to and from arrays in device memory
  d_to  = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,  sizeof(double) * intervals, to, &err);
  checkError(err, "Creating buffer d_to");

  d_from  = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,  sizeof(double) * intervals, from, &err);
  checkError(err, "Creating buffer d_from");

  // Set the arguments to fill kernel
  err  = clSetKernelArg(ko_fill, 0, sizeof(cl_mem), &d_from);
  err |= clSetKernelArg(ko_fill, 1, sizeof(double), &intervals);
  checkError(err, "Setting fill kernel arguments");

  double rtime = wtime();
  
  // Enqueue fill kernel command
  global = intervals;
  err = clEnqueueNDRangeKernel(commands, ko_fill, 1, NULL, &global, NULL, 0, NULL, NULL);
  checkError(err, "Enqueueing fill kernel");

  // Set the arguments to calc kernel
  err = clSetKernelArg(ko_calc, 0, sizeof(cl_mem), &d_to);
  err |= clSetKernelArg(ko_calc, 1, sizeof(cl_mem), &d_from);
  err |= clSetKernelArg(ko_calc, 2, sizeof(double), &intervals);
  err |= clSetKernelArg(ko_calc, 3, sizeof(int), &time_steps);
  checkError(err, "Setting calc kernel arguments");

  // Enqueue calc kernel command
  err = clEnqueueNDRangeKernel(commands, ko_calc, 1, NULL, &global, NULL, 0, NULL, NULL);
  checkError(err, "Enqueueing calc kernel");

  // Wait for the commands to complete before stopping the timer
  err = clFinish(commands);
  checkError(err, "Waiting for kernel to finish");

  rtime = wtime() - rtime;
  printf("\nThe kernel ran in %lf seconds\n",rtime);
  
  // Read back the results from the compute device
  err = clEnqueueReadBuffer( commands, d_to, CL_TRUE, 0, sizeof(double) * intervals, to, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array!\n%s\n", err_code(err));
    exit(1);
  }
  
  for(long i = 2; i < 30; i += 2)
   printf("Interval %ld: %f\n", i, to[i]);

  // cleanup then shutdown
  clReleaseMemObject(d_to);
  clReleaseMemObject(d_from);
  clReleaseProgram(program);
  clReleaseKernel(ko_fill);
  clReleaseKernel(ko_calc);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  free(to);
  free(from);
  
  return 0;
}                
