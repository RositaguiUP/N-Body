
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#define N 15000 // Number of particles
#define p 256  // Number of particles per tile
#define EPS2 1e-6f

__device__ float3 bodyBodyInteraction (float4 bi, float4 bj, float3 ai) {
    float3 r;
    // r_ij [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    // distSqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
    // invDistCube =1/distSqr^(3/2) [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f / sqrtf(distSixth);
    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;
    // a_i = a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

__device__ float3 tile_calculation (float4 myPosition, float3 accel) {
    int i;
    extern __shared__ float4 shPosition[];
    for (i = 0; i < blockDim.x; i++) {
        accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
    }
    return accel;
}

__global__ void calculate_forces (void* devX, void* devA) {
    extern __shared__ float4 shPosition[];
    float4* globalX = (float4*)devX;
    float4* globalA = (float4*)devA;
    float4 myPosition;
    int i, tile;
    float3 acc = { 0.0f, 0.0f, 0.0f };
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    myPosition = globalX[gtid];
    for (i = 0, tile = 0; i < N; i += p, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        shPosition[threadIdx.x] = globalX[idx];
        // __syncthreads();
        acc = tile_calculation(myPosition, acc);
        // __syncthreads();
    }
    // Save the result in global memory for the integration step.
    float4 acc4 = { acc.x, acc.y, acc.z, 0.0f };
    globalA[gtid] = acc4;
}

void displayMe(void) {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);

    float x = 1.0f;
    float y = 0.0f;
    glVertex2f(x, y);
    
    for (float i = 0; i < (2 * 3.1416); i += 0.001)
    {
        // let 200 is radius of circle and as,
        // circle is defined as x=r*cos(i) and y=r*sin(i)
        x = 20 * cos(i);
        y = 20 * sin(i);

        glVertex2f(x, y);
    }

    glEnd();
    glFlush();
}

float4 hostX[N]; // Host array for particle positions
void displayParticles() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);

    float x, y;

    for (int i = 0; i < N; i++) {
        x = hostX[i].x;
        y = hostX[i].y;
        //printf("x = %f, y = %f\n", x, y);
        glVertex2f(x, y);
        /*for (float j = 0; j < (2 * 3.1416); j += 0.001)
        {
            // let 200 is radius of circle and as,
            // circle is defined as x=r*cos(i) and y=r*sin(i)
            x = 20 * cos(j) + hostX[i].x;
            y = 20 * sin(j) + hostX[i].y;
            //glVertex2f(x, y);
        }*/
    }
    glEnd();
    glFlush();
}



int main(int argc, char** argv)
{
   
    float4 hostA[N]; // Host array for particle acceleration
    
    /*
    // Initialize particle positions on the host
    float scale = 1000000;

    
    // Earth
    hostX[0].x = 0.0f;           // x-coordinate of the Earth (in meters)
    hostX[0].y = 0.0f;           // y-coordinate of the Earth (in meters)
    hostX[0].z = 0.0f;           // z-coordinate of the Earth (in meters)
    hostX[0].w = 5.972e24f;      // Mass of the Earth (in kilograms)

    // Moon
    hostX[1].x = 384400000.0f / scale;   // x-coordinate of the Moon (in meters)
    hostX[1].y = 0.0f / scale;           // y-coordinate of the Moon (in meters)
    hostX[1].z = 0.0f / scale;           // z-coordinate of the Moon (in meters)
    hostX[1].w = 7.342e22f;      // Mass of the Moon (in kilograms)

    // Asteroid
    hostX[2].x = 100000000.0f / scale;   // x-coordinate of the asteroid (in meters)
    hostX[2].y = 200000000.0f / scale;   // y-coordinate of the asteroid (in meters)
    hostX[2].z = 50000000.0f / scale;    // z-coordinate of the asteroid (in meters)
    hostX[2].w = 1.0e10f;        // Mass of the asteroid (in kilograms)
    

    */

    // Initialize particle positions randomly on the host
    srand(time(NULL));
    for (int i = 0; i < 15000; i++) {
        hostX[i].x = ((float)rand() / 10000) * 2.0f - 1.0f + 150; // Random x-coordinate between -1 and 1
        hostX[i].y = ((float)rand() / 10000) * 2.0f - 1.0f + 150; // Random y-coordinate between -1 and 1
        hostX[i].z = ((float)rand() / 10000) * 2.0f - 1.0f; // Random z-coordinate between -1 and 1
        //hostX[i].w = 1.0f; // Set mass to 1

        // Generate a random mass within the desired range
        float minMass = 0.10f;  // Minimum mass
        float maxMass = 1.0f; // Maximum mass
        float randomMass = minMass + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (maxMass - minMass)));

        // Assign the random mass to the particle
        hostX[i].w = randomMass;
    }

    for (int i = 2000; i < 4000; i++) {
        hostX[i].x = ((float)rand() / 10000) * 2.0f - 1.0f + 150; // Random x-coordinate between -1 and 1
        hostX[i].y = ((float)rand() / 10000) * 2.0f - 1.0f - 150; // Random y-coordinate between -1 and 1
        hostX[i].z = ((float)rand() / 10000) * 2.0f - 1.0f; // Random z-coordinate between -1 and 1
        //hostX[i].w = 1.0f; // Set mass to 1

        // Generate a random mass within the desired range
        float minMass = 0.10f;  // Minimum mass
        float maxMass = 1.0f; // Maximum mass
        float randomMass = minMass + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (maxMass - minMass)));

        // Assign the random mass to the particle
        hostX[i].w = randomMass;
    }

    for (int i = 4000; i < 6000; i++) {
        hostX[i].x = ((float)rand() / 10000) * 2.0f - 1.0f - 150; // Random x-coordinate between -1 and 1
        hostX[i].y = ((float)rand() / 10000) * 2.0f - 1.0f - 150; // Random y-coordinate between -1 and 1
        hostX[i].z = ((float)rand() / 10000) * 2.0f - 1.0f; // Random z-coordinate between -1 and 1
        //hostX[i].w = 1.0f; // Set mass to 1

        // Generate a random mass within the desired range
        float minMass = 0.10f;  // Minimum mass
        float maxMass = 1.0f; // Maximum mass
        float randomMass = minMass + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (maxMass - minMass)));

        // Assign the random mass to the particle
        hostX[i].w = randomMass;
    }

    for (int i = 6000; i < 8000; i++) {
        hostX[i].x = ((float)rand() / 10000) * 2.0f - 1.0f - 150; // Random x-coordinate between -1 and 1
        hostX[i].y = ((float)rand() / 10000) * 2.0f - 1.0f + 150; // Random y-coordinate between -1 and 1
        hostX[i].z = ((float)rand() / 10000) * 2.0f - 1.0f; // Random z-coordinate between -1 and 1
        //hostX[i].w = 1.0f; // Set mass to 1

        // Generate a random mass within the desired range
        float minMass = 0.10f;  // Minimum mass
        float maxMass = 1.0f; // Maximum mass
        float randomMass = minMass + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (maxMass - minMass)));

        // Assign the random mass to the particle
        hostX[i].w = randomMass;
    }
   
    /*
    2000
    10000
    0.1
    1.0


    2000
    10000
    1
    10

    8000
    1000
    1
    10

    8000
    1000
    0.1
    1.0

    */

    float4* devX; // Device array for particle positions
    float4* devA; // Device array for particle acceleration

    // Allocate memory on the GPU for particle positions and acceleration
    cudaMalloc((void**)&devX, N * sizeof(float4));
    cudaMalloc((void**)&devA, N * sizeof(float4));


    // Define grid and block sizes
    dim3 grid(N / p, 1, 1);
    dim3 block(p, 1, 1);


    // OpenGL initialization code...
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(1366, 768);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("N-Body Simulation");

    // Set up GLEW if necessary
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        printf("GLEW initialization failed!\n");
        return 1;
    }

    // Set up OpenGL viewport
    
    /*
    //glViewport(0, 0, 800, 600);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
    //gluOrtho2D(-780, 780, -420, 420);
    glOrtho(0, 800, 0, 600, -1, 1);
    

    /*
    glViewport(0, 0, 800, 600);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
    ///


    glutDisplayFunc(displayParticles);
    glutMainLoop();
    */


    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // setting window dimension in X- and Y- direction
    gluOrtho2D(-780, 780, -420, 420);

    glutDisplayFunc(displayParticles);
    //glutMainLoop();

    // Main simulation loop
    while (true) {
        // Run the CUDA kernel to calculate forces
        calculate_forces << <grid, block, p * sizeof(float4) >> > (devX, devA);

        // Transfer particle acceleration from device to host memory
        cudaMemcpy(hostA, devA, N * sizeof(float4), cudaMemcpyDeviceToHost);
        
        // Update particle positions based on the calculated accelerations
        for (int i = 0; i < N; i++) {
            hostX[i].x += hostA[i].x;
            hostX[i].y += hostA[i].y;
        }

        // Transfer particle positions from host to device memory
        cudaMemcpy(devX, hostX, N * sizeof(float4), cudaMemcpyHostToDevice);

        // Display particles using OpenGL
        //glutSwapBuffers();
        //glutPostRedisplay();
        //glutDisplayFunc(displayParticles);


        // Display particles using OpenGL
        displayParticles();

        // Flush OpenGL buffer
        glFlush();

        // Process OpenGL events
        glutMainLoopEvent();
    }

    // Print the calculated accelerations
    /*for (int i = 0; i < N; i++) {
        printf("Particle %d: Acceleration (x: %.2f, y: %.2f, z: %.2f)\n", i, hostA[i].x, hostA[i].y, hostA[i].z);
    }*/

    // Free device memory
    cudaFree(devX);
    cudaFree(devA);

    return 0;
}

