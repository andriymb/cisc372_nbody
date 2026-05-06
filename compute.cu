#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

__global__ void computeAccels(vector3 *dPos, vector3 *dAccel, double *dMass) {

    int entityOne = blockIdx.x * blockDim.x + threadIdx.x;
    int entityTwo = blockIdx.y * blockDim.y + threadIdx.y;
    if (entityOne >= NUMENTITIES || entityTwo >= NUMENTITIES) return;

    if (entityOne == entityTwo) {
        FILL_VECTOR(dAccel[entityOne * NUMENTITIES + entityTwo], 0, 0, 0);
        return;
    }

    vector3 distance;
    int axis;
    for (axis = 0; axis < 3; axis++) distance[axis] = dPos[entityOne][axis] - dPos[entityTwo][axis];

    double magnitudeSquared = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
	double magnitude = sqrt(magnitudeSquared);
	double accelMag = -1 * GRAV_CONSTANT * dMass[entityTwo] / magnitudeSquared;

	FILL_VECTOR(dAccel[entityOne * NUMENTITIES + entityTwo],
        accelMag * distance[0] / magnitude,
        accelMag * distance[1] / magnitude,
        accelMag * distance[2] / magnitude
    );
}

__global__ void updateEntities(vector3 *dPos, vector3 *dVel, vector3 *dAccel) {

    int entityOne = blockIdx.x * blockDim.x + threadIdx.x;
    if (entityOne >= NUMENTITIES) return;

    vector3 accelSum = {0, 0, 0};

    int entityTwo, axis;
    for (entityTwo = 0; entityTwo < NUMENTITIES; entityTwo++) {
        for (axis = 0; axis < 3; axis++)
            accelSum[axis] += dAccel[entityOne * NUMENTITIES + entityTwo][axis];
    }

    for (axis = 0; axis < 3; axis++) {
        dVel[entityOne][axis] += accelSum[axis] * INTERVAL;
        dPos[entityOne][axis] += dVel[entityOne][axis] * INTERVAL;
    }
}

void compute(vector3 *dPos, vector3 *dVel, vector3 *dAccel, double *dMass) {

    dim3 blockSize(16, 16);
    dim3 numBlocks((NUMENTITIES + blockSize.x - 1) / blockSize.x, (NUMENTITIES + blockSize.y - 1) / blockSize.y);
    computeAccels<<<numBlocks, blockSize>>>(dPos, dAccel, dMass);
    cudaDeviceSynchronize();

    dim3 blockSize2(256);
    dim3 numBlocks2((NUMENTITIES + blockSize2.x - 1) / blockSize2.x);
    updateEntities<<<numBlocks2, blockSize2>>>(dPos, dVel, dAccel);
    cudaDeviceSynchronize();
}