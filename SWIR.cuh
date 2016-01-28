//CUDA related include files
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "device_launch_parameters.h"


/*
this function runs on the CUDA device and ultimately takes the 8 channel values of the pixel at the lThreadId location of the image and 
turns those values into a vector in R^8. Then for each channel the mean of the library values are subtracted and the variances 
of the library values are used to ensure that each dimension independently has 0 mean and unit variance. We then take the inner product 
of this new vector and a vector assumbled in the same way from the library data. The scaler result of the inner product is then used as
a measure of similarity for the pixel vector and the library vector. 

the function reports back out the indexes and scores for the top three matches best, prior best, and prior prior best.

*/

__global__ void ProcessesSWIR(unsigned short * SWIRData0, unsigned short * SWIRData1, unsigned short * SWIRData2, unsigned short * SWIRData3,
	unsigned short * SWIRData4, unsigned short * SWIRData5, unsigned short * SWIRData6, unsigned short * SWIRData7,
	int imagewidth, int imageheight,
	float * ReflectanceLibrary, int ReflectanceStride, int NumReflectanceItems,
	int ReflectanceIndexStart, int ReflectanceIndexEnd,
	int * CurrentMatch, float * CurrentScore, 
	int * PMatch, float * PScore,
	int * P2Match, float * P2Score,
	int* LibIndices,int numLibIndices,
	float * dLibraryMoments,
	bool reprocess){

	//get global ID of the thread
	int lBlockId = blockIdx.x + blockIdx.y * gridDim.x;
	int lThreadId = lBlockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	
	bool update = false;

	//make sure that the thread is processing within the bounds of the image
	if (lThreadId < imagewidth*imageheight){
		//grab the 8 bands of the SWIR pixel
		float lSWIRVec[8];
		lSWIRVec[0] = SWIRData0[lThreadId];
		lSWIRVec[1] = SWIRData1[lThreadId];
		lSWIRVec[2] = SWIRData2[lThreadId];
		lSWIRVec[3] = SWIRData3[lThreadId];
		lSWIRVec[4] = SWIRData4[lThreadId];
		lSWIRVec[5] = SWIRData5[lThreadId];
		lSWIRVec[6] = SWIRData6[lThreadId];
		lSWIRVec[7] = SWIRData7[lThreadId];

		//normalize the SWIR vector
		//and convert it with statistical moments
		double lMagnitude = 0.0;
		for (int i = 0; i < 8; i++){
			lMagnitude += lSWIRVec[i] * lSWIRVec[i];
		}
		lMagnitude = sqrt(lMagnitude);
		lMagnitude = 1 / lMagnitude;

		for (int i = 0; i < 8; i++){
			lSWIRVec[i] *= lMagnitude;
			lSWIRVec[i] -= dLibraryMoments[i];
			lSWIRVec[i] /= dLibraryMoments[i + 8];
		}

		//renormalize for dot product
		lMagnitude = 0.0;
		for (int i = 0; i < 8; i++){
			lMagnitude += lSWIRVec[i] * lSWIRVec[i];
		}
		lMagnitude = sqrt(lMagnitude);
		lMagnitude = 1 / lMagnitude;
		for (int i = 0; i < 8; i++){
			lSWIRVec[i] *= lMagnitude;
		}

		//initialize the variables we will use for tracking the best data projection
		float lProjection = 0.0;
		float lProjMax = 0.0;
		int lBestIndex = -1;

		//if this is a reprocess job, then load the current best match values
		if (reprocess){
			lBestIndex = CurrentMatch[lThreadId];
			lProjMax = CurrentScore[lThreadId];
		}

		//cycle through a section of the reflectance library
		/*windows times out if you spend too long on a GPU task. to overcome this limitation,
		I had to examine only a subset of the reflectance library at a time
		*/
		
		if (lThreadId == 0){
			CurrentMatch[lThreadId] = lBestIndex; // just here for testing it doesn't need to be long term
		}
		//perform a dot product the SWIR vector has magnitude 1 and the Reflectance Library does too
		for (int i = ReflectanceIndexStart; i < ReflectanceIndexEnd; i++){
			for (int k = 0; k < numLibIndices; k++){
				lProjection = 0.0;

				if (LibIndices[k] == i){
				//if this index is one we should process
					lMagnitude = 0.0;
					for (int j = 0; j < 8; j++){
						lProjection += lSWIRVec[j] * ReflectanceLibrary[i*ReflectanceStride + j + 1];
						lMagnitude += ReflectanceLibrary[i*ReflectanceStride + j + 1] * ReflectanceLibrary[i*ReflectanceStride + j + 1]; //get magnitude of library vector
					}
					lMagnitude = sqrt(lMagnitude);
					lProjection /= lMagnitude; //make sure to normalize by magnitude of the reflectance library

					if (lProjMax < lProjection){
						//if projection is the new best match save the information
						update = true;
						lProjMax = lProjection;
						lBestIndex = i;
					}
				}
			}
		}
		//set the values
		
		if (!reprocess){
			//if this is the first time through then initialize variables to arbitrary values
			P2Match[lThreadId] = -2;
			PMatch[lThreadId] = -2;
			CurrentMatch[lThreadId] = -2;

			P2Score[lThreadId] = 0.0;
			PScore[lThreadId] = 0.0;
			CurrentScore[lThreadId] = 0.0;

		}

		if (update){
			//update values if a new best match is found
			P2Match[lThreadId] = PMatch[lThreadId];
			PMatch[lThreadId] = CurrentMatch[lThreadId];
			CurrentMatch[lThreadId] = lBestIndex;

			P2Score[lThreadId] = PScore[lThreadId];
			PScore[lThreadId] = CurrentScore[lThreadId];
			CurrentScore[lThreadId] = lProjMax;
		}
		
		
	}
	else{
		//threadID is not within the image so default the values
		P2Match[lThreadId] = PMatch[lThreadId];
		PMatch[lThreadId] = CurrentMatch[lThreadId];
		CurrentMatch[lThreadId] = -2;

		P2Score[lThreadId] = PScore[lThreadId];
		PScore[lThreadId] = CurrentScore[lThreadId];
		CurrentScore[lThreadId] = 0.0;
		
	}
}