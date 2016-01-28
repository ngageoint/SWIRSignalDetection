
//CUDA relevant functions
#include "SWIR.cuh"

//standard include files
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <map>
#include <string>

//includes specific to this program
#include "USGS_SpectralDataReader.h"
#include "SpectralData.h"
#include <DGDataReader.h>

//3rd party includes
#include "../freeglut/include/GL/freeglut.h" //http://files.transmissionzero.co.uk/software/development/GLUT/freeglut-MSVC-3.0.0-2.mp.zip

//namespaces
using namespace std;

//GLUT variables
double g_RegX = 0.0;
double g_RegY = 0.0;
double g_RegScale = 1.0;
double g_ScaleTranslate = 0;
int g_ZoomLevel = 0;
GLuint g_texture[2];
int g_Width = 512 * 2;
int g_Height = 512 * 2;
int g_imagewidth = 512;
int g_imageheight = 512;
int g_mouseX = 0;
int g_mouseY = 0;
bool g_leftdown = 0;


//data for processing
float * g_data;

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);									//reset the viewport
	glColor3f(1.0, 1.0, 1.0);										//set drawing color to white

	glPushMatrix();													//push down Modelview Matrix to get a copy that we can manipulate
	glTranslatef(-g_ScaleTranslate, -g_ScaleTranslate, 0.0);		//transform as appropriate
	glScalef(g_RegScale, g_RegScale, 1.0);
	glTranslatef(-g_RegX, -g_RegY, 0.0);
		
	glBindTexture(GL_TEXTURE_2D, g_texture[1]);						//bind the texture with the keystone image

	glBegin(GL_QUADS);												//draw the keystone image to the screen
	glTexCoord2f(0.0f, 0.0f); glVertex2i(g_imagewidth, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2i(g_imagewidth, g_imageheight);
	glTexCoord2f(1.0f, 1.0f); glVertex2i(0, g_imageheight);
	glTexCoord2f(1.0f, 0.0f); glVertex2i(0, 0);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, g_texture[0]);						//bind the 2nd texture

	glBegin(GL_QUADS);												//draw the 2nd image to the screen
	glTexCoord2f(0.0f, 0.0f); glVertex2i(g_imagewidth, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2i(g_imagewidth, g_imageheight);
	glTexCoord2f(1.0f, 1.0f); glVertex2i(0, g_imageheight);
	glTexCoord2f(1.0f, 0.0f); glVertex2i(0, 0);
	glEnd();

	glPopMatrix();													//reset the MaodelView matrix
	
	glFlush();														//complete any OpenGL related stuff and write to the screen

}
void keyboard(unsigned char key, int, int)
{
	
	switch (key)
	{
	case 'a':
		g_RegX -= 10.0;
		break;
	case 's':
		g_RegX += 10.0;
		break;
	case 'z':
		g_RegY -= 10.0;
		break;
	case 'w':
		g_RegY += 10.0;
		break;
	case '=':
		g_ScaleTranslate += (g_Width / 2) << (g_ZoomLevel);
		g_RegScale *= 2.0;
		g_ZoomLevel++;
		
		break;
	case '-':
		if (g_ZoomLevel > -5){
			g_ZoomLevel--;
			g_ScaleTranslate -= (g_Width / 2) << (g_ZoomLevel);
			g_RegScale /= 2.0;
		}
		break;
	default:
		break;
	}

	//update title with registration shifts
	char reg[256];
	sprintf_s(reg, "Registration Offsets %3.1f X, %3.1f Y", g_RegX, g_RegY);
	glutSetWindowTitle(reg);
	
	glutPostRedisplay();
}
void SpecialInput(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_UP:
		g_RegY += 1.0;
		break;
	case GLUT_KEY_DOWN:
		g_RegY -= 1.0;
		break;
	case GLUT_KEY_LEFT:
		g_RegX -= 1.0;
		break;
	case GLUT_KEY_RIGHT:
		g_RegX += 1.0;
		break;
	}

	char reg[256];
	sprintf_s(reg, "Registration Offsets %3.1f X, %3.1f Y", g_RegX, g_RegY);
	glutSetWindowTitle(reg);
	
	glutPostRedisplay();
}
void MouseCallback(int button, int state, int x, int y){

	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN){
		g_leftdown = true;
		
	}
	else{
		g_leftdown = false;
	}

}
void MotionCallback(int x, int y)
{
	//move the image around in the viewport
	if (g_leftdown && g_mouseX - x < 30 && g_mouseX - x> -30
		&& g_mouseY - y < 30 && g_mouseY - y> -30)
	{
		g_RegX += g_mouseX - x;
		g_RegY -= g_mouseY - y;
		

	}
	g_mouseX = x;
	g_mouseY = y;
	char reg[256];
	sprintf_s(reg, "Registration Offsets %3.1f X, %3.1f Y", g_RegX, g_RegY);
	glutSetWindowTitle(reg);

	glutPostRedisplay();

}
void reshape(int x, int y)
{
	glViewport(0, 0, x, y);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

}

void startGLUT(int argc, char* argv[]);

int main(int argc, char * argv[])
{
	cout << "SWIR Material Identification\n";
	
	//reset the CUDA device 
	cudaDeviceReset();

	//for use with exporting data
	ofstream lOutputFile;
	string lFilename;
	bool lWriteLibrary = false;

	//load the spectral library from USGS
	SpectralData lSpectralData = SpectralData();
	SpectralData::WV3Spectra data2 = lSpectralData.aGetStatNormedWV3Record(0);

	//load the SWIR tiff
	DGDataReader * SWIRImage = new DGDataReader();
	SWIRImage->aLoadXMLDoc(argv[5]);
	SWIRImage->aIntializeMapper();
	SWIRImage->aLoadImages(argv[6]);


	//set up the CUDA variables
	//device side variables
	unsigned short *	dSWIRData[8];													//	holds pointers to the 8 channels of SWIR data
	float *				dReflectanceLibrary;											//	holds reflectance data from USGS on device
	int *				dCurrentMatch;													//	holds the index of the current best reflectance library match for a pixel - its size is image width * image height
	int *				dCurrentTest;
	float *				dCurrentScore;													// holds the current dot product for the pixel and the current match data
	int *				dPMatch;
	float *				dPScore;
	int *				dP2Match;
	float *				dP2Score;
	int*				dLibraryIndices;												// holds the indices that we are concerned with
	int					dLibrarySize = lSpectralData.aGetReducedLibrary()[0].size();	// tells us how many library entries are in the reduced set
	float *				dLibraryMoments;
	int					lReflectanceStride = 9;											//	number of values in a single record of the dReflectanceLibrary data
	int					lNumReflectanceItems = 0;										//  number of records in the dReflectanceLibrary data
	int					lImageWidth = SWIRImage->aGetImageRowLength();					//	number of pixels in the width of the image
	int					lImageHeight = SWIRImage->aGetImageRowsTotal();					//	number of rows in the height of the image

	//host side variables
	float *						hReflectanceLibrary;									//	host variable holding the reflectance data from USGS in a form that can be transfered to dReferenceLibrary
	float *						hReflectanceHistogram;									//	host variable that holds a histogram of the instances of the best matches for each Referencelibrary record
	map<int, unsigned char *>	ReflectanceName;										//map holding the title of the reflectance record in index int
	int *						hReducedLibraryIndices;
	float 						hMoments[SpectralData::numBands * 2];

	//populate mean and variance in hMoments array
	memcpy(hMoments, lSpectralData.aGetMoment(0), sizeof(float)*SpectralData::numBands);
	memcpy(&(hMoments[8]), lSpectralData.aGetMoment(1), sizeof(float)*SpectralData::numBands);

	//setup the reduced library indices
	//we only use part of the library at a time because the library records project onto each other significantly with teh reduced bands
	hReducedLibraryIndices = new int[dLibrarySize];
	for (int i = 0; i < dLibrarySize; i++){
		hReducedLibraryIndices[i] = lSpectralData.aGetReducedLibrary()[0][i];
	}

	//cycle through the library
	for (int i = 0; i < lSpectralData.aGetNumSpectralRecords(); i++){
		data2 = lSpectralData.aGetStatNormedWV3Record(i);
		if (data2.USGSBase.id > 0){
			lNumReflectanceItems++;
		}
	}

	//instantiate library and histogram
	hReflectanceLibrary = new float[lNumReflectanceItems * 9];
	hReflectanceHistogram = new float[lNumReflectanceItems];
	

	//create zeroed out arrays for memcpy operations
	float lFloatEmpty[8];
	float lValue = 0.0;
	int count = 0;

	for (int i = 0; i < 40; i++){
		lFloatEmpty[i % 8] = -1.0;
	}

	//populate the reflectance library related arrays
	for (int i = 0; i < lSpectralData.aGetNumSpectralRecords(); i++){
		data2 = lSpectralData.aGetStatNormedWV3Record(i);
		if (data2.USGSBase.id > 0){
			hReflectanceHistogram[count] = 0.0;
			ReflectanceName[count] = new unsigned char[40];
			memcpy(ReflectanceName[count], &(data2.USGSBase.title), 40);
			hReflectanceLibrary[count * 9] = data2.USGSBase.id;
			for (int j = 0; j < 8; j++){
				lValue += data2.SimulatedBands[j];
			}
			if (lValue > .5){//check and make sure that the data set has something in it.
				memcpy(&(hReflectanceLibrary[count * 9 + 1]), data2.SimulatedBands, 8 * sizeof(float));
			}
			else{
				memcpy(&(hReflectanceLibrary[count * 9 + 1]), lFloatEmpty, 8 * sizeof(float));
			}
			count++;
		}
	}


	//allocate the memory for the SWIR data processing on the device
	cudaMalloc((void**)&(dSWIRData[0]), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dSWIRData[1]), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dSWIRData[2]), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dSWIRData[3]), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dSWIRData[4]), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dSWIRData[5]), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dSWIRData[6]), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dSWIRData[7]), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dCurrentMatch), sizeof(int)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dPMatch), sizeof(int)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dP2Match), sizeof(int)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dCurrentTest), sizeof(int)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dCurrentScore), sizeof(float)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dPScore), sizeof(float)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dP2Score), sizeof(float)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal());
	cudaMalloc((void**)&(dReflectanceLibrary), sizeof(float)*count*lReflectanceStride);
	cudaMalloc((void**)&(dLibraryIndices), sizeof(int)*dLibrarySize);
	cudaMalloc((void**)&(dLibraryMoments), sizeof(float)*SpectralData::numBands * 2);


	//copy variables and arrays to device memory
	cudaMemcpy(dSWIRData[0], SWIRImage->aGetImage(0), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(), cudaMemcpyHostToDevice);
	cudaMemcpy(dSWIRData[1], SWIRImage->aGetImage(1), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(), cudaMemcpyHostToDevice);
	cudaMemcpy(dSWIRData[2], SWIRImage->aGetImage(2), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(), cudaMemcpyHostToDevice);
	cudaMemcpy(dSWIRData[3], SWIRImage->aGetImage(3), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(), cudaMemcpyHostToDevice);
	cudaMemcpy(dSWIRData[4], SWIRImage->aGetImage(4), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(), cudaMemcpyHostToDevice);
	cudaMemcpy(dSWIRData[5], SWIRImage->aGetImage(5), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(), cudaMemcpyHostToDevice);
	cudaMemcpy(dSWIRData[6], SWIRImage->aGetImage(6), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(), cudaMemcpyHostToDevice);
	cudaMemcpy(dSWIRData[7], SWIRImage->aGetImage(7), sizeof(unsigned short)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(), cudaMemcpyHostToDevice);
	cudaMemcpy(dReflectanceLibrary, hReflectanceLibrary, sizeof(float)*count*lReflectanceStride, cudaMemcpyHostToDevice);
	cudaMemcpy(dLibraryMoments, hMoments, sizeof(float)*SpectralData::numBands * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(dLibraryIndices, hReducedLibraryIndices, sizeof(int)*dLibrarySize, cudaMemcpyHostToDevice);

	dim3 lBlockSize;
	lBlockSize.x = 32;
	lBlockSize.y = 32;

	int lBlocksInX = lImageWidth / 32 + 1;
	int lBlocksInY = lImageHeight / 32 + 1;

	dim3 lGridSize;
	lGridSize.x = lBlocksInX;
	lGridSize.y = lBlocksInY;

	//setup host side variables for processing the SWIR dataset
	int * lBestMatch = new int[SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal()];
	float * lBestScore = new float[SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal()];
	int * lPMatch = new int[SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal()];
	float * lPScore = new float[SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal()];
	int * lP2Match = new int[SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal()];
	float * lP2Score = new float[SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal()];
	int * lCurrentMatch = new int[SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal()];
	int * lCurrentTest = new int[SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal()];
	float * lCurrentScore = new float[SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal()];

	int lEndIndex = 0;
	bool lReprocess = false;

	//processes each reflectance record 30 at a time
	//the 30 limit is a function of running this on windows 
	//running with many more will cause windows to believe that the GPU is not responding
	cudaProfilerStart();

	for (int i = 0; i < lNumReflectanceItems; i += 30){
		cout << "Processing " << (float)i / (float)(lNumReflectanceItems)*100.0 << " % complete . \r";
		//make sure that the end index actually represents something in the available dataset
		if (i + 30 < lNumReflectanceItems){
			lEndIndex = i + 30;
		}
		else{
			lEndIndex = lNumReflectanceItems - 1;
		}
		//cudaMemcpy(hReflectanceLibrary, dReflectanceLibrary, sizeof(float)*count*ReflectanceStride, cudaMemcpyDeviceToHost);
		cudaMemcpy(dReflectanceLibrary, hReflectanceLibrary, sizeof(float)*count*lReflectanceStride, cudaMemcpyHostToDevice);

		//start device processing
		ProcessesSWIR << <lGridSize, lBlockSize >> >(dSWIRData[0], dSWIRData[1], dSWIRData[2], dSWIRData[3],
			dSWIRData[4], dSWIRData[5], dSWIRData[6], dSWIRData[7],
			lImageWidth, lImageHeight,
			dReflectanceLibrary, lReflectanceStride, lNumReflectanceItems,
			i, lEndIndex,
			dCurrentMatch, dCurrentScore,
			dPMatch, dPScore,
			dP2Match, dP2Score,
			dLibraryIndices, dLibrarySize,
			dLibraryMoments,
			lReprocess);
		//set the reprocess flag for future processing after the first 30 items in the library
		lReprocess = true; //yes I know this gets executed each time

		//wait for processing to complete
		cudaDeviceSynchronize();
		//I don't know why, but if I don't capture these arrays each cycle they aren't properly calculated there after
		cudaMemcpy(lBestMatch, dCurrentMatch, sizeof(int)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(), cudaMemcpyDeviceToHost);
		cudaMemcpy(lBestScore, dCurrentScore, sizeof(float)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(), cudaMemcpyDeviceToHost);
		cudaMemcpy(lPMatch, dPMatch, sizeof(int)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(), cudaMemcpyDeviceToHost);
		cudaMemcpy(lPScore, dPScore, sizeof(float)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(), cudaMemcpyDeviceToHost);
		cudaMemcpy(lP2Match, dP2Match, sizeof(int)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(), cudaMemcpyDeviceToHost);
		cudaMemcpy(lP2Score, dP2Score, sizeof(float)*SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(), cudaMemcpyDeviceToHost);

	}

	cudaProfilerStop();
	
	g_data = new float[SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal() * 4];	//instantiate g_data array with 4 channels for display later
	bool founditem = false;																		// tells whether or not the pixel should be counted
	float threshold = .935;																		// the threshold to use for saying that the pixel should be counted as a match

	//cycle through the arrays and if the Best match or 2 prior matches' scores are above threshold then we use them
	for (int i = 0; i < SWIRImage->aGetImageRowLength() * SWIRImage->aGetImageRowsTotal(); i++){
		founditem = false;
		if (lBestMatch[i] < lNumReflectanceItems && lBestMatch[i] > -1 && lBestScore[i] > threshold){
			hReflectanceHistogram[lBestMatch[i]]++;
			founditem = true;
		}
		if (lPMatch[i] < lNumReflectanceItems && lPMatch[i] > -1 && lPScore[i] > threshold){
			hReflectanceHistogram[lPMatch[i]]++;
			founditem = true;
		}
		if (lP2Match[i] < lNumReflectanceItems && lP2Match[i] > -1 && lP2Score[i] > threshold){
			hReflectanceHistogram[lP2Match[i]]++;
			founditem = true;
		}
		//assigne the g_data to whichever bands you want 0,4,7 are not magic, they are just the ones I chose
		g_data[i * 4 + 0] = SWIRImage->aGetImage(0)[i];
		g_data[i * 4 + 1] = SWIRImage->aGetImage(4)[i];
		g_data[i * 4 + 2] = SWIRImage->aGetImage(7)[i];

		//the fourth element is the alpha channel and only set if the threshold was exceeded by at least one score.
		if (founditem){
			g_data[i * 4 + 3] = 255;
		}
		else{
			g_data[i * 4 + 3] = 1;
		}

	}
	
	//grab width and height of image for GLUT reasons
	g_imagewidth = SWIRImage->aGetImageRowLength();
	g_imageheight = SWIRImage->aGetImageRowsTotal();

	//display the results
	startGLUT(argc, argv);

	//write out the histogram
	lFilename = "./output/SWIRHistogram1.csv";
	lOutputFile.open(lFilename);
	lOutputFile << "wavelength, reflectance\n";
	for (int i = 0; i < lNumReflectanceItems; i++){
		lOutputFile << ReflectanceName[i] << "," << hReflectanceHistogram[i] << "\n";
	}
	lOutputFile.close();

	//write out the spectral response
	lFilename = "./response.csv";
	lOutputFile.open(lFilename);
	for (int i = 0; i < lNumReflectanceItems; i++){
		lOutputFile << ReflectanceName[i];
		for (int j = 0; j < 8; j++){
			lOutputFile << "," << hReflectanceLibrary[i * 9 + j];
		}
		lOutputFile << "\n";
	}
	lOutputFile.close();

	//write out a subset of the pixel data
	lFilename = "./sdata.csv";
	lOutputFile.open(lFilename);
	for (int i = 0; i < 1000; i++){
		lOutputFile << SWIRImage->aGetImage(0)[i] << "," << SWIRImage->aGetImage(1)[i] << ","
			<< SWIRImage->aGetImage(2)[i] << "," << SWIRImage->aGetImage(3)[i] << ","
			<< SWIRImage->aGetImage(4)[i] << "," << SWIRImage->aGetImage(5)[i] << ","
			<< SWIRImage->aGetImage(6)[i] << "," << SWIRImage->aGetImage(7)[i] << "\n";

	}
	lOutputFile.close();


	//write out the best scores
	lFilename = "./output/Scores.csv";
	lOutputFile.open(lFilename);
	lOutputFile << "pixel, score\n";
	for (int i = 0; i < SWIRImage->aGetImageRowLength(); i++){
		lOutputFile << i << "," << lBestScore[i] << "\n";
	}
	lOutputFile.close();

	lFilename = "./dotproducts.csv";
	lOutputFile.open(lFilename);
	lOutputFile << "index1,index2, score\n";
	cout << "writing out dot products";
	double score = 0.0;

	for (int i = 0; i < lNumReflectanceItems; i++){
		lOutputFile << ReflectanceName[i] << ",";
		for (int j = 0; j < lNumReflectanceItems; j++){
			score = 0.0;

			for (int k = 1; k < 9; k++){
				score += (hReflectanceLibrary[i * 9 + k] - lSpectralData.aGetMoment(0)[k - 1]) / lSpectralData.aGetMoment(1)[k - 1] *
					(hReflectanceLibrary[j * 9 + k] - lSpectralData.aGetMoment(0)[k - 1]) / lSpectralData.aGetMoment(1)[k - 1];// hReflectanceLibrary[j * 9 + k];
			}
			if (j < lNumReflectanceItems - 1){
				lOutputFile << score << ",";
			}
			else{
				lOutputFile << score << "\n";
			}
		}
	}

	lOutputFile.close();

	cout << "\nFinished processing SWIR Image\n\n Push Any Key to Exit.";
	std::cin.get();

	return 0;
}

void startGLUT(int argc, char* argv[]){

	//instantiate local variables
	int lWindow;
	int lWidth = g_Width;
	int lHeight = g_Height;
	bool lIsSWIR = true;
	unsigned char * lTexture1 = new unsigned char[g_imagewidth *g_imageheight * 4];
	unsigned char * lTexture2 = new unsigned char[g_imagewidth *g_imageheight * 4];
	
	//generate Glut window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB);
	glutInitWindowSize(lWidth, lHeight);
	lWindow = glutCreateWindow("SWIR Process Viewer");

	//initialize GLUT environment
	glClearColor(0.0, 0.0, 0.0, 1.0);
	gluOrtho2D(0, lWidth, 0, lHeight);
	glutDisplayFunc(display);
	glutMouseFunc(MouseCallback);
	glutMotionFunc(MotionCallback);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(SpecialInput);

	//load the texture byte arrays with data
	for (int i = 0; i < g_imageheight; i++){
		for (int j = 0; j < g_imagewidth; j++){
			if (lIsSWIR){
				//this texture gets populated with processing detections
				//scaling is 14 bit for SWIR on World View 3
				lTexture1[4 * (i*g_imagewidth + j)] = 0;
				lTexture1[4 * (i*g_imagewidth + j) + 1] = pow(g_data[4 * (i*g_imagewidth + j) + 1], 8.0 / 14.0) - 1.0;
				lTexture1[4 * (i*g_imagewidth + j) + 2] = 0;
				lTexture1[4 * (i*g_imagewidth + j) + 3] = g_data[4 * (i*g_imagewidth + j) + 3];

			}
			else{
				//this is being added for completeness
				//scaling is 12 bit for VNIRS on World View 3

				lTexture1[4 * (i*g_imagewidth + j)] = pow(g_data[4 * (i*g_imagewidth + j)], 8.0 / 12.0) - 1.0;
				lTexture1[4 * (i*g_imagewidth + j) + 1] = pow(g_data[4 * (i*g_imagewidth + j) + 1], 8.0 / 12.0) - 1.0;
				lTexture1[4 * (i*g_imagewidth + j) + 2] = pow(g_data[4 * (i*g_imagewidth + j) + 2], 8.0 / 12.0) - 1.0;
				lTexture1[4 * (i*g_imagewidth + j) + 3] = g_data[4 * (i*g_imagewidth + j) + 3];
			}
			
			//we populate lTexture2 with a gray scale based upone the lTexture1 the choice of bands is arbitarary as the texture is
			//just used for background context
			lTexture2[4 * (i*g_imagewidth + j)] = lTexture1[4 * (i*g_imagewidth + j) + 1];
			lTexture2[4 * (i*g_imagewidth + j) + 1] = lTexture1[4 * (i*g_imagewidth + j) + 1];
			lTexture2[4 * (i*g_imagewidth + j) + 2] = lTexture1[4 * (i*g_imagewidth + j) + 1];
			lTexture2[4 * (i*g_imagewidth + j) + 3] = 128;
		}
	}

	//generate the GL textures and populate them with the byte arrays
	glGenTextures(1, &g_texture[0]);
	glBindTexture(GL_TEXTURE_2D, g_texture[0]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_imagewidth, g_imageheight, 0, GL_RGBA, GL_UNSIGNED_BYTE, lTexture1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glGenTextures(1, &g_texture[1]);
	glBindTexture(GL_TEXTURE_2D, g_texture[1]);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_imagewidth, g_imageheight, 0, GL_RGBA, GL_UNSIGNED_BYTE, lTexture2);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	//setup the GL environment
	glDisable(GL_LIGHTING);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//start the main loop
	glutMainLoop();

}