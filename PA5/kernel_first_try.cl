#define TILE_WIDTH 8
#define IMAGE_CHANNELS 3
__kernel void convolution2D(
    __global int * inputData, __global int * outputData, __constant int * maskData,
    int width, int height, int maskWidth,  int imageChannels, int stride){
    //@@ Insert code to implement matrix multiplication here
   
    /**
    maskWidth := 5
    maskRadius := maskWidth/2 # this is integer division, so the result is 2
    for i from 0 to height do
        for j from 0 to width do
            for k from 0 to channels
                accum := 0
                for y from -maskRadius to maskRadius do
                    for x from -maskRadius to maskRadius do
                        xOffset := j + x
                        yOffset := i + y
                        if xOffset >= 0 && xOffset < width &&
                            yOffset >= 0 && yOffset < height then
                            imagePixel := I[(yOffset * width + xOffset) * channels + k]
                            maskValue := K[(y+maskRadius)*maskWidth+x+maskRadius]
                            accum += imagePixel * maskValue
                        end
                    end
                end
                # pixels are in the range of 0 to 1
                P[(i * width + j)*channels + k] = clamp(accum, 0, 1)
            end
        end
    end**/

	// define variables
	// maskWidth already defined
	unsigned int maskRadius = maskWidth / 2;
	int iLoc = get_local_id(1);
	int jLoc = get_local_id(0);
	int iGlob = get_global_id(1); // shift by mask radius
	int jGlob = get_global_id(0);
	int xOffset, yOffset, xOffsetLocal, yOffsetLocal;

	// load memory 
	__local int tileMem[TILE_WIDTH][TILE_WIDTH][3];
	// the 3 is imageChannels. there's always 3.
	if(iGlob < width && jGlob < height){
		tileMem[jLoc][iLoc][0] = inputData[jGlob * height*3 + iGlob*3]; // load from memory
		tileMem[jLoc][iLoc][1] = inputData[jGlob * height*3 + iGlob*3 + 1]; // load from memory
		tileMem[jLoc][iLoc][2] = inputData[jGlob * height*3 + iGlob*3 + 2]; // load from memory
	}

	barrier(CLK_LOCAL_MEM_FENCE); // end of load



	// calculate 
	// each work item handles one value
	int accum[3] = {0,0,0};
	if(iGlob < width && jGlob < height){
		for (int x = -maskRadius; x <= maskRadius; x++){
			for (int y = -maskRadius; y <= maskRadius; y++){
				int maskValue = maskData[(y+maskRadius)*maskWidth+x+maskRadius];
				xOffset = jLoc + x;
				yOffset = iLoc + y;
				if(xOffset >= 0 && xOffset < width
						&& yOffset >= 0 && yOffset < height){
					accum[0] += tileMem[yOffset][xOffset][0] * maskValue;
					accum[1] += tileMem[yOffset][xOffset][1] * maskValue;
					accum[2] += tileMem[yOffset][xOffset][2] * maskValue;
				}
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE); // end of calc

	if(iGlob < width && jGlob < height){
		int baseInd = (jGlob * width + iGlob) * 3;
		for(int i = 0; i < 3; i++){
			outputData[baseInd+i] = accum[i];
		}
	}
	// wrap around
}
