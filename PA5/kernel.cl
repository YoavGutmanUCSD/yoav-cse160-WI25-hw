#define TILE_WIDTH 30
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
	int maskRadius = maskWidth/2;
	int xLoc = get_local_id(1);
	int yLoc = get_local_id(0);
	int xSize = get_local_size(1);
	int ySize = get_local_size(0);
	int xNum = get_group_id(1);
	int yNum = get_group_id(0);
	int xGlob = get_global_id(1);
	int yGlob = get_global_id(0);

	// remember:
	// i is x
	// j is y

	// remember:
	// we index by output tile now. each work item produces an item of output. 
	// have to go from -maskRadius to TILE_WIDTH+maskRadius. so if xLoc >= 0, go to TILE_WIDTH+maskRadius*2, 
	//   and subtract maskRadius from the indexed spot.
	// Now, there is potential in treating tileMem as a significantly larger container that is partially used.
	// I believe it will be simpler to use it first in a more normal way. Row and col by TILE_WIDTH dims.
	// Later on it can be flattened.
	//
	// So, tileMem[k][j][i] = inputData[k][j+TILE_WIDTh*jNum - maskRadius][i+TILE_WIDTH*iNum - maskRadius]
	// j = jLoc; j < TILE_WIDTH+2*maskRadius; j+= jSize
	// i = iLoc; i < TILE_WIDTH+2*maskRadius; i+= iSize
	// k = 0; k < imageChannels; k++
	//
	// PROBLEM: TILE_WIDTH + 2 * maskRadius is out of bounds! 
	// Solution: have to make it maskWidth + 2 * maskRadius OR just < TILE_WIDTH
	//   Chosen: maskWidth for now, since 
	//
	// PROBLEM: jLoc and iLoc can be 0. We don't want negative indices. 
	//   Solution: implicitly assume jLoc and iLoc are translated down, and must be translated back up again.
	//   tileMem[k][j][i] = inputData[k][j+jSize*jNum][i+TILE_WIDTH*iNum]
	//   LATER ON: inputData[k][yGlob * width + xGlob +- maskRadius] 
	//     = tileMem[k][(yLoc +maskRadius +- maskRadius) * TILE_WIDTH + xLoc + maskRadius +- maskRadius]

	__local int tileMem[3][TILE_WIDTH][TILE_WIDTH];

	/*// load memory*/
	if (xGlob < width && yGlob < height){
		for(int j = yLoc; j < maskWidth + 2 * maskRadius; j += ySize){
			for(int i = xLoc; i < maskWidth + 2 * maskRadius; i += xSize){
				for(int k = 0; k < imageChannels; k++){
					int indeX = i + TILE_WIDTH * xNum;
					int indeY = j + TILE_WIDTH * yNum;
					if(indeX < width && indeY < height)
						tileMem[k][j][i] = inputData[k+imageChannels*(indeY*width+indeX)];
					else 
						tileMem[k][j][i] = 0;
				}
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	/*if(xLoc == 0 && yLoc == 0){*/
	/*	for(int j = 0; j < TILE_WIDTH; j++){*/
	/*		for(int i = 0; i < TILE_WIDTH; i++){*/
	/*			for(int k = 0; k < 3; k++){*/
	/*				printf("%d ", tileMem[k][j][i]);*/
	/*			}*/
	/*		}*/
	/*		printf("\n");*/
	/*	}*/
	/*}*/


	// bounds check for input dimension work items
	// if (xGlob < width - maskRadius &&  yGlob < height - maskRadius
	// 		&& xGlob - maskRadius >= 0 && yGlob - maskRadius >= 0
	// 		){
	// bounds check for output dimension work items
	if(xGlob < width - 2*maskRadius && yGlob < height - 2*maskRadius
			/*&& xGlob >= 0 && yGlob >= 0*/
			){
		int accum, xOffset, yOffset, imagePixel, maskValue, xOffsetTile, yOffsetTile;
		// channels 
		for(int channel = 0; channel < imageChannels; channel++){
			accum = 0;
			for(int y = -maskRadius; y <= maskRadius; y++){
				for(int x = -maskRadius; x <= maskRadius; x++){
					/*xOffset = xGlob + x + maskRadius;*/
					/*yOffset = yGlob + y + maskRadius;*/
					xOffsetTile = xLoc + x+maskRadius;
					yOffsetTile = yLoc + y+maskRadius;
					/*xOffset = xLoc + x +xSize*xNum;*/
					/*yOffset = yLoc + y +ySize*yNum;*/
					/*if (xOffset >= 0 && xOffset < width &&*/
					/*		yOffset >= 0 && yOffset < height) {*/
					if (xOffsetTile >= 0 && xOffsetTile < maskRadius*2+maskWidth &&
							yOffsetTile >= 0 && yOffsetTile < maskRadius*2+maskWidth) {
						/*imagePixel = inputData[(yOffset * width + xOffset) * imageChannels + channel];*/
						imagePixel = tileMem[channel][yOffsetTile][xOffsetTile];
						maskValue = maskData[(y+maskRadius)*maskWidth+x+maskRadius];
						// maskValue = maskData[y+maskRadius][x+maskRadius]
						// imagePixel = tileMem[channel][yOffsetTile][xOffsetTile]
						accum += imagePixel * maskValue;
					}
					else {
						printf("oh. (%d < %d,%d < %d)\n", xOffset, yOffset, width, height);
					}
				}
			}
			outputData[(yGlob * (width-maskRadius*2) + xGlob)*imageChannels + channel] = accum;
		}
	}
}
