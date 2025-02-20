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
	int iLoc = get_local_id(1);
	int jLoc = get_local_id(0);
	int iSize = get_local_size(1);
	int jSize = get_local_size(0);
	int iNum = get_group_id(1);
	int jNum = get_group_id(0);
	int iGlob = get_global_id(1);
	int jGlob = get_global_id(0);

	__local int tileMem[3][TILE_WIDTH][TILE_WIDTH];

	/*// load memory*/
	if (iGlob < width && jGlob < height){
		for(int k = 0; k < imageChannels; k++){
			for(int j = jLoc; j < TILE_WIDTH; j+= jSize){
				for(int i = iLoc; i < TILE_WIDTH; i+= iSize){
					int yW = j+jSize*jNum;
					int xW = i+iSize*iNum;
					if(xW < height && yW < width){
						int write = inputData[(yW*width+xW)*imageChannels+k];
						tileMem[k][j][i] = write;
					}
					else{
						/*printf("can I write to (%d,%d)? %s\n", xW, yW, (xW < height && yW < width) ? "YES" : "NO");*/
						/*printf("(%d,%d) compared (width, height) of (%d,%d). can't write\n", yW, xW, width, height);*/
						tileMem[k][j][i] = 0;
					}
				}
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	/*if(iLoc == 0 && jLoc == 0){*/
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
	if (iGlob < width - maskRadius &&  jGlob < height - maskRadius
			&& iGlob - maskRadius >= 0 && jGlob - maskRadius >= 0
			){
		int accum, xOffset, yOffset, imagePixel, maskValue, xOffsetTile, yOffsetTile;
		// channels 
		for(int channel = 0; channel < imageChannels; channel++){
			accum = 0;
			for(int x = -maskRadius; x <= maskRadius; x++){
				for(int y = -maskRadius; y <= maskRadius; y++){
					/*xOffset = iGlob + x;*/
					/*yOffset = jGlob + y;*/
					xOffsetTile = iLoc + x+maskRadius;
					yOffsetTile = jLoc + y+maskRadius;
					/*xOffset = iLoc + x +iSize*iNum;*/
					/*yOffset = jLoc + y +jSize*jNum;*/
					/*if (xOffset >= 0 && xOffset < width &&*/
					/*		yOffset >= 0 && yOffset < height) {*/
					if (xOffsetTile >= 0 && xOffsetTile < TILE_WIDTH &&
							yOffsetTile >= 0 && yOffsetTile < TILE_WIDTH) {
						/*imagePixel = inputData[(yOffset * width + xOffset) * imageChannels + channel];*/
						imagePixel = tileMem[channel][yOffsetTile][xOffsetTile];
						maskValue = maskData[(y+maskRadius)*maskWidth+x+maskRadius];
						// maskValue = maskData[y+maskRadius][x+maskRadius]
						// imagePixel = tileMem[channel][yOffsetTile][xOffsetTile]
						accum += imagePixel * maskValue;
					}
					else {
						printf("oh\n");
					}
				}
			}
			outputData[((jGlob-maskRadius) * (width-maskRadius*2) + iGlob-maskRadius)*imageChannels + channel] = accum;
		}
	}
}
