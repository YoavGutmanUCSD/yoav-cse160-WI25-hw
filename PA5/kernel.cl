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
	int maskRadius = maskWidth/2;
	int iLoc = get_local_id(1);
	int jLoc = get_local_id(0);
	int iSize = get_local_size(1);
	int jSize = get_local_size(0);
	int iNum = get_group_id(1);
	int jNum = get_group_id(0);
	int iGlob = get_global_id(1)+maskRadius;
	int jGlob = get_global_id(0)+maskRadius;
	/*printf("%d\n", maskRadius);*/

	/*if(jGlob - maskRadius >= 0 && iGlob - maskRadius >= 0*/
	/*		&& jGlob + maskRadius < width*/
	/*		&& iGlob + maskRadius < height*/
	/*		){*/
	/*if(jGlob < width && iGlob < height){*/
	for(int k = 0; k < imageChannels; k++){
		int accum = 0;
		for(int y = -maskRadius; y <= maskRadius; y++){
			for(int x = -maskRadius; x <= maskRadius; x++){
				int xOffset = jGlob + x;
				int yOffset = iGlob + y;
				if (xOffset >= 0 && xOffset < width
						&& yOffset >= 0 && yOffset < height){
					int imagePixel = inputData[(yOffset * width + xOffset)*imageChannels + k];
					int maskValue = maskData[(y+maskRadius)*maskWidth+x+maskRadius];
					/*printf("%d ", accum);*/
					/*printf("%d,%d\n ", imagePixel, maskValue);*/
					accum += (imagePixel * maskValue);
				}
			}
		}
		if(accum == 0){
			/*printf("fuck\n");*/
		}
		outputData[((iGlob-maskRadius)*width+jGlob-maskRadius)*imageChannels+k] = accum;
	}
}
/*}*/
