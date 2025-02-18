#define TILE_WIDTH 8
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
	/*int iLoc = get_local_id(1);*/
	/*int jLoc = get_local_id(0);*/
	/*int iSize = get_local_size(1);*/
	/*int jSize = get_local_size(0);*/
	/*int iNum = get_group_id(1);*/
	/*int jNum = get_group_id(0);*/
	int iGlob = get_global_id(1)+maskRadius;
	int jGlob = get_global_id(0)+maskRadius;
	/*int iIn = iGlob + maskRadius;*/
	/*int jIn = jGlob + maskRadius;*/

	// i and j index something in the result matrix 
	// 1. check that iGlob + maskRadius < width
	// 2. check that jGlob + maskRadius < height

	// bounds check
	if (iGlob < width - maskRadius &&  jGlob < height - maskRadius){
		int accum, xOffset, yOffset, imagePixel, maskValue;
		// channels 
		for(int channel = 0; channel < imageChannels; channel++){
			accum = 0;
			for(int x = -maskRadius; x <= maskRadius; x++){
				for(int y = -maskRadius; y <= maskRadius; y++){
					xOffset = iGlob + x;
					yOffset = jGlob + y;
					if (xOffset >= 0 && xOffset < width &&
							yOffset >= 0 && yOffset < height) {
						imagePixel = inputData[(yOffset * width + xOffset) * imageChannels + channel];
						maskValue = maskData[(y+maskRadius)*maskWidth+x+maskRadius];
						accum += imagePixel * maskValue;
					}
				}
			}
			outputData[((jGlob-maskRadius) * (width-maskRadius*2) + iGlob-maskRadius)*imageChannels + channel] = accum;
		}
	}
}
