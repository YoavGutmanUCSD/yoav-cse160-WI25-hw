import os
import numpy as np
import random
import sys
from typing import List, Tuple

# without stride
matrix_sizes_without_stride = [
    ((3, 3),    (1, 1), [1]),
    ((64,64),   (5, 5), [1]),    
    ((64, 45),  (3, 3), [1]),
    ((128, 64), (5, 5), [1]), 
    ((64, 128), (5, 5), [1]),
    ((32, 32),  (5, 5), [1]),
    ((48, 64),  (5, 5), [1]),
    ((64, 5),   (5, 5), [1]), 
    ((64, 36),  (3, 3), [1]), 
    ((228, 128), (5, 5), [1]), 
    ((28, 12),   (5, 5), [1]),
    ((12, 32),   (5, 5), [1]),
    ((48, 1024), (5, 5), [1]),
    ((512, 512), (7, 7), [1]), 
    ((1024, 1025), (17, 17), [1]),
    ((2048, 2048), (12, 12), [1]),
]

# with stride
matrix_sizes_with_stride = [
    ((3, 3),  (1, 1), [1,2,4]),
    ((64,64), (5, 5), [1,2,4]),    
    ((64, 45), (3, 3), [1,2]),
    ((128, 64), (5, 5), [1,4]), 
    ((64, 128), (5, 5), [1,2]),
    ((32, 32), (5, 5), [1,2,4]),
    ((48, 64), (5, 5), [1,2,4]),
    ((64, 5), (5, 5), [1]), 
    ((64, 36), (3, 3), [1]), 
    ((228, 128), (5, 5), [1,2,4]), 
    ((28, 12), (5, 5), [1,2]),
    ((12, 32), (5, 5), [1]),
    ((48, 1024), (5, 5), [1,2,4]),
    ((512, 512), (7, 7), [1,2,4]), 
    ((1024, 1025), (17, 17), [1,2,4]),
    ((2048, 2048), (12, 12), [1,2,4]),
]

def compute_output_size(input_size: Tuple[int, int], kernel_size: Tuple[int, int], stride: int) -> Tuple[int, int]:
    rows, cols = input_size
    K, _ = kernel_size
    output_rows = (rows - K ) // stride + 1
    output_cols = (cols - K) // stride + 1
    return (output_rows, output_cols)

def generate_datasets(use_strides: bool = False):
    matrix_sizes = matrix_sizes_with_stride if use_strides else matrix_sizes_without_stride

    parent_folder = "with_strides" if use_strides else "without_strides"
    os.makedirs(parent_folder, exist_ok=True)
    
    images = [np.random.randint(low=0,
            high=255,
            size = (sizes[0][0], sizes[0][1], 3),
            dtype = np.int32) for sizes in matrix_sizes]

    kernels = []
    for sizes in matrix_sizes:
        k_height, k_width = sizes[1]
        tmp = np.random.randint(
            low=0,
            high=255,
            size=(k_height, k_width),
            dtype=np.int32
        )
        tmp = (tmp + np.rot90(tmp, 2)) // 2
        kernels.append(tmp)


    counter = 0

    for idx in range(len(matrix_sizes)):
        image = images[idx]
        kernel = kernels[idx]
        rows, cols = matrix_sizes[idx][0]
        K, _ = matrix_sizes[idx][1]
        strides = matrix_sizes[idx][2] 
        print(f"Generating dataset {counter}")
        
        for stride in strides:
            dataset_dir = os.path.join(parent_folder, str(counter))
            os.makedirs(dataset_dir, exist_ok=True)
            # if use_strides:
            #     print(f"Generating dataset {counter} with stride {stride}")

            # output size based on stride
            output_rows, output_cols = compute_output_size((rows, cols), (K, K), stride)
            output = np.zeros((output_rows, output_cols, 3), dtype=np.int32)

            # Save input image
            img_path = os.path.join(dataset_dir, f"input0.raw")
            with open(img_path, 'w') as f:
                f.write(f"# ({rows}, {cols}, 3)\n")
                for row in image:
                    for col in row:
                        for channel in col:
                            f.write(f"{channel} ")
                    f.write("\n")

            # Save kernel
            kernel_path = os.path.join(dataset_dir, f"kernel0.raw")
            with open(kernel_path, 'w') as f:
                f.write(f"# ({K}, {K})\n")
                for row in kernel:
                    for col in row:
                        f.write(f"{col} ")
                    f.write("\n")

            maskWidth = K
            maskRadius = maskWidth//2
            # Convolution
            for i in range(0, rows - K + 1, stride):
                for j in range(0, cols - K + 1, stride):
                    output_row = i // stride
                    output_col = j // stride
                    for k in range(3):
                        accum = 0
                        for y in range(0, maskWidth):
                            for x in range(0, maskWidth):
                                # xOffset = j + (x -maskRadius)
                                # yOffset = i + (y -maskRadius)
                                xOffset = j + x
                                yOffset = i + y
                                if xOffset >= 0 and xOffset < cols and yOffset >= 0 and yOffset < rows:
                                    imagePixel = image[yOffset, xOffset, k]
                                    maskValue = kernel[y][x]
                                    accum += imagePixel * maskValue
                        output[output_row, output_col, k] = accum
            
            stride_path = os.path.join(dataset_dir, f"stride.raw")
            with open(stride_path, 'w') as f:
                f.write(f"{stride}")

            # Save output
            out_path = os.path.join(dataset_dir, f"output.raw")
            with open(out_path, 'w') as f:
                f.write(f"# ({output_rows}, {output_cols}, 3)\n")
                for i in range(output_rows):
                    for j in range(output_cols):
                        for k in range(3):
                            f.write(f"{output[i, j, k]} ")
                    f.write("\n")
            
            counter += 1

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--with_strides":
        generate_datasets(use_strides=True)
    else:
        generate_datasets(use_strides=False)