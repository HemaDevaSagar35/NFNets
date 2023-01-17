import numpy as np
from skimage.filters import gaussian ## skimage is the library I am adding new
from skimage.transform import rotate 
from matplotlib import pyplot as plt


"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    ### YOUR CODE HERE
    image = record.reshape((3, 32, 32))
    #image = np.transpose(depth_major, [1, 2, 0])

    ### END CODE HERE

    image = preprocess_image(image, training) # If any.

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [3, 32, 32].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32]. The processed image.
    """
    ### YOUR CODE HERE
    if training:
        padder = np.zeros((3, 32, 4))
        image = np.concatenate([padder, image, padder], axis=2)
        
        padder = np.zeros((3, 4, 40))
        image = np.concatenate([padder, image, padder], axis=1)

        lp = np.random.randint(9, size=(1,2))
        x = lp[0,0]
        y = lp[0,1]
        image = image[:, x:x+32, y:y+32]

        # horizontal flipping, rotate, gaussian blur, normal image
        signal = np.random.randint(1, 10, 1)[0]
        #gf = gaussian(channnel_axis=0)
        if (signal == 5) or (signal == 6):
            #gaussian blur
            #print('blurring')
            image = gaussian(image, channel_axis=0)
            
        elif (signal == 7) or (signal == 8):
            # rotate
            #print("rotate")
            #print(image.shape)
            angle = np.random.randint(-15, 15, 1)[0]

            image = rotate(image, angle)
        elif (signal == 9) or (signal == 10):
            #print("flip")
            image = np.flip(image, axis=2)
        
    image = (image - np.mean(image))/np.std(image)
    ### END CODE HERE

    return image


def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    image = image.reshape((3, 32, 32))
    image = np.transpose(image, [1, 2, 0])
    ### YOUR CODE HERE
    
    plt.imshow(image)
    plt.savefig(save_name)
    return image

# Other functions
### YOUR CODE HERE

### END CODE HERE