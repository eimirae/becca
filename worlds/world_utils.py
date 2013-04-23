import matplotlib.pyplot as plt
import numpy as np

import agent.utils as ut

"""
Utilities shared between several worlds dealing with visual input
"""

def center_surround(fov, fov_span, verbose=False):
    """ 
    Convert a 2D array of b/w pixel values to center-surround 
    
    fov (field of view) is the 2D array of pixel values and 
    fov_span is the number of center-surround superpixel rows and columns.
    Returns a 2D array of the center surround vales.
    """ 
    fov_height = fov.shape[0]
    fov_width = fov.shape[1]
    block_width = np.round(fov_width / (fov_span + 2))
    block_height = np.round(fov_height / (fov_span + 2))
    super_pixels = np.zeros((fov_span + 2, fov_span + 2))
    center_surround_pixels = np.zeros((fov_span, fov_span))
    # Create the superpixels by averaging pixel blocks
    for row in range(fov_span + 2):
        for column in range(fov_span + 2):
            super_pixels[row][column] = np.mean(
                    fov[row * block_height:(row + 1) * block_height,
                        column * block_width: (column + 1) * block_width ])
    for row in range(fov_span):
        for column in range(fov_span):
            # Calculate a center-surround value that represents
            # the difference between the pixel and its surroundings.
            center_surround_pixels[row][column] = \
                super_pixels[row + 1][column + 1] - \
                super_pixels[row    ][column + 1] / 6 - \
                super_pixels[row + 2][column + 1] / 6 - \
                super_pixels[row + 1][column    ] / 6 - \
                super_pixels[row + 1][column + 2] / 6 - \
                super_pixels[row    ][column    ] / 12 - \
                super_pixels[row + 2][column    ] / 12 - \
                super_pixels[row    ][column + 2] / 12 - \
                super_pixels[row + 2][column + 2] / 12
    if verbose:
        # Display the field of view clipped from the original image
        plt.figure("fov")
        plt.gray()
        im = plt.imshow(fov)
        im.set_interpolation('nearest')
        plt.title("field of view")
        plt.draw() 
        # Display the pixelized version, a.k.a. superpixels
        plt.figure("super_pixels")
        plt.gray()
        im = plt.imshow(super_pixels)
        im.set_interpolation('nearest')
        plt.title("super pixels")
        plt.draw() 
        # Display the center-surround filtered superpixels
        plt.figure("center_surround_pixels")
        plt.gray()
        im = plt.imshow(center_surround_pixels)
        im.set_interpolation('nearest')
        plt.title("center surround pixels")
        plt.draw() 
    return center_surround_pixels

def vizualize_pixel_array_feature(feature, level_index, feature_index, 
                                  world_name=None, save_eps=False, 
                                  save_jpg=False, filename='log/feature'):
    """ Show a visual approximation of an array of center-surround features """
    # Calculate the number of pixels that span the field of view
    n_pixels = feature.shape[0]/ 2
    fov_span = np.sqrt(n_pixels)
    level_str = str(level_index).zfill(2)
    feature_str = str(feature_index).zfill(3)
    fig_title = ('Level ' + level_str + ' Feature ' + feature_str + 
                 ' from ' + world_name)
    fig_name = 'Features from ' + world_name
    fig = plt.figure(ut.str_to_int(fig_name))
    fig.clf()
    num_states = feature.shape[1]
    for state_index in range(num_states):
        feature_sensors = feature[:,state_index]
        # Maximize contrast
        feature_sensors *= 1 / (np.max(feature_sensors) + ut.EPSILON)
        pixel_values = ((feature_sensors[ 0:n_pixels] - 
                         feature_sensors[n_pixels:2 * n_pixels]) + 1.0) / 2.0
        feature_pixels = pixel_values.reshape(fov_span, fov_span)
        plt.gray()
        ax = fig.add_axes((float(state_index)/float(num_states), 0., 
                           1/float(num_states), 1.), frame_on=False)
        im = plt.imshow(feature_pixels, vmin=0.0, vmax=1.0, 
                        interpolation='nearest')
    filename = (filename + '_' + world_name  + '_' + level_str + 
                   '_' + feature_str + '.png')
    fig.savefig(filename, format='png')
    '''if save_eps:
        epsfilename = (filename + '_' + world_name  + '_' + level_str + 
                       '_' + feature_str + '.eps')
        fig.savefig(epsfilename, format='eps')
    if save_jpg:
        try:
            jpgfilename = (filename + '_' + world_name  + '_' + level_str + 
                           '_' + feature_str + '.jpg')
            fig.savefig(jpgfilename, format='jpg')
        except:
            print("I think you need to have PIL installed to print " +
                  "in .jpg format.")
            save_jpg = False
    '''
    fig.show()
    fig.canvas.draw()
    return