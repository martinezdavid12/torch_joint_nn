import torch

def split_reim_channels(array):
    """Split a complex valued tensor into its real and imaginary parts.

    Args:
      array(complex): A tensor of shape (batch_size, N, N) or (batch_size, N, N, 1)

    Returns:
      split_array(float): A tensor of shape (batch_size, N, N, 2) containing the real part on one channel and the imaginary part on another channel

    """
    real = array.real
    imag = array.imag
    split_array = torch.cat((real, imag), dim=1)
    return split_array

def join_reim_channels(array):
    """Join the real and imaginary channels of a tensor to form a complex-valued tensor.

    Args:
      array (torch.Tensor): A real-valued tensor of shape (batch_size, ch, H, W)

    Returns:
      joined_array (torch.Tensor): A complex-valued tensor of shape (batch_size, ch/2, H, W)
    """
    ch = array.shape[1]  # Get number of channels (C)
    # Ensure the number of channels is even
    assert ch % 2 == 0, "Number of channels must be even to separate real and imaginary parts."
    # Split channels into real and imaginary parts
    real_part = array[:, :ch // 2, :, :]
    imag_part = array[:, ch // 2:, :, :]
    # Combine into a complex tensor
    joined_array = torch.complex(real_part, imag_part)  # Shape: (batch_size, ch/2, H, W)
    return joined_array

def convert_channels_to_freq(images):
    """Convert a tensor of images to their Fourier transforms.

    The tensor contains ch channels representing ch/2 real parts and ch/2 imag parts.

    Args:
      images(float): A tensor of shape (batch_size, C, H, W)

    Returns:
      spectra(float): An FFT-ed tensor of shape (batch_size, C, H, W)

    """
    reim_imgs = join_reim_channels(images) # complex tensor (N, C/2, H, W)
    fft_imgs = torch.fft.fft2(reim_imgs)  # Shape remains (N, C/2, H, W) but complex
    # Apply FFT shift along spatial dimensions (H, W)
    split_imgs = split_reim_channels(fft_imgs)
    spectra = torch.fft.fftshift(split_imgs, dim=(2, 3))  # Shift along H and W
    return spectra

def convert_channels_to_image(images):
    """Convert a tensor of Fourier spectra to the corresponding images.

    The tensor contains ch channels representing ch/2 real parts and ch/2 imag parts.

    Args:
      spectra(float): An array of shape (batch_size, C, H, W)

    Returns:
      images(float): An IFFT-ed array of shape (batch_size, C, H, W)

    """
    shifted_spectra = torch.fft.fftshift(images, dim=(2, 3))
    reim_spectra = join_reim_channels(shifted_spectra) # channel --> complex tensor (N, C/2, H, W)
    perm_images = torch.fft.ifft2(reim_spectra)  # Shape remains (N, C/2, H, W) but complex
    split_images = split_reim_channels(perm_images)
    return split_images