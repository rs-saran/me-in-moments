import cv2
import os

class ImageProcessor:
    """
    Handles image preprocessing, such as converting to grayscale, histogram equalization, 
    and saving processed images, with robust error handling.
    """

    def __init__(self):
        """
        Initializes the ImageProcessor class.
        """
        pass

    def preprocess(self, image_path):
        """
        Preprocess an image by generating grayscale and histogram-equalized versions.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: A dictionary containing the original image and processed versions.

        Raises:
            FileNotFoundError: If the input image does not exist.
            ValueError: If the image cannot be read or processed.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # Read the original image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read the image file. It may be corrupted or unsupported: {image_path}")

        try:
            # Convert to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Histogram equalization for grayscale image
            gray_img_he = cv2.equalizeHist(gray_img)

            # Convert to YUV and apply histogram equalization
            yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            yuv_img_he = yuv_img.copy()
            yuv_img_he[:, :, 0] = cv2.equalizeHist(yuv_img_he[:, :, 0])
            yuv_img_he = cv2.cvtColor(yuv_img_he, cv2.COLOR_YUV2BGR)

        except Exception as e:
            raise ValueError(f"Error during image preprocessing: {str(e)}")

        return {
            "original": img,
            "gray": gray_img,
            "gray_hist_eq": gray_img_he,
            "yuv_hist_eq": yuv_img_he,
        }

    def save_processed_images(self, image_dict, output_folder):
        """
        Saves processed images to the specified folder.

        Args:
            image_dict (dict): Dictionary of processed images.
            output_folder (str): Path to the folder where images will be saved.

        Raises:
            OSError: If the output folder cannot be created or files cannot be saved.
        """
        try:
            os.makedirs(output_folder, exist_ok=True)
        except Exception as e:
            raise OSError(f"Failed to create output folder: {output_folder}. Error: {str(e)}")

        file_names = {
            "original": "original.jpg",
            "gray": "gray.jpg",
            "gray_hist_eq": "gray_hist_eq.jpg",
            "yuv_hist_eq": "yuv_hist_eq.jpg",
        }

        for key, image in image_dict.items():
            try:
                save_path = os.path.join(output_folder, file_names[key])
                cv2.imwrite(save_path, image)
            except Exception as e:
                raise OSError(f"Failed to save {key} image to {save_path}. Error: {str(e)}")

    def process_and_save(self, image_path, output_folder):
        """
        Preprocesses an image and saves the processed versions to a folder.

        Args:
            image_path (str): Path to the input image.
            output_folder (str): Path to the folder where processed images will be saved.

        Returns:
            None

        Raises:
            Exception: Propagates exceptions from preprocessing or saving.
        """
        try:
            image_dict = self.preprocess(image_path)
            self.save_processed_images(image_dict, output_folder)
        except (FileNotFoundError, ValueError, OSError) as e:
            raise Exception(f"Error in process_and_save: {str(e)}")
