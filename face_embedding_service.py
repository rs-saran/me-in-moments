import os
import numpy as np
from deepface import DeepFace


class FaceEmbeddingService:
    """
    Manages embedding generation, retrieval, and comparison for facial images with robust error handling.
    """

    def __init__(self, model_name="Facenet512", detector_backend="retinaface"):
        """
        Initializes the EmbeddingManager with specified model and detector.

        Args:
            model_name (str): The name of the DeepFace model to use for embeddings.
            detector_backend (str): The backend for face detection.
        """
        self.model_name = model_name
        self.detector_backend = detector_backend

    def get_embeddings(self, image_path):
        """
        Generates embeddings for a given image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            list: A list of embeddings with additional metadata (e.g., facial area).

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If embeddings cannot be generated.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at: {image_path}")

        try:
            embeddings = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=False
            )
            return embeddings
        except Exception as e:
            raise ValueError(f"Error generating embeddings for {image_path}: {str(e)}")

    def get_folder_embeddings(self, folder_path, ref_image=False):
        """
        Generates embeddings for all images in a folder.

        Args:
            folder_path (str): Path to the folder containing images.
            ref_image (bool): Indicates whether the embeddings are for a reference image.

        Returns:
            list: A list of embeddings for all images in the folder.

        Raises:
            FileNotFoundError: If the folder does not exist or contains no images.
            ValueError: If a reference image has zero or multiple faces.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found at: {folder_path}")

        image_paths = [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if not image_paths:
            raise FileNotFoundError(f"No valid images found in folder: {folder_path}")

        all_embeddings = []
        for image_path in image_paths:
            try:
                embeddings = self.get_embeddings(image_path)
            except (FileNotFoundError, ValueError) as e:
                raise ValueError(f"Error processing image {image_path}: {str(e)}")

            image_type = os.path.splitext(os.path.basename(image_path))[0]

            if ref_image and len(embeddings) != 1:
                raise ValueError(f"Reference image must contain exactly one face.")

            if len(embeddings) == 0:
                embeddings = [{"embedding": [0] * 512}]

            for embedding in embeddings:
                embedding['image_type'] = image_type
            all_embeddings.extend(embeddings)

        return all_embeddings

    def compare_embeddings(self, target_embeddings, ref_embeddings):
        """
        Compares embeddings of target images with reference embeddings.

        Args:
            target_embeddings (list): List of target embeddings with metadata.
            ref_embeddings (list): List of reference embeddings with metadata.

        Returns:
            list: Comparison results with similarity scores and bounding box details.

        Raises:
            ValueError: If inputs are empty or not properly structured.
        """
        def cosine_similarity(v1, v2):
            """
            Computes cosine similarity between two vectors.

            Args:
                v1 (list): First vector.
                v2 (list): Second vector.

            Returns:
                float: Cosine similarity score.
            """
            v1 = np.array(v1)
            v2 = np.array(v2)
            return np.dot(v1, v2) / ((np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-10)

        if not target_embeddings or not ref_embeddings:
            raise ValueError("Target or reference embeddings cannot be empty.")

        results = []
        try:
            for target in target_embeddings:
                for ref in ref_embeddings:
                    similarity = 1 - cosine_similarity(target["embedding"], ref["embedding"])
                    results.append({
                        # "target_image_type": target["image_type"],
                        # "ref_image_type": ref["image_type"],
                        "similarity_score": similarity,
                        # "target_bd": target.get("facial_area"),
                        # "ref_bd": ref.get("facial_area"),
                    })
        except Exception as e:
            raise ValueError(f"Error during embedding comparison: {str(e)}")

        return results
