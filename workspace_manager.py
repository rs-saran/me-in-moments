import os

class WorkspaceManager:
    """
    Handles directory and file management for image processing workspaces.
    """

    def __init__(self, base_workspace="snapsearch_workspace"):
        """
        Initializes the WorkspaceManager and sets up the base workspace path.

        Args:
            base_workspace (str): The root directory for storing all workspace-related files.
        """
        self.base_workspace = base_workspace
        self.ref_workspace = os.path.join(base_workspace, "ref_image_ws")
        self.target_workspace = os.path.join(base_workspace, "target_image_ws")
        self.target_repo = os.path.join(base_workspace, "target_repo")

        self.create_workspace()

    def create_workspace(self):
        """
        Creates necessary directories for the workspace.
        """
        os.makedirs(self.base_workspace, exist_ok=True)
        os.makedirs(self.ref_workspace, exist_ok=True)
        os.makedirs(self.target_workspace, exist_ok=True)
        os.makedirs(self.target_repo, exist_ok=True)

    def save_file(self, uploaded_file, save_path):
        """
        Saves an uploaded file to the specified path.

        Args:
            uploaded_file: The uploaded file object from Streamlit.
            save_path (str): The full path where the file should be saved.
        """
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

    def get_image_paths(self, folder_path, extensions=("jpg", "jpeg", "png")):
        """
        Retrieves paths to all images in a folder.

        Args:
            folder_path (str): The directory to search for images.
            extensions (tuple): Allowed file extensions.

        Returns:
            list: A list of full paths to images in the folder.
        """
        return [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.lower().endswith(extensions)
        ]

    def clear_workspace(self):
        """
        Clears the entire workspace by deleting its contents.
        """
        import shutil
        if os.path.exists(self.base_workspace):
            shutil.rmtree(self.base_workspace)
        self.create_workspace()
