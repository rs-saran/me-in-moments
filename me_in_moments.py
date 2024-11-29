import os
import pandas as pd
import streamlit as st
from workspace_manager import WorkspaceManager
from image_processor import ImageProcessor
from face_embedding_service import FaceEmbeddingService
import zipfile


# Streamlit App
st.set_page_config(page_title="Me in Moments", page_icon="📸")
st.title("📸 Me in Moments")
st.subheader("Lost in a sea of event photos?")
st.write(
    """
    You know the drill: everyone’s snapping pics at events, weddings, or parties, 
    and they’re scattered across a dozen devices and apps. Finding *your* photos? 
    A total nightmare. 
    
    That’s where **Me in Moments** comes in! This fun little app is here to save 
    the day (and your sanity).It helps you track down all your best moments—whether 
    they’re on your phone, your friend’s camera, or some random cloud folder. 
    
    Upload your photos, Let’s find *you* in those moments that matter!
    """
)

# Define session state keys for managing reset states
if "res_df" not in st.session_state:
    st.session_state.res_df = None  # Initialize res_df as None if not available


st.write("🚀 Ready to get started? Upload your photos below!")

# File Uploaders
uploaded_ref_image = st.file_uploader("📸 Upload Your Photo (Reference Image)", type=["jpg", "jpeg", "png"])
uploaded_target_images = st.file_uploader("📂 Upload Photos to Search (Target Images)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Create WorkspaceManager, ImageProcessor, and FaceEmbeddingService instances
image_processor = ImageProcessor()
embedding_service = FaceEmbeddingService()


# Recompute similarity scores if both images are uploaded and res_df is None
if uploaded_ref_image and uploaded_target_images and st.session_state.res_df is None:

    status_text_p = st.empty()
    
    
    # Paths for saving the uploaded files
    workspace_path = "meinmoments_workspace"
    status_text_p.write("Creating temporary workspace...")
    workspace_manager = WorkspaceManager(base_workspace=workspace_path)
    workspace_manager.clear_workspace()
    status_text_p.write("Done")
    

    # Save uploaded files
    status_text_p.write("Uploading reference image to workspace...")
    ref_image_path = os.path.join(workspace_manager.base_workspace, "ref_image.jpg")
    with open(ref_image_path, "wb") as f:
        f.write(uploaded_ref_image.read())

    
    status_text_p.write("Uploading target images to workspace...")
    target_image_paths = []
    for uploaded_file in uploaded_target_images:
        target_image_path = os.path.join(workspace_manager.target_repo, uploaded_file.name)
        with open(target_image_path, "wb") as f:
            f.write(uploaded_file.read())
        target_image_paths.append(target_image_path)

    # Calculate similarity scores and store in session state
    try:
        status_text_p.write("Processing reference image...")
        # Generate embeddings for reference image
        image_processor.process_and_save(ref_image_path, workspace_manager.ref_workspace)
        ref_embeddings = embedding_service.get_folder_embeddings(workspace_manager.ref_workspace, ref_image=True)
        
        if not ref_embeddings:
            st.error("Reference image has no valid faces or multiple faces. Please upload a valid reference image.")
            st.session_state.res_df = pd.DataFrame()
        else:
            # Get embeddings for all target images
            best_comparision_results = []
            progress_bar = st.progress(0)
            total_images = len(target_image_paths)

            for i, target_image_path in enumerate(target_image_paths):
                # Generate embeddings for target image
                status_text_p.text(f"Processing {i+1} of {total_images} target images: {os.path.basename(target_image_path).split("/")[-1]}")
                image_processor.process_and_save(target_image_path,workspace_manager.target_workspace)
                target_embeddings = embedding_service.get_folder_embeddings(workspace_manager.target_workspace)

                # Compare reference image embeddings with target embeddings
                comparison_results = embedding_service.compare_embeddings(target_embeddings, ref_embeddings)
                
                best_comp_result = min(comparison_results, key=lambda d: d['similarity_score'])
                best_comp_result['image_path'] = target_image_path
                best_comparision_results.append(best_comp_result)

                progress = int((i + 1) / total_images * 100)
                progress_bar.progress(progress,)

            res_df = pd.DataFrame(best_comparision_results)
            st.session_state.res_df = res_df
            progress_bar.empty()
            status_text_p.empty()

    except Exception as e:
        st.error(f"Error processing images: {str(e)}")

# Get the current similarity threshold from the user
if st.session_state.res_df is not None:
    st.write("#### Similarity Settings")
    col1, col2 = st.columns([4, 1])
    with col1:
        similarity_threshold = st.slider("Set Similarity Threshold", 0.0, 2.0, 0.4)
    with col2:
        st.button("ℹ️", help="Adjust the similarity threshold to fine-tune face matching. "
                              "Lower values make matches stricter, while higher values allow for more leniency. Suggested range 0.3 - 0.5")

    st.write(f"Current similarity threshold: {similarity_threshold}")
    

    # Filter results based on the similarity threshold
    filtered_results = st.session_state.res_df[st.session_state.res_df['similarity_score'] < similarity_threshold]
    st.write(f"Found {len(filtered_results)} matching images for the given similarity threshold.")

    # Display the results
    if len(filtered_results) > 0:

        zip_path = os.path.join(workspace_manager.base_workspace, "meinmoments_matching_images.zip")

        with zipfile.ZipFile(zip_path, "w") as zipf:
            for idx, result in filtered_results.iterrows():
                zipf.write(result['image_path'], os.path.basename(result['image_path']))
        
        # Add download button for the zip file
        with open(zip_path, "rb") as file:
            st.download_button(
                label="📥 Download All Matching Images",
                data=file,
                file_name="matching_images.zip",
                mime="application/zip"
            )
        # Loop through the filtered results and display the images and details
        for idx, result in filtered_results.iterrows():
            
            
            st.subheader(f"Match Found - Image: {result['image_path'].split("/")[-1]}")
            st.image(result['image_path'], caption=f"Similarity Score: {result['similarity_score']:.2f}")
            # st.text(f"Reference Image Type: {result['ref_image_type']} - Target Image Type: {result['target_image_type']}")

            
    else:
        st.write("No images found with similarity under the selected threshold.")
else:
    st.warning("Please upload both reference and target images to get the results.")

# Reset All button
# if st.button("🔄 Reset All"):
#     # Clear session state to reset everything
#     st.session_state.clear()  # Clears all session state
#     st.session_state.res_df = None
    # st.rerun()