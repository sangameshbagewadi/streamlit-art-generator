import streamlit as st
import requests
import io
from PIL import Image, ImageEnhance, ImageFilter
import time
import os  # Import os module for accessing environment variables

# Hugging Face API settings
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
API_KEY = os.getenv("API_KEY")  # Ensure correct environment variable name

# Check if API key exists
if not API_KEY:
    st.error("⚠️ API key is missing! Set the 'API_KEY' environment variable.")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Caching function to avoid redundant requests
@st.cache_resource
def generate_image_from_api(prompt, resolution="Medium"):
    """Generate an AI image from a text prompt using Hugging Face API with error handling."""
    max_retries = 3  # Reduced retries for smoother experience
    retry_delay = 15  # Reduced retry delay to speed up

    # Define different resolution sizes
    resolutions = {"Small": (256, 256), "Medium": (512, 512), "Large": (1024, 1024)}
    width, height = resolutions.get(resolution, (512, 512))  # Default to Medium

    for _ in range(max_retries):
        response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})

        # Handle JSON response errors
        if response.headers.get("Content-Type", "").startswith("application/json"):
            data = response.json()

            # If the model is still loading, retry after waiting
            if "error" in data and "loading" in data["error"]:
                estimated_time = data.get("estimated_time", 0)
                st.write(
                    f"⏳ Model is loading. Estimated time: {estimated_time} seconds. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
                continue  # Retry the request

            st.error(f"❌ API Error: {data.get('error', 'Unknown error')}")
            return None

        # Handle successful image response
        try:
            return response.content  # Return the image bytes directly
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            return None

    # If maximum retries reached
    st.error("⚠️ Model took too long to load. Please try again later.")
    return None

# Streamlit UI setup
st.title("🎨 AI-Generated Art Gallery")
st.markdown("Enter a creative prompt, and let AI generate stunning images!")

# Initialize session state for prompt
if "prompt" not in st.session_state:
    st.session_state["prompt"] = ""  # Initialize if not already set

# User input for prompt
prompt = st.text_input("Enter your prompt:", value=st.session_state["prompt"])

# Dropdown for selecting resolution
resolution = st.selectbox("Choose image resolution:", ["Small", "Medium", "Large"])

# Dropdown for applying filters to the generated image
filter_option = st.selectbox("Apply filter to image:", ["None", "Blur", "Sharpen", "Grayscale"])

# Reset button to clear input and generated image
if st.button("Reset"):
    st.session_state["prompt"] = ""  # Clear prompt
    st.session_state.pop("image", None)  # Remove generated image if it exists

# Generate button functionality
if st.button("Generate Art"):
    if not prompt.strip():  # Check if the prompt is empty
        st.error("⚠️ Please enter a prompt to generate an image.")
    else:
        with st.spinner("Generating your image... please wait ⏳"):
            start_time = time.time()  # Start timer for performance tracking
            image_bytes = generate_image_from_api(prompt, resolution)

        if image_bytes:
            try:
                # Convert image bytes to an actual image
                image = Image.open(io.BytesIO(image_bytes))

                # Apply selected filter
                if filter_option == "Blur":
                    image = image.filter(ImageFilter.BLUR)
                elif filter_option == "Sharpen":
                    enhancer = ImageEnhance.Sharpness(image)
                    image = enhancer.enhance(2.0)  # Increase sharpness level
                elif filter_option == "Grayscale":
                    image = image.convert("L")  # Convert to grayscale

                # Display the generated image
                st.image(image, caption="Generated Art", use_container_width=True)

                # Display time taken for image generation
                elapsed_time = time.time() - start_time
                st.write(f"🕒 Image generated in {elapsed_time:.2f} seconds")

                # Download button for saving the image
                st.download_button(
                    label="📥 Download Image",
                    data=image_bytes,
                    file_name="generated_art.png",
                    mime="image/png",
                )

            except Exception as e:
                st.error(f"❌ Failed to display image: {e}")
        else:
            st.error("⚠️ Failed to generate image. Try again!")
