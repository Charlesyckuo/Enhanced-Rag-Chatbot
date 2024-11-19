import os
import base64
from io import BytesIO
import requests
import fitz  # PyMuPDF library
from PIL import Image
from llama_index.llms.nvidia import NVIDIA

# Set environment variables for NVIDIA API key
def set_environment_variables():
    """Set necessary environment variables."""
    os.environ["NVIDIA_API_KEY"] = "YOUR_NVIDIA_API_KEY"

# Convert image content to a base64 encoded string
def get_b64_image_from_content(image_content):
    """Convert image content to base64 encoded string."""
    img = Image.open(BytesIO(image_content))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Check if an image is a graph, plot, chart, or table
def is_graph(image_content):
    """Determine if an image is a graph, plot, chart, or table."""
    description = describe_image(image_content)
    keywords = ["graph", "plot", "chart", "table"]
    return any(keyword in description.lower() for keyword in keywords)

# Process a graph image and generate a description
def process_graph(image_content):
    """Process a graph image and generate a description."""
    deplot_description = process_graph_deplot(image_content)
    nvidia_llm = NVIDIA(model_name="meta/llama-3.1-70b-instruct")
    prompt = "Your responsibility is to explain charts. You are an expert in describing the responses of linearized tables into plain English text for LLMs to use. Explain the following linearized table. "
    response = nvidia_llm.complete(prompt + deplot_description)
    return response.text

# Generate a description of an image using NVIDIA API
def describe_image(image_content):
    """Generate a description of an image using NVIDIA API."""
    image_b64 = get_b64_image_from_content(image_content)
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
    api_key = os.getenv("NVIDIA_API_KEY")
    
    if not api_key:
        raise ValueError("NVIDIA API Key is not set. Please set the NVIDIA_API_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f'Describe what you see in this image. <img src="data:image/png;base64,{image_b64}" />'
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.70,
        "seed": 0,
        "stream": False
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    return response.json()["choices"][0]['message']['content']

# Process a graph image using NVIDIA's Deplot API to generate underlying data
def process_graph_deplot(image_content):
    """Process a graph image using NVIDIA's Deplot API."""
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/google/deplot"
    image_b64 = get_b64_image_from_content(image_content)
    api_key = os.getenv("NVIDIA_API_KEY")
    
    if not api_key:
        raise ValueError("NVIDIA API Key is not set. Please set the NVIDIA_API_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f'Generate underlying data table of the figure below: <img src="data:image/png;base64,{image_b64}" />'
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.20,
        "stream": False
    }
    
    response = requests.post(invoke_url, headers=headers, json=payload)
    return response.json()["choices"][0]['message']['content']

# Extract text around an item in a page
def extract_text_around_item(text_blocks, bbox, page_height, threshold_percentage=0.1):
    """Extract text above and below a given bounding box on a page."""
    before_text, after_text = "", ""
    vertical_threshold_distance = page_height * threshold_percentage
    horizontal_threshold_distance = bbox.width * threshold_percentage

    for block in text_blocks:
        block_bbox = fitz.Rect(block[:4])
        vertical_distance = min(abs(block_bbox.y1 - bbox.y0), abs(block_bbox.y0 - bbox.y1))
        horizontal_overlap = max(0, min(block_bbox.x1, bbox.x1) - max(block_bbox.x0, bbox.x0))

        # Determine if the text block is near the given bounding box
        if vertical_distance <= vertical_threshold_distance and horizontal_overlap >= -horizontal_threshold_distance:
            if block_bbox.y1 < bbox.y0 and not before_text:
                before_text = block[4]
            elif block_bbox.y0 > bbox.y1 and not after_text:
                after_text = block[4]
                break

    return before_text, after_text

# Group text blocks based on a character count threshold
def process_text_blocks(text_blocks, char_count_threshold=500):
    """Group text blocks based on a character count threshold."""
    current_group = []
    grouped_blocks = []
    current_char_count = 0

    for block in text_blocks:
        if block[-1] == 0:  # Check if the block is of text type
            block_text = block[4]
            block_char_count = len(block_text)

            # Add blocks to group if within character limit
            if current_char_count + block_char_count <= char_count_threshold:
                current_group.append(block)
                current_char_count += block_char_count
            else:
                if current_group:
                    grouped_content = "\n".join([b[4] for b in current_group])
                    grouped_blocks.append((current_group[0], grouped_content))
                current_group = [block]
                current_char_count = block_char_count

    # Append the last group
    if current_group:
        grouped_content = "\n".join([b[4] for b in current_group])
        grouped_blocks.append((current_group[0], grouped_content))

    return grouped_blocks

# Save an uploaded file to a temporary directory
def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary directory."""
    temp_dir = os.path.join(os.getcwd(), "vectorstore", "ppt_references", "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    
    return temp_file_path
