import streamlit as st
import os
import pandas as pd
import json
import io
from shopping_assistant import ShoppingAssistant
from openai_client import OpenAIClient
from data_processor import ProductDataProcessor

# Set page configuration
st.set_page_config(
    page_title="AI Shopping Assistant",
    page_icon="🛒",
    layout="wide"
)

# CSS for product cards
st.markdown("""
<style>
.product-card {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: transform 0.3s;
}
.product-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
.product-image {
    width: 100%;
    height: 200px;
    object-fit: cover;
    border-radius: 5px;
    margin-bottom: 10px;
}
.product-title {
    font-weight: bold;
    font-size: 1.2em;
    margin-bottom: 5px;
}
.product-price {
    color: #e63946;
    font-weight: bold;
    font-size: 1.1em;
    margin-bottom: 5px;
}
.product-rating {
    color: #fca311;
    margin-bottom: 5px;
}
.product-category {
    background-color: #e9ecef;
    padding: 3px 8px;
    border-radius: 10px;
    font-size: 0.8em;
    display: inline-block;
    margin-bottom: 10px;
}
.product-stock {
    color: #2a9d8f;
    font-size: 0.9em;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'products' not in st.session_state:
    st.session_state.products = []

# App title and description
st.title("🛒 AI Shopping Assistant")
st.markdown("""
This AI shopping assistant helps you find products based on your needs.
Simply describe what you're looking for, and the assistant will recommend products from our catalog.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# OpenAI API key
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# OpenAI model selection
openai_model = st.sidebar.selectbox(
    "Select OpenAI Model",
    ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
    index=0
)

# Number of recommendations
num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

# Dataset upload
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")

# Load product data and initialize/reload assistant when config changes
if api_key and uploaded_file:
    # Read uploaded file
    df = pd.read_csv(uploaded_file)
    
    # Check for changes in model or file
    file_key = uploaded_file.name + str(uploaded_file.size)  # Simple key to detect file change
    prev_file_key = st.session_state.get('file_key', '')
    prev_model = st.session_state.get('model', '')
    needs_init = ('assistant' not in st.session_state) or (prev_file_key != file_key) or (prev_model != openai_model)
    
    if needs_init:
        try:
            # Initialize data processor with uploaded data
            data_processor = ProductDataProcessor()
            data_processor.load_data(df)
            data_processor.build_faiss_index()

            # Initialize OpenAI client
            openai_client = OpenAIClient(model=openai_model, api_key=api_key)

            # Initialize shopping assistant
            assistant = ShoppingAssistant(data_processor, openai_client, api_key=api_key)
            st.session_state.assistant = assistant
            st.session_state.file_key = file_key
            st.session_state.model = openai_model
            st.session_state.df = df  # Store for reference
            st.sidebar.success(f"Loaded dataset: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error loading Shopping Assistant: {str(e)}")
            st.stop()
    else:
        # Reuse existing assistant
        assistant = st.session_state.assistant
        df = st.session_state.df
else:
    if not api_key:
        st.sidebar.warning("Please enter your OpenAI API key.")
    if not uploaded_file:
        st.sidebar.warning("Please upload a CSV dataset.")
    st.stop()

# Function to display product cards
def display_product_cards(products):
    # Display a message if no products are available
    if not products:
        st.info("No products found for your query. Try a different search!")
        return
        
    # Sort products by relevance score if available
    if 'relevance_score' in products[0]:
        sorted_products = sorted(products, key=lambda x: x.get('relevance_score', 0), reverse=True)
    else:
        sorted_products = products
    
    # Display products in columns
    cols = st.columns(3)  # Display 3 products per row
    for i, product in enumerate(sorted_products):
        with cols[i % 3]:
            # Get relevance score if available
            relevance = product.get('relevance_score', 0)
            relevance_display = f" - Relevance: {relevance:.2f}" if relevance > 0 else ""
            
            # Extract description details if in JSON format
            description_display = product.get('description', '')
            if isinstance(description_display, str) and description_display.startswith('[{'):
                try:
                    import json
                    desc_json = json.loads(description_display.replace("'", '"'))
                    desc_text = ""
                    # Extract product details from JSON
                    for item in desc_json:
                        if 'Product Details' in item:
                            desc_text = item['Product Details']
                            break
                    if not desc_text and len(desc_json) > 0:
                        # Use first item if no product details found
                        for key, value in desc_json[0].items():
                            desc_text = value
                            break
                    description_display = desc_text
                except:
                    pass
            
            # Truncate description if too long
            if len(str(description_display)) > 150:
                description_display = str(description_display)[:150] + "..."
                
            # Get size information
            size_info = product.get('size', '')
            
            # Extract image URL from 'images' field if 'image_url' is not available
            image_url = product.get('image_url', '')
            if not image_url and 'images' in product and product['images']:
                try:
                    # Handle different formats of the images field
                    images_data = product['images']
                    
                    # Special handling for the complex nested format
                    if isinstance(images_data, str):
                        # Clean up the string to extract the actual URL
                        # First, check if it's in the format ["['url']"] or similar
                        if images_data.startswith('["[') or images_data.startswith('[\"['):
                            # Extract the URL directly using string manipulation
                            # Find the first http and extract until the end or a quote
                            if 'http' in images_data:
                                start_idx = images_data.find('http')
                                end_idx = images_data.find("'", start_idx)
                                if end_idx == -1:  # If no single quote, try double quote
                                    end_idx = images_data.find('"', start_idx)
                                if end_idx == -1:  # If still no quote, use the rest of the string
                                    image_url = images_data[start_idx:].strip("']\"")
                                else:
                                    image_url = images_data[start_idx:end_idx]
                        # Try to parse as JSON if it's a simpler format
                        elif images_data.startswith('['):
                            try:
                                import json
                                # Replace single quotes with double quotes for proper JSON parsing
                                cleaned_data = images_data.replace("'", '"')
                                images_list = json.loads(cleaned_data)
                                if images_list and len(images_list) > 0:
                                    # If the first item is a string, use it directly
                                    if isinstance(images_list[0], str):
                                        image_url = images_list[0]
                                    # If it's another list, get the first item from it
                                    elif isinstance(images_list[0], list) and len(images_list[0]) > 0:
                                        image_url = images_list[0][0]
                            except json.JSONDecodeError:
                                # If JSON parsing fails, try direct string extraction
                                if 'http' in images_data:
                                    start_idx = images_data.find('http')
                                    end_idx = images_data.find("'", start_idx)
                                    if end_idx == -1:  # If no single quote, try double quote
                                        end_idx = images_data.find('"', start_idx)
                                    if end_idx == -1:  # If still no quote, use the rest of the string
                                        image_url = images_data[start_idx:].strip("']\"")
                                    else:
                                        image_url = images_data[start_idx:end_idx]
                        # If it's a direct URL
                        elif images_data.startswith('http'):
                            image_url = images_data
                    # If it's already a list
                    elif isinstance(images_data, list) and len(images_data) > 0:
                        if isinstance(images_data[0], str) and images_data[0].startswith('http'):
                            image_url = images_data[0]
                        elif isinstance(images_data[0], list) and len(images_data[0]) > 0:
                            image_url = images_data[0][0]
                except Exception as e:
                    # Don't show the error in the UI, just print to console
                    print(f"Error processing image URL: {str(e)}")
            
            st.markdown(f"""
            <div class="product-card">
                <img src="{image_url}" class="product-image">
                <div class="product-title">{product.get('name', '')}</div>
                <div class="product-price">${product.get('price', '')}</div>
                <div class="product-category">{product.get('category', '')}</div>
                <div class="product-color">Color: {product.get('color', '')}</div>
                <div class="product-size">Size: {size_info}</div>
                <div class="product-stock">SKU: {product.get('SKU', '')}</div>
                <div>{description_display}</div>
            </div>
            """, unsafe_allow_html=True)

# Display chat messages and products
st.subheader("Chat with the Shopping Assistant")

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Display current product recommendations if any
if st.session_state.products:
    st.subheader("Recommended Products")
    display_product_cards(st.session_state.products)

# Chat input
if prompt := st.chat_input("What are you looking for today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get product recommendations and LLM response
                products, response = st.session_state.assistant.get_product_recommendations(prompt, num_products=num_recommendations)
                
                # Display LLM response
                st.write(response)
                
                # Store products in session state
                st.session_state.products = products
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display product recommendations
                st.subheader("Recommended Products")
                display_product_cards(products)
                
            except Exception as e:
                error_msg = f"Error generating recommendations: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Instructions in the sidebar
st.sidebar.markdown("""
## How to use
1. Enter your OpenAI API key
2. Upload a CSV file with product data
3. Select your preferred model from the dropdown
4. Ask the assistant about products you're interested in
5. The assistant will recommend products based on your query

## Example queries:
- "I need a new pair of running shoes"
- "What's a good water bottle for the gym?"
- "I'm looking for electronics under $50"
""")
