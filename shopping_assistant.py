from data_processor import ProductDataProcessor
from openai_client import OpenAIClient
import os

class ShoppingAssistant:
    def __init__(self, data_processor_or_csv_path, openai_client_or_model="gpt-3.5-turbo", api_key=None):
        """
        Initialize the shopping assistant.
        
        Args:
            data_processor_or_csv_path: Either a ProductDataProcessor instance or a path to the product CSV file
            openai_client_or_model: Either an OpenAIClient instance or the name of the OpenAI model to use
            api_key: OpenAI API key
        """
        # Handle data processor
        if isinstance(data_processor_or_csv_path, ProductDataProcessor):
            self.data_processor = data_processor_or_csv_path
        else:
            # Assume it's a CSV path
            self.data_processor = ProductDataProcessor()
            self.data_processor.load_data(data_processor_or_csv_path)
            self.data_processor.build_faiss_index()
        
        # Handle OpenAI client
        if isinstance(openai_client_or_model, OpenAIClient):
            self.llm_client = openai_client_or_model
        else:
            # Assume it's a model name
            self.llm_client = OpenAIClient(model=openai_client_or_model, api_key=api_key)
        
    def get_product_recommendations(self, user_query, num_products=5):
        """
        Get product recommendations based on the user query.
        
        Args:
            user_query (str): The user's query or request
            num_products (int): Number of products to recommend
            
        Returns:
            tuple: (list of product dictionaries, LLM response text)
        """
        # Start with a high relevance threshold
        initial_threshold = 0.65
        products = self.data_processor.search_products(user_query, k=num_products, relevance_threshold=initial_threshold)
        
        # If we don't have enough products, gradually lower the threshold until we get at least 5
        thresholds = [0.6, 0.55, 0.5, 0.45, 0.4, 0.3, 0.2, 0.0]
        
        for threshold in thresholds:
            if len(products) >= num_products:
                break
            
            # Get more products with a lower threshold
            more_products = self.data_processor.search_products(user_query, k=num_products*2, relevance_threshold=threshold)
            
            # Add new products that aren't already in our list
            # Use SKU or URL as unique identifier
            existing_ids = set()
            for p in products:
                if 'SKU' in p:
                    existing_ids.add(p['SKU'])
                elif 'url' in p:
                    existing_ids.add(p['url'])
                
            for product in more_products:
                product_id = product.get('SKU', product.get('url', ''))
                if product_id and product_id not in existing_ids:
                    products.append(product)
                    existing_ids.add(product_id)
                    
                    # Stop if we have enough products
                    if len(products) >= num_products:
                        break
        
        # If we still don't have enough products, just get the top N products without filtering
        if len(products) < num_products:
            all_products = self.data_processor.search_products(user_query, k=num_products, relevance_threshold=0)
            
            # Add any new products
            existing_ids = set()
            for p in products:
                if 'SKU' in p:
                    existing_ids.add(p['SKU'])
                elif 'url' in p:
                    existing_ids.add(p['url'])
                    
            # We already set up existing_ids above, so we can reuse it
            for product in all_products:
                product_id = product.get('SKU', product.get('url', ''))
                if product_id and product_id not in existing_ids:
                    products.append(product)
                    existing_ids.add(product_id)
                    
        # Sort products by relevance score
        products.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Take only the top num_products
        products = products[:num_products]
        
        # Format the product information for the LLM
        product_info = ""
        for i, product in enumerate(products, 1):
            product_info += f"Product {i}:\n"
            product_info += f"Name: {product.get('name', '')}\n"
            product_info += f"Category: {product.get('category', '')}\n"
            product_info += f"Price: ${product.get('price', '')}\n"
            product_info += f"Color: {product.get('color', '')}\n"
            product_info += f"Size: {product.get('size', '')}\n"
            
            # Format description - handle JSON if needed
            description = product.get('description', '')
            if isinstance(description, str) and description.startswith('[{'):
                try:
                    import json
                    desc_json = json.loads(description.replace("'", '"'))
                    desc_text = ""
                    for item in desc_json:
                        for key, value in item.items():
                            desc_text += f"{key}: {value}\n"
                    product_info += f"Description: {desc_text}\n"
                except:
                    product_info += f"Description: {description}\n"
            else:
                product_info += f"Description: {description}\n"
                
            product_info += f"SKU: {product.get('SKU', '')}\n"  # Note: Using 'SKU' with uppercase as per your column names
            product_info += f"Relevance Score: {product.get('relevance_score', 0):.2f}\n\n"
        
        # Create the prompt for the LLM
        system_prompt = """
        You are a friendly, conversational shopping assistant who speaks like a helpful friend.
        Your goal is to recommend ONLY the products that are truly relevant to what the user is looking for.
        
        Important guidelines:
        1. Be warm, friendly, and conversational - like you're chatting with a friend
        2. ONLY recommend products that are directly relevant to the user's query
        3. If none of the products are relevant, be honest and say so
        4. Explain WHY each recommended product would be good for the user's specific needs
        5. Keep your response concise and focused
        6. Use a casual, friendly tone - use contractions, simple language, and be personable
        7. Don't list all products if some aren't relevant - only mention the truly relevant ones
        """
        
        user_prompt = f"""
        The user said: \"{user_query}\"
        
        Here are some potential product matches (with relevance scores):
        
        {product_info}
        
        Please respond in a friendly, conversational way. ONLY recommend products that are truly relevant to what they're looking for.
        If some products aren't relevant, don't mention them at all. If none are relevant, be honest about it.
        The product details will be displayed separately, so focus on being helpful and explaining why these specific products would be good for them.
        """
        
        # Generate a response using OpenAI
        response = self.llm_client.generate(user_prompt, system_prompt=system_prompt, temperature=0.7)
        return products, response
