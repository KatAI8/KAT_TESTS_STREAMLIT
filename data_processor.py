import pandas as pd
import numpy as np
import faiss
import os
import pickle
import hashlib
from sentence_transformers import SentenceTransformer

class ProductDataProcessor:
    def __init__(self, csv_path=None):
        """Initialize the data processor with the path to the product CSV file."""
        self.csv_path = csv_path
        self.df = None
        self.embeddings = None
        self.index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize cache variables
        self.cache_dir = None
        self.embeddings_cache_file = None
        self.index_cache_file = None
        
        # Set up cache if csv_path is provided
        if csv_path:
            # Create cache directory if it doesn't exist
            self.cache_dir = os.path.join(os.path.dirname(csv_path), 'cache')
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Generate cache filenames based on CSV file hash
            self.embeddings_cache_file = self._get_cache_filename('embeddings')
            self.index_cache_file = self._get_cache_filename('index')
    
    def _get_cache_filename(self, prefix):
        """Generate a cache filename based on the CSV file's content hash"""
        # Check if csv_path is available
        if not self.csv_path or not os.path.exists(self.csv_path):
            return None
            
        # Compute a hash of the CSV file to use in the cache filename
        # This ensures that if the CSV changes, we'll regenerate the embeddings
        hasher = hashlib.md5()
        with open(self.csv_path, 'rb') as f:
            buf = f.read(65536)  # Read in 64kb chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        file_hash = hasher.hexdigest()[:10]  # Use first 10 chars of hash
        
        # Ensure cache_dir exists
        if not self.cache_dir:
            self.cache_dir = os.path.join(os.path.dirname(self.csv_path), 'cache')
            os.makedirs(self.cache_dir, exist_ok=True)
            
        return os.path.join(self.cache_dir, f"{prefix}_{file_hash}.pkl")
        
    def load_data(self, data_source=None):
        """Load the product data from a CSV file path or DataFrame.
        
        Args:
            data_source: Either a path to the CSV file (str) or a pandas DataFrame.
        """
        if data_source is None:
            if self.csv_path:
                data_source = self.csv_path
            else:
                raise ValueError("No data source provided. Please provide a CSV path or DataFrame.")
        
        # Setup cache if it's a file path
        if isinstance(data_source, str):
            self.csv_path = data_source
            self.cache_dir = os.path.join(os.path.dirname(data_source), 'cache')
            os.makedirs(self.cache_dir, exist_ok=True)
            self.embeddings_cache_file = self._get_cache_filename('embeddings')
            self.index_cache_file = self._get_cache_filename('index')
            
            # Load from file
            self.df = pd.read_csv(self.csv_path)
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source
            self.csv_path = None
            self.embeddings_cache_file = None
            self.index_cache_file = None
        else:
            raise ValueError("data_source must be a string (file path) or pandas DataFrame.")
        
        return self.df
    
    def create_embeddings(self):
        """Create embeddings for the product descriptions or load from cache if available."""
        if self.df is None:
            self.load_data()
            
        # Try to load embeddings from cache first
        if self.embeddings_cache_file and os.path.exists(self.embeddings_cache_file):
            try:
                print(f"Loading embeddings from cache: {self.embeddings_cache_file}")
                with open(self.embeddings_cache_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                print(f"Loaded {len(self.embeddings)} embeddings from cache")
                return self.embeddings
            except Exception as e:
                print(f"Error loading embeddings from cache: {e}")
                # If loading fails, we'll generate them again
        
        print("Generating new embeddings...")
        # Combine relevant product information for embedding
        texts = []
        for _, row in self.df.iterrows():
            # Get values with fallbacks to empty strings if columns don't exist
            name = str(row.get('name', ''))
            category = str(row.get('category', ''))
            description = str(row.get('description', ''))
            price = str(row.get('price', ''))
            color = str(row.get('color', ''))
            size = str(row.get('size', ''))
            
            # Extract more details from description if it's in JSON format
            desc_details = ''
            if description and description.startswith('[{'):
                try:
                    import json
                    desc_json = json.loads(description.replace("'", '"'))
                    for item in desc_json:
                        for key, value in item.items():
                            desc_details += f" {key}: {value}"
                except:
                    desc_details = description
            else:
                desc_details = description
                
            # Create comprehensive text representation with double weight on name and category
            text = f"{name} {name} - {category} {category} - {desc_details} - Price: ${price} - Color: {color} - Size: {size}"
            texts.append(text)
        
        # Generate embeddings
        print(f"Encoding {len(texts)} product descriptions...")
        self.embeddings = self.model.encode(texts)
        
        # Save embeddings to cache
        if self.embeddings_cache_file:
            try:
                print(f"Saving embeddings to cache: {self.embeddings_cache_file}")
                with open(self.embeddings_cache_file, 'wb') as f:
                    pickle.dump(self.embeddings, f)
            except Exception as e:
                print(f"Error saving embeddings to cache: {e}")
                
        return self.embeddings
    
    def build_faiss_index(self):
        """Build a FAISS index for fast similarity search or load from cache if available."""
        if self.embeddings is None:
            self.create_embeddings()
            
        # Try to load index from cache first
        if self.index_cache_file and os.path.exists(self.index_cache_file):
            try:
                print(f"Loading FAISS index from cache: {self.index_cache_file}")
                with open(self.index_cache_file, 'rb') as f:
                    self.index = pickle.load(f)
                print("FAISS index loaded from cache")
                return self.index
            except Exception as e:
                print(f"Error loading FAISS index from cache: {e}")
                # If loading fails, we'll generate it again
        
        print("Building new FAISS index...")
        # Create a FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype(np.float32))
        
        # Save index to cache
        if self.index_cache_file:
            try:
                print(f"Saving FAISS index to cache: {self.index_cache_file}")
                with open(self.index_cache_file, 'wb') as f:
                    pickle.dump(self.index, f)
            except Exception as e:
                print(f"Error saving FAISS index to cache: {e}")
                
        return self.index
    
    def search_products(self, query, k=5, relevance_threshold=0.6):
        """
        Search for products similar to the query with improved relevance filtering.
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            relevance_threshold (float): Minimum similarity score (0-1) for a product to be included
            
        Returns:
            list: List of dictionaries containing product information
        """
        if self.index is None:
            self.build_faiss_index()
        
        # Extract key terms from the query
        query_terms = self._extract_key_terms(query)
        enhanced_query = ' '.join(query_terms)
        
        # Add price filtering if price mentioned in query
        price_filter = None
        for term in query_terms:
            if '$' in term or 'under' in term.lower() or 'below' in term.lower() or 'less than' in term.lower():
                try:
                    # Extract price value from terms like "$100", "under $100", etc.
                    import re
                    price_matches = re.findall(r'\$?(\d+)', query)
                    if price_matches:
                        price_filter = float(price_matches[0])
                except:
                    pass
        
        # Encode the query
        query_vector = self.model.encode([enhanced_query]).astype(np.float32)
        
        # Search the index - get more results initially to filter later
        max_results = min(k * 5, len(self.df))
        distances, indices = self.index.search(query_vector, max_results)
        
        # Convert distances to similarity scores (higher is better)
        # FAISS L2 distance: smaller is better, so we invert it
        max_distance = np.max(distances) + 1e-6  # Avoid division by zero
        similarities = 1 - (distances / max_distance)
        
        # Get the product information with similarity scores
        results = []
        for i, idx in enumerate(indices[0]):
            similarity = similarities[0][i]
            
            # Skip products below the relevance threshold
            if similarity < relevance_threshold:
                continue
                
            product = self.df.iloc[idx].to_dict()
            
            # Apply price filter if specified
            if price_filter is not None:
                try:
                    product_price = float(product['price'])
                    if product_price > price_filter:
                        continue  # Skip products above the price filter
                except:
                    pass
            
            # Add relevance score to the product
            product['relevance_score'] = float(similarity)
            
            # Check if product matches any key terms in the query
            name = str(product.get('name', '')).lower()
            category = str(product.get('category', '')).lower()
            description = str(product.get('description', '')).lower()
            color = str(product.get('color', '')).lower()
            size = str(product.get('size', '')).lower()
            
            product_text = f"{name} {category} {description} {color} {size}"
            
            # Count term matches with weighted scoring
            term_matches = 0
            exact_category_match = False
            exact_name_match = False
            
            for term in query_terms:
                term = term.lower()
                # Check for exact matches in important fields
                if term in name:
                    term_matches += 2  # Double weight for name matches
                    if term == name or f"{term}s" == name or term == f"{name}s":
                        exact_name_match = True
                        term_matches += 3  # Bonus for exact name match
                        
                if term in category:
                    term_matches += 2  # Double weight for category matches
                    if term == category or f"{term}s" == category or term == f"{category}s":
                        exact_category_match = True
                        term_matches += 3  # Bonus for exact category match
                
                # Regular matches in other fields
                if term in description:
                    term_matches += 1
                if term in color:
                    term_matches += 1
                if term in size:
                    term_matches += 1
                    
                # Special handling for compound terms like "running shoes"
                if ' ' in term:
                    if term in product_text:
                        term_matches += 4  # Extra weight for multi-word matches
                        
            # Store the match information
            product['term_matches'] = term_matches
            product['exact_category_match'] = exact_category_match
            product['exact_name_match'] = exact_name_match
            
            # Add image URL from the images field
            if 'images' in product and product['images']:
                try:
                    # Handle JSON array format
                    if isinstance(product['images'], str) and (product['images'].startswith('[') or product['images'].startswith('"http')):
                        import json
                        # Replace single quotes with double quotes for proper JSON parsing
                        images_str = product['images'].replace("'", '"')
                        # Handle both array and single URL formats
                        if images_str.startswith('['):
                            images = json.loads(images_str)
                            if images and len(images) > 0:
                                product['image_url'] = images[0]  # Use the first image
                        else:
                            # It might be a direct URL string
                            product['image_url'] = images_str.strip('"')
                    else:
                        product['image_url'] = product['images']
                except Exception as e:
                    print(f"Error processing image URL: {e}")
                    product['image_url'] = ''
            else:
                product['image_url'] = ''
            
            results.append(product)
        
        # Sort by combination of exact matches, term matches, and similarity score
        # This prioritizes products that exactly match compound terms like "running shoes"
        results.sort(key=lambda x: (
            x.get('exact_category_match', False),  # First by exact category match
            x.get('exact_name_match', False),      # Then by exact name match
            x['term_matches'],                     # Then by term match count
            x['relevance_score']                   # Finally by relevance score
        ), reverse=True)
        
        # Print some debug info about the top results
        if results:
            print(f"\nTop result for '{query}':\n")
            print(f"Name: {results[0].get('name', '')}")
            print(f"Category: {results[0].get('category', '')}")
            print(f"Term matches: {results[0].get('term_matches', 0)}")
            print(f"Exact category match: {results[0].get('exact_category_match', False)}")
            print(f"Exact name match: {results[0].get('exact_name_match', False)}")
            print(f"Relevance score: {results[0].get('relevance_score', 0):.2f}\n")
        
        # Return only the top k results
        return results[:k]
        
    def _extract_key_terms(self, query):
        """
        Extract key terms from the query to improve search relevance.
        
        Args:
            query (str): The user query
            
        Returns:
            list: List of key terms
        """
        # Remove common stop words
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                     'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                     'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                     'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                     'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                     'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                     'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                     'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'im', 'looking'}
        
        # Split the query into words and filter out stop words
        words = query.lower().split()
        key_terms = [word for word in words if word not in stop_words]
        
        # Add the original query as well to maintain context
        if key_terms:
            key_terms.append(query)
        else:
            key_terms = [query]
        
        # Add compound terms to improve context understanding
        # For example, if query contains "running shoes", add it as a single term
        # This helps distinguish between "running shoes" (footwear) and "running" (activity) + "shoes"
        for i in range(len(words) - 1):
            compound_term = words[i] + ' ' + words[i + 1]
            if compound_term not in key_terms:
                key_terms.append(compound_term)
        
        # Add specific product type mappings
        product_type_mappings = {
            'running shoes': ['shoes', 'footwear', 'sneakers', 'trainers', 'athletic shoes', 'casual sneakers'],
            'sneakers': ['shoes', 'footwear', 'sneakers', 'trainers', 'athletic shoes', 'casual sneakers'],
            'dress': ['dresses', 'gown', 'frock'],
            'jeans': ['denim', 'pants', 'trousers'],
            'shirt': ['top', 'tee', 't-shirt', 'blouse'],
            'jacket': ['coat', 'outerwear', 'blazer']
        }
        
        # Check if any mappings apply to our query
        expanded_terms = list(key_terms)  # Create a copy to avoid modifying during iteration
        for term in key_terms:
            for product_type, synonyms in product_type_mappings.items():
                if term in product_type or product_type in term:
                    expanded_terms.extend(synonyms)
                    # Also add the full product type if it's a partial match
                    if term in product_type and product_type not in expanded_terms:
                        expanded_terms.append(product_type)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
                
        return unique_terms
