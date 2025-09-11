import openai

class OpenAIClient:
    def __init__(self, model="gpt-3.5-turbo", api_key=None):
        """
        Initialize the OpenAI client.
        
        Args:
            model (str): The model to use for generation
            api_key (str): The OpenAI API key
        """
        self.model = model
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = openai.OpenAI()  # Assumes OPENAI_API_KEY environment variable

    def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=500):
        """
        Generate a response using OpenAI.
        
        Args:
            prompt (str): The prompt to send to the model
            system_prompt (str, optional): System prompt for context
            temperature (float): Controls randomness (0.0 to 1.0)
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: The generated response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error communicating with OpenAI: {str(e)}"