import anthropic

class AnthropicClient:
    def __init__(self, model="claude-3-sonnet-20240229", api_key=None):
        """
        Initialize the Anthropic client.
        
        Args:
            model (str): The model to use for generation
            api_key (str): The Anthropic API key
        """
        self.model = model
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = anthropic.Anthropic()  # Assumes ANTHROPIC_API_KEY environment variable

    def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=500):
        """
        Generate a response using Anthropic.
        
        Args:
            prompt (str): The prompt to send to the model
            system_prompt (str, optional): System prompt for context
            temperature (float): Controls randomness (0.0 to 1.0)
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: The generated response
        """
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Anthropic uses system parameter separately, not in messages
            if system_prompt:
                response = self.client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            return response.content[0].text
        except Exception as e:
            return f"Error communicating with Anthropic: {str(e)}"
