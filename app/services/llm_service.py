import os
from typing import List, Dict
import openai
from anthropic import Anthropic
import google.generativeai as genai

# Pricing per 1M tokens
# Documentation:
# - OpenAI: https://openai.com/api/pricing/
# - Anthropic: https://www.anthropic.com/pricing#api
# - Google: https://ai.google.dev/gemini-api/docs/pricing
PRICING = {
    'gpt-4.1-mini': {
        'input': 0.40,  # $0.40 per 1M input tokens
        'output': 1.60  # $1.60 per 1M output tokens
    },
    'claude-3-5-haiku-latest': {
        'input': 0.80,  # $0.80 per 1M input tokens
        'output': 4.00  # $4.00 per 1M output tokens
    },
    'gemini-2.0-flash': {
        'input': 0.10,  # $0.10 per 1M input tokens
        'output': 0.40  # $0.40 per 1M output tokens
    }
}

class LLMService:
    def __init__(self):
        # Initialize OpenAI (new API)
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.openai_model = 'gpt-4.1-mini'
        
        # Initialize Anthropic
        self.anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.anthropic_model = 'claude-3-5-haiku-latest'
        
        # Initialize Google Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.gemini_model_name = 'gemini-2.0-flash'
        try:
            self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
        except Exception as e:
            print(f"Error initializing Gemini model '{self.gemini_model_name}': {e}")
            self.gemini_model = None

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of a response based on token usage.
        
        Prices are per 1M tokens, so we divide by 1,000,000 to get the cost.
        
        Pricing documentation:
        - OpenAI: https://openai.com/api/pricing/
        - Anthropic: https://www.anthropic.com/pricing#api
        - Google: https://ai.google.dev/gemini-api/docs/pricing
        """
        if model not in PRICING:
            return 0.0
        
        pricing = PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        return input_cost + output_cost

    def get_responses(self, question: str) -> List[Dict]:
        """Get responses from multiple LLMs for a given question."""
        responses = []
        
        # Get GPT-4 response
        try:
            gpt_response = self._get_gpt4_response(question)
            responses.append({
                'llm_name': 'OpenAI',
                'content': gpt_response['content'],
                'metadata': gpt_response['metadata']
            })
        except Exception as e:
            responses.append({
                'llm_name': 'OpenAI',
                'content': f"Error: {str(e)}",
                'metadata': {
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                    'cost': 0.0,
                    'model': self.openai_model,
                    'finish_reason': 'error'
                }
            })

        # Get Claude response
        try:
            claude_response = self._get_claude_response(question)
            responses.append({
                'llm_name': 'Claude',
                'content': claude_response['content'],
                'metadata': claude_response['metadata']
            })
        except Exception as e:
            responses.append({
                'llm_name': 'Claude',
                'content': f"Error: {str(e)}",
                'metadata': {
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                    'cost': 0.0,
                    'model': self.anthropic_model,
                    'stop_reason': 'error'
                }
            })

        # Get Gemini response
        try:
            gemini_response = self._get_gemini_response(question)
            responses.append({
                'llm_name': 'Gemini',
                'content': gemini_response['content'],
                'metadata': gemini_response['metadata']
            })
        except Exception as e:
            # Try to list available models for debugging
            try:
                available_models = genai.list_models()
                model_names = [m.name for m in available_models]
                error_msg = f"Error: {str(e)}\nAvailable Gemini models: {model_names}"
            except Exception as e2:
                error_msg = f"Error: {str(e)}\nAlso failed to list Gemini models: {str(e2)}"
            responses.append({
                'llm_name': 'Gemini',
                'content': error_msg,
                'metadata': {
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                    'cost': 0.0,
                    'model': self.gemini_model_name,
                    'safety_ratings': None
                }
            })

        return responses

    def _get_gpt4_response(self, question: str) -> Dict:
        """Get response from GPT-4.1 mini using OpenAI's new API."""
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": question}
            ]
        )
        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        return {
            'content': response.choices[0].message.content,
            'metadata': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'cost': self._calculate_cost(self.openai_model, input_tokens, output_tokens),
                'model': self.openai_model,
                'finish_reason': response.choices[0].finish_reason
            }
        }

    def _get_claude_response(self, question: str) -> Dict:
        """Get response from Claude."""
        response = self.anthropic.messages.create(
            model=self.anthropic_model,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        
        return {
            'content': response.content[0].text,
            'metadata': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'cost': self._calculate_cost(self.anthropic_model, input_tokens, output_tokens),
                'model': self.anthropic_model,
                'stop_reason': response.stop_reason
            }
        }

    def _get_gemini_response(self, question: str) -> Dict:
        """Get response from Google Gemini."""
        if not self.gemini_model:
            raise Exception(f"Gemini model '{self.gemini_model_name}' is not available.")
        
        response = self.gemini_model.generate_content(question)
        token_count = getattr(response, 'token_count', 0)
        
        # For Gemini, we'll estimate input/output tokens as 50/50 split
        input_tokens = token_count // 2
        output_tokens = token_count - input_tokens
        
        return {
            'content': response.text,
            'metadata': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': token_count,
                'cost': self._calculate_cost(self.gemini_model_name, input_tokens, output_tokens),
                'model': self.gemini_model_name,
                'safety_ratings': getattr(response, 'safety_ratings', None)
            }
        } 