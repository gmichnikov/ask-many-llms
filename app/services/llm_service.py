import os
from typing import List, Dict
import openai
from anthropic import Anthropic
import google.generativeai as genai

class LLMService:
    def __init__(self):
        # Initialize OpenAI (new API)
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.openai_model = 'gpt-4.1-mini'  # Updated model name
        
        # Initialize Anthropic
        self.anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.anthropic_model = 'claude-3-5-haiku-latest'  # Updated model name
        
        # Initialize Google Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.gemini_model_name = 'gemini-2.0-flash'  # Updated model name
        try:
            self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
        except Exception as e:
            print(f"Error initializing Gemini model '{self.gemini_model_name}': {e}")
            self.gemini_model = None

    def get_responses(self, question: str) -> List[Dict]:
        """Get responses from multiple LLMs for a given question."""
        responses = []
        
        # Get GPT-4 response
        try:
            gpt_response = self._get_gpt4_response(question)
            responses.append({
                'llm_name': 'GPT-4',
                'content': gpt_response['content'],
                'tokens_used': gpt_response['tokens_used']
            })
        except Exception as e:
            responses.append({
                'llm_name': 'GPT-4',
                'content': f"Error: {str(e)}",
                'tokens_used': 0
            })

        # Get Claude response
        try:
            claude_response = self._get_claude_response(question)
            responses.append({
                'llm_name': 'Claude',
                'content': claude_response['content'],
                'tokens_used': claude_response['tokens_used']
            })
        except Exception as e:
            responses.append({
                'llm_name': 'Claude',
                'content': f"Error: {str(e)}",
                'tokens_used': 0
            })

        # Get Gemini response
        try:
            gemini_response = self._get_gemini_response(question)
            responses.append({
                'llm_name': 'Gemini',
                'content': gemini_response['content'],
                'tokens_used': gemini_response['tokens_used']
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
                'tokens_used': 0
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
        return {
            'content': response.choices[0].message.content,
            'tokens_used': response.usage.total_tokens
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
        return {
            'content': response.content[0].text,
            'tokens_used': response.usage.input_tokens + response.usage.output_tokens
        }

    def _get_gemini_response(self, question: str) -> Dict:
        """Get response from Google Gemini."""
        if not self.gemini_model:
            raise Exception(f"Gemini model '{self.gemini_model_name}' is not available.")
        response = self.gemini_model.generate_content(question)
        return {
            'content': response.text,
            'tokens_used': getattr(response, 'token_count', 0)
        } 