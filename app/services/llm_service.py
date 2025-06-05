import os
from typing import List, Dict
import openai
from anthropic import Anthropic
import google.generativeai as genai

class LLMService:
    def __init__(self):
        # Initialize OpenAI
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Initialize Anthropic
        self.anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Initialize Google Gemini
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-pro')

    async def get_responses(self, question: str) -> List[Dict]:
        """Get responses from multiple LLMs for a given question."""
        responses = []
        
        # Get GPT-4 response
        try:
            gpt_response = await self._get_gpt4_response(question)
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
            claude_response = await self._get_claude_response(question)
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
            gemini_response = await self._get_gemini_response(question)
            responses.append({
                'llm_name': 'Gemini',
                'content': gemini_response['content'],
                'tokens_used': gemini_response['tokens_used']
            })
        except Exception as e:
            responses.append({
                'llm_name': 'Gemini',
                'content': f"Error: {str(e)}",
                'tokens_used': 0
            })

        return responses

    async def _get_gpt4_response(self, question: str) -> Dict:
        """Get response from GPT-4."""
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": question}
            ]
        )
        return {
            'content': response.choices[0].message.content,
            'tokens_used': response.usage.total_tokens
        }

    async def _get_claude_response(self, question: str) -> Dict:
        """Get response from Claude."""
        response = await self.anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return {
            'content': response.content[0].text,
            'tokens_used': response.usage.input_tokens + response.usage.output_tokens
        }

    async def _get_gemini_response(self, question: str) -> Dict:
        """Get response from Google Gemini."""
        response = await self.gemini_model.generate_content(question)
        return {
            'content': response.text,
            'tokens_used': response.candidates[0].token_count if hasattr(response, 'candidates') else 0
        } 