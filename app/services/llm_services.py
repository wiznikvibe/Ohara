from google import genai
from groq import Groq
from app.config import settings

gemini_client = genai.Client(api_key=settings.gemini_api_key)
groq_client = Groq(api_key=settings.groq_api_key)

def generate(system_prompt: str, user_message: str, model: str | None = None, temperature: float = 0.0)-> dict:
    try:    
        if model is None:
            model = settings.llm_model_answer
        
        response = gemini_client.models.generate_content(
            model=model,
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": user_message}]
                }
            ],
            system=system_prompt,
            temperature=temperature
        )
        text = response.text

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage.total_tokens else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage.total_tokens else 0,
            "total_tokens": response.usage.total_tokens if response.usage.total_tokens else 0,
        }

        return {
            "text": text,
            "usage": usage
        }
    except Exception as e:
        print(f"Error generating response: {e}")
        return {
            "text": "",
            "usage": {}
        }

def generate_with_json(system_prompt: str, user_message: str, model: str | None = None, temperature: float = 0.0)-> dict:
    try:
        if model is None:
            model = settings.llm_model_grader

        response = gemini_client.models.generate_content(
            model=model,
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": user_message}]
                }
            ],
            system=system_prompt,
            temperature=temperature,
            config={"response_mime_type": "application/json"},
        )

        text = response.text

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage.total_tokens else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage.total_tokens else 0,
            "total_tokens": response.usage.total_tokens if response.usage.total_tokens else 0,
        }

        return {
            "text": text,
            "usage": usage
        }
    
    except Exception as e:
        print(f"Error generating response: {e}")
        return {
            "text": "",
            "usage": {}
        }
 