from google import genai
from google.genai import types
from groq import Groq

from app.config import settings

gemini_client = genai.Client(api_key=settings.gemini_api_key)
groq_client = Groq(api_key=settings.groq_api_key)


def _usage_from_gemini(response) -> dict:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return {}
    return {
        "prompt_tokens": getattr(meta, "prompt_token_count", 0) or 0,
        "completion_tokens": getattr(meta, "candidates_token_count", 0) or 0,
        "total_tokens": getattr(meta, "total_token_count", 0) or 0,
    }


def generate(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    try:
        if model is None:
            model = settings.llm_model_answer

        response = gemini_client.models.generate_content(
            model=model,
            contents=user_message,  # plain string is fine
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
            ),
        )
        return {"text": response.text or "", "usage": _usage_from_gemini(response)}
    except Exception as e:
        print(f"Error generating response: {e}")
        return {"text": "", "usage": {}}


def generate_with_json(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    try:
        if model is None:
            model = settings.llm_model_grader

        # Grader id looks like a Groq model — route accordingly
        if model.startswith("openai/") or "gpt-oss" in model:
            completion = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            usage = completion.usage
            return {
                "text": completion.choices[0].message.content or "",
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                },
            }

        response = gemini_client.models.generate_content(
            model=model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        return {"text": response.text or "", "usage": _usage_from_gemini(response)}
    except Exception as e:
        print(f"Error generating response: {e}")
        return {"text": "", "usage": {}}