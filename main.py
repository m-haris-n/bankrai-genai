from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import os
import json
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Configure the Gemini API
model_id = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

# Load transaction history
with open('plaid_transaction_history.json', 'r') as f:
    transaction_history = json.load(f)

app = FastAPI(title="Gemini Chat API")

# Configure CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[ChatMessage]] = None
    search_threshold: float = 0.5  # Default threshold for search grounding

class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        # Create system prompt with transaction history
        system_prompt = f"""You are a Plaid Financial Advisor AI. You have access to the following transaction history:

{json.dumps(transaction_history, indent=2)}

Please analyze the transaction history and provide detailed insights based on the transaction data, including spending patterns, recurring expenses, and any notable financial behaviors."""

        google_search_tool = Tool(
            google_search = GoogleSearch()
        )

        # # Create content array starting with system prompt
        # contents = [{"role": "user", "parts": [{"text": system_prompt}]},
        #             {"role": "model", "parts": [{"text": "I understand. Please provide your question."}]}]

        # Add chat history if provided
        contents = []
        if request.chat_history:
            for msg in request.chat_history:
                # Convert roles to match Gemini's expected format
                role = "user" if msg.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

        # Add current user message
        contents.append({"role": "user", "parts": [{"text": request.message}]})

        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            )
        )

        # for each in response.candidates[0].content.parts:
        #     print(each.text)
        # Example response:
        # The next total solar eclipse visible in the contiguous United States will be on ...

        response_text = ""
        for each in response.candidates[0].content.parts:
            response_text += each.text 
        print("RESPONSE: \n", response_text)
        # To get grounding metadata as web content.
        # print(response.candidates[0].grounding_metadata.search_entry_point.rendered_content)
        
        
        sources = []
        if hasattr(response.candidates[0], 'grounding_metadata'):
            if hasattr(response.candidates[0].grounding_metadata, 'search_entry_point'):
                if hasattr(response.candidates[0].grounding_metadata.search_entry_point, 'rendered_content'):
                    sources.append(response.candidates[0].grounding_metadata.search_entry_point.rendered_content)   
        return ChatResponse(
            response=response_text,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 