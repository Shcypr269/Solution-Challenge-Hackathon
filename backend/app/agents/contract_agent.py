from app.agents.state import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings

class ContractRAG:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model_name,
            temperature=0.0
        )
        
    async def extract_penalties(self, delay_mins: int) -> dict:
        # Mocking Vertex AI Vector Search
        # In actual implementation: embed context -> search vector db -> synthesize
        prompt = f"Given a delay of {delay_mins} minutes, what are the SLA contract penalties?"
        
        # We will mock the output for rapid development
        return {
            "penalty_description": "Delayed deliveries incur a penalty of $50 per hour.",
            "potential_penalty_usd": (delay_mins // 60) * 50
        }

rag = ContractRAG()

async def contract_agent_node(state: AgentState) -> dict:
    print("Agent 4: Analyzing contract via RAG...")
    
    # If no route alternative was generated or no delay, skip penalty extraction
    if not state.get("route_alternatives"):
        return {"contract_intelligence": None}
        
    # Let's assess the original penalty vs new penalty
    original_delay = 120 # static mock for now
    
    # RAG Extraction
    penalties = await rag.extract_penalties(original_delay)
    
    return {
        "contract_intelligence": penalties
    }
