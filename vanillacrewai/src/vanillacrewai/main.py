# --- Monkey Patch CrewAI's LLM.call to intercept messages ---
import crewai.llm
from datetime import datetime

original_llm_call = crewai.llm.LLM.call

def intercepted_llm_call(self, messages, *args, **kwargs):
    # Ensure messages is a list of dicts
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    # Print intercepted messages
    print("\n🧠 [Intercepted LLM Messages]:")
    print("=" * 60)
    for msg in messages:
        print(f"[{msg.get('role', 'user')}] {msg.get('content', '')}")
    print("=" * 60)

    # Proceed as normal
    return original_llm_call(self, messages, *args, **kwargs)

# Apply the monkey patch
crewai.llm.LLM.call = intercepted_llm_call

# --- Import and run your Crew ---
from src.vanillacrewai.crew import OpportunityInsightCrew

def run():
    print("🚀 Launching OpportunityInsightCrew...")

    crew_instance = OpportunityInsightCrew()

    inputs = {
        "topic": "AI in Personalized Fitness and Nutrition Coaching",  
        "current_year": datetime.now().year
    }

    result = crew_instance.crew().kickoff(inputs=inputs)

    print("\n✅ Final Result:")
    print(result)
    return result

if __name__ == "__main__":
    run()