# Import modules and libraries
import dspy
from dotenv import load_dotenv
import os
import crewai.llm
from typing import List, Dict, Union, Callable

# --- Configuration ---
# Loading environmental variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")

# Configure DSPy with Claude 3 Opus for the main task LLM
main_task_lm = dspy.LM('anthropic/claude-3-opus-20240229', api_key=ANTHROPIC_API_KEY, cache=True)
dspy.configure(lm=main_task_lm)
print("DSPy is configured with Claude 3 Opus for main task optimization.")

# Configure a separate Claude Sonnet 4 for the AI-assisted metric (you can use the same llm for main task and metric)
assess_lm = dspy.LM('anthropic/claude-sonnet-4-20250514', api_key=ANTHROPIC_API_KEY, cache=True)
print("DSPy metric LLM is configured with Claude Sonnet 4.")

# --- DSPy Components for Prompt Optimization ---
# Define the NEW Signature for Prompt Optimization
class ImprovePrompt(dspy.Signature):
    """
    Transforms a raw, generic, or vaguely defined prompt constructed by CrewAI prompt engine into a highly structured,
    explicit, and effective prompt designed for optimal performance by an LLM.
    This prompt optimization can include enhancing clarity, adding specific constraints, defining desired output formats,
    and providing illustrative examples to guide the LLM's response generation.
    """
    crewai_prompt: str = dspy.InputField(
        desc="The initial, often loosely defined, or boilerplate-laden CrewAI prompt with system and user parts. "
             "This prompt typically provides a general task description, basic persona, and minimal output expectations, "
             "and is stitched together by CrewAI's prompt construction engine from the YAML description of agents and tasks plus inputs and a few prompt template (boilerplate) lines."
    )
    dspy_improved_prompt: str = dspy.OutputField(
        desc="A meticulously crafted and expanded version of the CrewAI prompt."
             "This improved prompt has enhanced clarity and adds specific constraints like incorporates explicit instructions for the LLM's role, objectives, context, "
             "detailed task directives, precise requirements (e.g., length, content constraints), "
             "clear formatting guidelines, and relevant examples among others. "
             "The aim is to eliminate ambiguity and provide comprehensive guidance for generating a high-quality, "
             "predictable, and relevant output from the LLM."
    )

# Create a simple custom module using the new signature
# This is the module that DSPy will optimize to perform the prompt improvement task
class PromptOptimizerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.improver = dspy.Predict(ImprovePrompt)

    def forward(self, crewai_prompt: str):
        return self.improver(crewai_prompt=crewai_prompt)

basic_prompt_module = PromptOptimizerModule()

# --- Create Train and Dev Sets for Prompt Improvement ---
trainset = [
    dspy.Example(
        crewai_prompt=(
            "[System]: You are a Market Opportunity Explorer for AI in Personalized Fitness and Nutrition Coaching.\n"
            "A strategic thinker who collaborates with early-stage startups to uncover niche opportunities.\n"
            "Experienced in using market signals, customer behavior, and unmet pain points to guide innovation.\n\n"
            "Your personal goal is: Identify underserved needs, emerging demands, and untapped segments within "
            "the AI in Personalized Fitness and Nutrition Coaching domain\n\n"
            "To give my best complete final answer to the task respond using the exact following format:\n\n"
            "Thought: I now can give a great answer.\n\n"
            "[User]: Current Task: Identify gaps and emerging opportunities in the "
            "AI in Personalized Fitness and Nutrition Coaching space using up-to-date 2025 data.\n\n"
            "This is the expected criteria for your final answer: A list of 8-10 market gaps or opportunities with 1-2 lines of context each.\n\n"
            "You MUST return the actual complete content as the final answer, not a summary.\n\n"
            "Begin! This is VERY important to you, use the tools available and give your best Final Answer, your job depends on it! "
        ),
        dspy_improved_prompt=(
            "[System]: ROLE: Market Opportunity Analyst\n"
            "DOMAIN: AI for Personalized Fitness and Nutrition Coaching\n"
            "OBJECTIVE: Identify strategic market gaps, emerging demands, and underserved segments.\n"
            "CONTEXT:\n"
            "- You are a domain expert who advises early-stage startups.\n"
            "- You use behavioral trends, market signals, and customer pain points.\n"
            "- Your output guides innovation and product strategy.\n"
            "[User]: TASK: Generate a list of 8-10 specific market opportunities in the domain of AI for Personalized Fitness and Nutrition Coaching.\n\n"
            "REQUIREMENTS:\n"
            "- Each opportunity must include:\n"
            "  • A clear, short title\n"
            "  • 1-2 lines of explanation grounded in user behavior, trends, or unmet needs\n"
            "- Avoid vague or overly general phrasing\n"
            "- Do not summarize or give analysis before or after the list\n\n"
            "FORMAT:\n"
            "1. [Opportunity Title] - Contextual explanation\n"
            "2. ...\n\n"
            "EXAMPLES:\n"
            "1. Personalized Menopause Coaching - A growing demand among middle-aged women for tailored fitness and nutrition plans that adapt to hormonal changes.\n"
            "2. AI Meal Planning for Gut Health - Rising interest in gut microbiome optimization creates a need for personalized meal recommendations based on digestive data.\n\n"
            "Begin your list now. Return only the structured output."
        )
    ).with_inputs("crewai_prompt"),

    dspy.Example(
        crewai_prompt=(
            "[System]: You are a Cybersecurity Consultant for Small Businesses. Your goal is to help small businesses "
            "with their cybersecurity needs. Your backstory: You're an expert who understands small business "
            "challenges and how to keep their data safe.\n\n"
            "To give my best complete final answer to the task respond using the exact following format:\n\n"
            "Thought: I now can give a great answer.\n\n"
            "[User]: Current Task: Advise a small business on common cybersecurity threats they face and how to protect themselves. "
            "Your expected output is a list of three threats and simple solutions.\n\n"
            "You MUST return the actual complete content as the final answer, not a summary.\n\n"
            "Begin! This is VERY important to you, use the tools available and give your best Final Answer, your job depends on it! "
        ),
        dspy_improved_prompt=(
            "[System]: **Consultant Profile:** Cybersecurity Specialist for Small & Medium Enterprises (SMEs).\n"
            "**Core Mission:** Provide concise, actionable security guidance tailored for businesses with limited IT resources.\n"
            "**Key Expertise:** Risk assessment, incident response planning, and practical security tool implementation.\n" 
            "**Target Audience:** Non-technical business owners and staff requiring clear, understandable advice.\n\n"
            "[User]: **Assignment:** Detail 4 common cybersecurity threats relevant to a typical small business, each with 2 practical mitigation steps.\n\n"
            "**Output Directives:**\n"
            "- Focus on immediately implementable and high-impact solutions.\n"
            "- Ensure advice is current and avoids jargon.\n"
            "- Present as a clear, numbered list.\n\n"
            "**Format Example:**\n"
            "1. **Threat:** [Threat Name]\n"
            "   **Mitigation:** [Actionable Step 1]; [Actionable Step 2]\n\n"
            "**Illustrative Entry:**\n"
            "1. **Threat:** Phishing and Spear Phishing Attacks\n"
            "   **Mitigation:** Conduct mandatory, regular employee training on identifying suspicious emails; Implement email gateway security with robust anti-phishing filters.\n\n"
            "Generate the cybersecurity threat guide now. Only the formatted list should be returned."
        )
    ).with_inputs("crewai_prompt"),

    dspy.Example(
        crewai_prompt=(
            "[System]: You are a Technology Risk Analyst whose goal is to look at new tech and find problems. "
            "Your backstory is that you are good at spotting risks in new digital systems and you specialize in IoT devices. \n\n"
            "To give my best complete final answer to the task respond using the exact following format:\n\n"
            "Thought: I now can give a great answer.\n\n"
            "[User]: Current Task: Identify security and privacy risks for deploying IoT in a smart home setting. "
            "Expected output is a list of 3 risks.\n\n"
            "This is the expected criteria for your final answer: Just a simple list of risks.\n\n"
            "You MUST return the actual complete content as the final answer, not a summary.\n\n"
            "Begin! This is VERY important to you, use the tools available and give your best Final Answer, your job depends on it! "
        ),
        dspy_improved_prompt=(
            "[System]:\n"
            "**Analyst Role:** IoT Security and Privacy Risk Assessment Specialist.\n"
            "**Primary Objective:** Systematically identify and articulate potential vulnerabilities in nascent IoT deployments.\n"
            "**Domain Expertise:** Comprehensive understanding of IoT device architecture, data flow, and emerging threat landscapes.\n"
            "**Guiding Ethos:** Proactive identification and concise articulation of risks to enable robust mitigation strategies.\n\n"
            "[User]:\n"
            "**Task Directive:** Compile a list of 3 critical security and privacy risks associated with the deployment of Internet of Things (IoT) devices within a typical smart home environment.\n\n"
            "**Content Specifications:**\n"
            "- Each risk must be clearly named and explained in 2 sentences, highlighting the potential impact.\n"
            "- Focus on risks inherent to consumer IoT devices and their interconnected nature.\n"
            "- Categorize risks for clarity (e.g., Data Security, Device Vulnerabilities, Privacy Concerns).\n\n"
            "**Expected Format:**\n"
            "## IoT Smart Home Risk Assessment\n"
            "\n"
            "### [Risk Category 1]\n"
            "- **[Risk Name 1]:** [Concise explanation of risk and impact]\n"
            "- **[Risk Name 2]:** [Concise explanation of risk and impact]\n"
            "\n"
            "### [Risk Category 2]\n"
            "- **[Risk Name 3]:** [Concise explanation of risk and impact]\n"
            "\n"
            "**Illustrative Entry:**\n"
            "## IoT Smart Home Risk Assessment\n"
            "\n"
            "### Device Vulnerabilities\n"
            "- **Weak Default Passwords:** Many IoT devices ship with easily guessable or hardcoded credentials, making them prime targets for unauthorized access and botnet recruitment.\n"
            "- **Unpatched Firmware:** Manufacturers often fail to provide timely security updates, leaving devices vulnerable to known exploits long after discovery.\n\n"
            "Generate the comprehensive risk assessment now. Provide only the markdown content, strictly adhering to the specified structure."
        )
    ).with_inputs("crewai_prompt"),
]

devset = [
    dspy.Example(
        crewai_prompt=(
            "[System]: You are a Climate Risk Analyst specializing in urban environments. Your expertise lies in using AI models "
            "to evaluate environmental risks and guide decision-making for city planners.\n\n"
            "To give my best complete final answer to the task respond using the exact following format:\n\n"
            "Thought: I now can give a great answer.\n\n"
            "[User]: Current Task: Use 2025 climate data to identify 3 climate-related risks "
            "that should influence the design of a coastal urban infrastructure project.\n\n"
            "This is the expected criteria for your final answer: A list of 5 specific climate risks with 1-2 lines of explanation each.\n\n"
            "You MUST return the actual complete content as the final answer, not a summary.\n\n"
            "Begin! This is VERY important to you, use the tools available and give your best Final Answer, your job depends on it!"
        ),
        dspy_improved_prompt=(
            "[System]:\n"
            "**Role:** Urban Climate Risk Analyst\n"
            "**Objective:** Surface actionable climate risks affecting city infrastructure design in coastal areas.\n"
            "**Domain:** AI-Driven Environmental Modeling\n\n"
            "[User]:\n"
            "**Task Directive:** List 5 critical climate risks that urban planners must consider when designing infrastructure for a coastal city.\n\n"
            "**Requirements:**\n"
            "- Each risk should be named clearly and briefly explained based on projected 2025 climate models.\n"
            "- Focus on risks relevant to construction, public safety, and city sustainability.\n"
            "- Do not include policy commentary or mitigation strategies.\n\n"
            "**Format:**\n"
            "1. [Risk Name] - [Short impact explanation]\n"
            "2. ...\n\n"
            "**Illustrative Entry:**\n"
            "1. Sea Level Rise - Coastal flooding events are expected to increase in frequency and severity due to rising tides and extreme weather.\n"
            "2. Heat Island Intensification - Dense urban areas will experience prolonged heat waves that strain public health and power systems.\n\n"
            "Generate your list of climate risks now. Only return the structured markdown list."
        )
    ).with_inputs("crewai_prompt"),

    dspy.Example(
        crewai_prompt=(
            "[System]: You are a Mental Health Resource Curator. Your goal is to find and share resources for people who need mental health support. "
            "Your backstory: You are good at finding reliable info and tools.\n\n"
            "To give my best complete final answer to the task respond using the exact following format:\n\n"
            "Thought: I now can give a great answer.\n\n"
            "[User]: Current Task: Provide resources for managing a specific mental health challenge. " # Generic task
            "Expected output: A list of resources.\n\n"
            "This is the expected criteria for your final answer: A list of resources for the specified challenge.\n\n"
            "You MUST return the actual complete content as the final answer, not a summary.\n\n"
            "Begin! This is VERY important to you, use the tools available and give your best Final Answer, your job depends on it! "
        ),
        dspy_improved_prompt=(
            "[System]:\n"
            "ROLE: Mental Wellness Resource Specialist\n"
            "DOMAIN: Mental Health Support and Stress Management\n"
            "OBJECTIVE: Curate and present accessible, evidence-based resources to improve mental well-being.\n"
            "CONTEXT:\n"
            "- You specialize in identifying credible tools, techniques, and programs for laypeople.\n"
            "- Your audience includes individuals seeking practical self-help options.\n"
            "- Resources should be current, actionable, and diverse in approach.\n\n"
            "[User]: TASK: List 4 high-quality resources or strategies for managing stress, tailored to the general public.\n\n"
            "REQUIREMENTS:\n"
            "- Each entry must include a name and a brief description (1-2 sentences).\n"
            "- Include at least one digital tool (e.g., app or website) and one offline method.\n"
            "- Ensure descriptions highlight how the resource helps with stress.\n\n"
            "FORMAT:\n"
            "- **Resource 1:** [Name] - [Description]\n"
            "- **Resource 2:** [Name] - [Description]\n"
            "- ...\n\n"
            "SAMPLE ENTRY:\n"
            "- **Resource 1:** Calm App - This mobile app offers guided meditations and breathing exercises to reduce stress and promote relaxation.\n"
            "- **Resource 2:** Journaling - Writing daily thoughts and feelings helps process emotions, lowering stress levels over time.\n\n"
            "Generate the resource list now. Provide only the formatted output."
        )
    ).with_inputs("crewai_prompt"),
]

# --- Define an AI-Assisted Metric ---
class AssessPromptImprovement(dspy.Signature):
    """Assess the quality of an improved prompt."""
    assessed_raw_prompt: str = dspy.InputField(desc="The original crewai prompt before improvement.")
    assessed_improved_prompt: str = dspy.InputField(desc="The prompt that was generated by the DSPy module based on the crewai prompt.")
    assessment_question: str = dspy.InputField(desc="The specific question to evaluate the dspy improved prompt.")
    assessment_answer: bool = dspy.OutputField(desc="True if the improved prompt meets the criteria, False otherwise.")

# Define a metric function to evaluate the improved prompt
def prompt_improvement_metric(example, pred, trace=None):
    """
    Evaluates if the predicted improved prompt (pred.dspy_improved_prompt) effectively transforms
    the crewai prompt (example.crewai_prompt) into a high-quality, actionable, and well-structured prompt,
    consistent with the standards set by the example.dspy_improved_prompt.
    """

    # Set the variable names for the raw and improved prompt
    raw_prompt = example.crewai_prompt # This is the original CrewAI prompt from the training examples
    expected_improved_prompt = example.dspy_improved_prompt  # This is the gold standard dspy improved prompt from the training examples
    predicted_improved_prompt = pred.dspy_improved_prompt # This is the improved prompt generated by the DSPy module

    # Metric 1: Structural Integrity & Clarity
    # Does the generated improved prompt follow a clear, well-structured format
    # and is it easy to parse for the LLM, similar to the golden examples?
    q1 = f"""
    Assess if the 'predicted_improved_prompt' successfully clarifies and structures the original 'raw_prompt' by using effective formatting and clear instructions.

    - **Raw Prompt:**
    ---
    {raw_prompt}
    ---
    
    - **Predicted Prompt:**
    ---
    {predicted_improved_prompt}
    ---

    - **Expected Prompt (Ideal):**
    ---
    {expected_improved_prompt}
    ---

    Considering the raw prompt's content, is the predicted prompt's format, use of headings, and overall clarity as good as the ideal expected prompt's? Answer True or False.
    """

    # Metric 2: Instructional Completeness & Specificity
    # Does the generated dspy improved prompt fully elaborate on the generic requirements
    # from the crewai prompt by adding specific constraints, examples, or detailed instructions,
    # making it more effective for the LLM to generate the desired output, as demonstrated by the golden examples?
    q2 = f"""
    Evaluate the instructional completeness and specificity of the 'predicted_improved_prompt'.
    
    Raw Prompt:
    ---
    {raw_prompt}
    ---
    
    Predicted Improved Prompt:
    ---
    {predicted_improved_prompt}
    ---
    
    Compared to the comprehensive and specific instructions of the 'expected_improved_prompt':
    ---
    {expected_improved_prompt}
    ---
    
    Does the 'predicted_improved_prompt' sufficiently elaborate on the generic requirements from the 'raw_prompt' by adding specific constraints, clear formatting guidance, and concrete examples to ensure the LLM generates the *desired final output* effectively, mirroring the thoroughness of the 'expected_improved_prompt'?
    Respond with True or False.
    """

    with dspy.context(lm=assess_lm):
        clarity_eval = dspy.Predict(AssessPromptImprovement)(
            assessed_raw_prompt=raw_prompt, # Pass raw_prompt for context
            assessed_improved_prompt=predicted_improved_prompt,
            assessment_question=q1
        )
        completeness_eval = dspy.Predict(AssessPromptImprovement)(
            assessed_raw_prompt=raw_prompt, # Pass raw_prompt for context
            assessed_improved_prompt=predicted_improved_prompt,
            assessment_question=q2
        )

    score = int(clarity_eval.assessment_answer) + int(completeness_eval.assessment_answer)

    # Based on the eval signature, the metric is set to return a bool, True/False for each example
    # Here, we can return True if both criteria for q1 and q2 are met.
    if trace is not None:
        # This part of the code is empty, but it's where you would add debugging.
        pass 
    return score == 2 # Return True if both criteria are met, False otherwise


# --- Optimize with BootstrapFewShot ---

# First we need to save the optimized module to avoid re-optimization every time we run with a new topic
# Define the file path for saving the optimized module
OPTIMIZED_MODULE_FILE = "optimized_prompt_module.json"

# Import the BootstrapFewShot optimizer
from dspy.teleprompt import BootstrapFewShot

def optimize_and_get_module_bootstrap():
    """
    Optimizes the PromptOptimizerModule using BootstrapFewShot,
    saves it, and returns the optimized module.
    """
    config = dict(max_labeled_demos=3)  # Limit to 3 labeled demos for bootstrapping
    teleprompter = BootstrapFewShot(
        metric=prompt_improvement_metric,
        **config,  # Unpack the configuration dictionary
    )

    print("\n🚀 Starting BootstrapFewShot optimization for prompt improvement...")
    optimized_module_result = teleprompter.compile( # <--- This returns the optimized module
        basic_prompt_module, # <--- This is the starting point for optimization
        trainset=trainset # this is the training examples we provided
    )
    print("✅ Module optimized with BootstrapFewShot.")

    # Save the optimized module
    optimized_module_result.save(OPTIMIZED_MODULE_FILE)
    print(f"💾 Optimized module saved to: {OPTIMIZED_MODULE_FILE}")


    # Evaluate on dev set
    dev_scores = []
    for example in devset:
        prediction = optimized_module_result(crewai_prompt=example.crewai_prompt)
        score = prompt_improvement_metric(example, prediction)  # Use the same AI-assisted metric
        dev_scores.append(score)

    avg_dev_score = sum(dev_scores) / len(dev_scores)
    print(f"\nAverage dev score (BootstrapFewShot AI feedback metric): {avg_dev_score:.2f}")

    return optimized_module_result

# --- CrewAI Monkey Patch with Optimized DSPy Module ---

# Store the original method once
_original_llm_call = crewai.llm.LLM.call

def create_patched_llm_call_function(optimized_dspy_module: dspy.Module) -> Callable:
    def patched_llm_call_inner(self, messages: Union[str, List[Dict[str, str]]], *args, **kwargs):
        # Ensure messages is a list of dicts.
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Print messages BEFORE DSPy optimization
        print("\n🟦 [Monkey Patch] Messges before DSPy Optimization:")
        print("=" * 60)
        for msg in messages:
            print(f"[{msg.get('role', 'user')}] {msg.get('content', '')}")
        print("=" * 60)

        optimized_messages = []
        for msg in messages:
            try:
                # Use the optimized_dspy_module captured by the closure.
                # Here, we pass the ENTIRE message content (system or/and user) to DSPy for optimization.
                improved = optimized_dspy_module(crewai_prompt=msg["content"])
                improved_content = improved.dspy_improved_prompt.strip()
                optimized_messages.append({"role": msg["role"], "content": improved_content})
            except Exception as e:
                print(f"⚠️ Error optimizing message with DSPy: {e}. Keeping original content for role '{msg.get('role')}'.")
                optimized_messages.append(msg)

        # Print messages AFTER DSPy optimization
        print("\n🟦 [Monkey Patch] Improved Prompt Sent to LLM after DSPy Optimization:")
        print("=" * 60)
        for msg in optimized_messages:
            print(f"[{msg.get('role', 'user')}] {msg.get('content', '')}")
        print("=" * 60)

        # Call the original LLM.call method, ensuring 'self' remains the original CrewAI LLM instance
        return _original_llm_call(self, optimized_messages, *args, **kwargs)
    return patched_llm_call_inner

# --- Global Cache ---
optimized_module = None

def run():
    global optimized_module # Declare intent to modify the global variable

    # This block ensures optimized_module is set once, either by loading or optimizing
    if optimized_module is None: # Check if it's already set from a previous call in the same session
        print("⚙️ Optimizing or loading DSPy module...")
        if os.path.exists(OPTIMIZED_MODULE_FILE):
            print(f"📦 Loading optimized module from {OPTIMIZED_MODULE_FILE}...")
            # Instantiate the module type *before* loading
            temp_module = PromptOptimizerModule()
            temp_module.load(OPTIMIZED_MODULE_FILE)
            optimized_module = temp_module # Assign the loaded module to the global variable
        else:
            optimized_module = optimize_and_get_module_bootstrap() # This function also saves the module
    else:
        print("✅ Reusing cached DSPy module...") # This message happens if run() is called multiple times in one script execution

    # Define inputs BEFORE the print statement that uses it
    inputs = {
        "topic": "Kenyan couple going to Netherlands for 5 days"  # Change as needed for your tests
    }

    # Now, `optimized_module` should definitively hold the loaded/optimized module before this print statement.
    is_optimized_applied = (optimized_module is not None)
    print(f"\n🚀 Kicking off CrewAI with topic: {inputs['topic']} (using {'optimized' if is_optimized_applied else 'original'} prompts)...")


    custom_patched_llm_call = create_patched_llm_call_function(optimized_module)
    crewai.llm.LLM.call = custom_patched_llm_call

    from src.crewaibootstrap.crew import BootStrapCrew
    crew_instance = BootStrapCrew()

    result = crew_instance.crew().kickoff(inputs=inputs)

    print("\n✅ Final Result:")
    print(result)
    return result

if __name__ == "__main__":
    run()