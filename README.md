# dspy_crewai_course
An instructional repo for learning dspy+crewai integration

# 🧠 CrewAI + DSPy Prompt Optimization Course Resources

This repository contains the complete codebase used in the course **"Optimizing CrewAI Prompts with DSPy"**. It is structured in progressive folders to help you learn how to:

- Intercept and analyze CrewAI-generated prompts
- Introduce and apply the DSPy framework for prompt optimization
- Integrate DSPy into CrewAI workflows
- Use advanced optimizers like MIPROv2 to enhance prompt effectiveness

---

## 🚀 What You’ll Learn

- How CrewAI internally builds prompts and how to intercept them
- How to use DSPy's `BootstrapFewShot` module for few-shot optimization
- How to integrate DSPy modules into CrewAI tasks and agents
- How to run full end-to-end optimized workflows with MIPROv2

---

## 🛠️ How to Clone and Run

### 📦 Requirements

- Python 3.10+
- Poetry or `pip` for dependency management
- API keys (see below)

### 📥 Clone the Repository

```bash
git clone https://github.com/Ronoh4/dspy_crewai_course.git
cd YOUR_REPO_NAME

📦 Install Dependencies
Using Poetry:
```bash
poetry install
poetry shell
Or using pip:
```bash
pip install -r requirements.txt

🔑 Environment Variables
Create a .env file in the root with the following contents depending on the language model you are using:
```bash
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
MISTRAL_API_KEY=...

📚 Folder Structure and What Each Teaches
📁 vanillacrewai/ – Basic CrewAI Prompt Interception
This folder demonstrates how to intercept CrewAI-generated prompts using monkey-patching. It's a minimal setup with a basic CrewAI task flow that prints out the system and user prompts before they're sent to the LLM.

🧪 Useful for debugging and understanding how CrewAI internally builds prompts from YAML configs.

📁 dspyintro/ – Introduction to DSPy Framework
This folder introduces DSPy's optimization capabilities using the BootstrapFewShot module. It shows how to:

Define raw prompts

Create training and dev sets

Run a few-shot optimization cycle to improve prompts

🧠 This section is LLM-agnostic and lays the groundwork for prompt quality improvement.

📁 crewaibootstrap/ – DSPy + CrewAI Integration
Here, you learn how to combine CrewAI with DSPy to improve task and agent prompts. It includes:

A CrewAI workflow with intercepted prompts

DSPy modules wrapping around those prompts

Returning optimized prompts back into the CrewAI flow before LLM call

🔄 This is the bridge between vanilla CrewAI and enhanced optimization.

📁 crewaimiprov2/ – Advanced Optimization with MIPROv2
This folder demonstrates the full power of DSPy's MIPROv2 optimizer on real CrewAI tasks. You'll learn how to:

Prepare longer, structured CrewAI prompts

Optimize them using MIPROv2 with both training and dev sets

Plug the optimized prompts back into CrewAI tasks

🧬 This is the most advanced integration and shows full-cycle optimization with feedback.

📦 Key Library Versions
Library	Version
```bash
crewai	0.152.0
dspy	2.6.27

🧵 Feedback and Contributions
This code is for educational purposes and meant to help learners build a mental model of how LLM prompt pipelines work. If you have suggestions or would like to contribute improvements, feel free to fork and submit a PR.

📄 License
This repository is licensed under the MIT License.
