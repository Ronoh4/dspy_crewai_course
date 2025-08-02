# 🧠 dspy_crewai_course  
An instructional repo for learning DSPy + CrewAI integration.

---

## 🧠 CrewAI + DSPy Prompt Optimization Course Resources

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
- How to run full end-to-end optimized workflows with `MIPROv2`

---

## 🛠️ How to Clone and Run

### 📦 Requirements

- Python 3.10+
- Poetry or `pip`
- API Keys (see below)

---

### 📥 Clone the Repository

```bash
git clone https://github.com/Ronoh4/dspy_crewai_course.git
cd dspy_crewai_course

📦 Install Dependencies
Using Poetry:

bash

poetry install
poetry shell
Or using pip:

bash

pip install -r requirements.txt
🔑 Environment Variables
Create a .env file in the root with the following contents depending on the language model(s) you are using:

env
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
MISTRAL_API_KEY=your_mistral_api_key

📚 Folder Structure and What Each Teaches
📁 vanillacrewai/ – Basic CrewAI Prompt Interception
Demonstrates how to intercept CrewAI-generated prompts using monkey-patching. It prints system and user prompts before they are sent to the LLM.

🧪 Useful for debugging and understanding how CrewAI builds prompts from YAML configs.

📁 dspyintro/ – Introduction to DSPy Framework
Introduces DSPy using the BootstrapFewShot optimizer. You’ll learn to:

Define raw prompts

Create training and dev sets

Run few-shot optimization cycles

🧠 This section is LLM-agnostic and teaches the core of DSPy's capabilities.

📁 crewaibootstrap/ – DSPy + CrewAI Integration
This shows how to combine CrewAI and DSPy by:

Intercepting CrewAI prompts

Optimizing them with DSPy

Reinjecting the improved prompts into CrewAI’s LLM flow

🔄 It bridges vanilla CrewAI with prompt optimization techniques.

📁 crewaimiprov2/ – Advanced Optimization with MIPROv2
Demonstrates using DSPy's MIPROv2 for optimizing full CrewAI workflows. You'll:

Work with long-form prompts

Optimize using training + dev examples

Reintegrate the optimized prompts into the CrewAI tasks

🧬 This is the most advanced example showing full-cycle LLM prompt optimization.

📦 Key Library Versions
Library	Version
crewai	0.152.0
dspy	2.6.27

🧵 Feedback and Contributions
This repository is for educational purposes to help learners understand LLM prompt engineering and agent task optimization.

Feel free to fork, improve, and submit a pull request!

📄 License
This repository is licensed under the MIT License.

📄 License
This repository is licensed under the MIT License.
