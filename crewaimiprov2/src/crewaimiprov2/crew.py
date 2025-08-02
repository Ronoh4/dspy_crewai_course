from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class StartupValidatorCrew():
    """Crew that validates startup ideas and recommends next steps"""

    @agent
    def market_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['market_researcher'],
            verbose=True,
            llm="anthropic/claude-sonnet-4-20250514"
        )

    @agent
    def startup_coach(self) -> Agent:
        return Agent(
            config=self.agents_config['startup_coach'],
            verbose=True,
            llm="anthropic/claude-sonnet-4-20250514"
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            output_file='market_research.md'
        )

    @task
    def coaching_task(self) -> Task:
        return Task(
            config=self.tasks_config['coaching_task'],
            input=self.research_task,
            output_file='startup_advice.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the startup validation crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
