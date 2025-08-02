from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class OpportunityInsightCrew():
    """Crew that explores opportunities and reports insights for a given topic"""

    @agent
    def opportunity_explorer(self) -> Agent:
        return Agent(
            config=self.agents_config['opportunity_explorer'],
            verbose=True,
            llm="anthropic/claude-sonnet-4-20250514"
        )

    @agent
    def insight_reporter(self) -> Agent:
        return Agent(
            config=self.agents_config['insight_reporter'],
            verbose=True,
            llm="anthropic/claude-sonnet-4-20250514"
        )

    @task
    def exploration_task(self) -> Task:
        return Task(
            config=self.tasks_config['exploration_task'],
            output_file='market_opportunities.md'
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            input=self.exploration_task,
            output_file='opportunity_report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the opportunity insight generation crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
