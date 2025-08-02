from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class BootStrapCrew():
    """Crew that plans international travel itineraries and checks visa requirements"""

    @agent
    def travel_planner(self) -> Agent:
        return Agent(
            config=self.agents_config['travel_planner'],
            verbose=True,
            # Configure LLM directly as requested
            llm="anthropic/claude-sonnet-4-20250514"
        )

    @agent
    def visa_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['visa_expert'],
            verbose=True,
            # Configure LLM directly as requested
            llm="anthropic/claude-sonnet-4-20250514"
        )

    @task
    def itinerary_creation_task(self) -> Task:
        return Task(
            config=self.tasks_config['itinerary_creation_task'],
            output_file='travel_itinerary.md' # Example output file
        )

    @task
    def visa_check_task(self) -> Task:
        return Task(
            config=self.tasks_config['visa_check_task'],
            input=self.itinerary_creation_task,
            output_file='visa_report.md' # Example output file
        )

    @crew
    def crew(self) -> Crew:
        """Creates the international travel planning crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )