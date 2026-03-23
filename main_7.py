import logging
import os

from agent_framework import ToolMode
from agent_framework.azure import AzureAIClient
from agent_framework.orchestrations import GroupChatBuilder
from agent_framework_devui import serve
from agent_framework_orchestrations import SequentialBuilder
from azure.ai.projects.models import MCPTool
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

from models.issue_analyzer import IssueAnalyzer
from tools.time_per_issue_tools import TimePerIssueTools

load_dotenv()


def main():
    logging.basicConfig(level=logging.ERROR, format="%(message)s")

    credential = AzureCliCredential()
    settings = {
        "project_endpoint": os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        "model_deployment_name": os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        "credential": credential,
    }
    client = AzureAIClient(**settings)
    timePerIssueTools = TimePerIssueTools()

    issue_analyzer_agent = client.as_agent(
        instructions="""
                        You are analyzing issues.
                        If the ask is a feature request the complexity should be 'NA'.
                        If the issue is a bug, analyze the stack trace and provide the likely cause and complexity level.
                    """,
        name="IssueAnalyzerAgent",
        default_options={"response_format": IssueAnalyzer},
        tools=[timePerIssueTools.calculate_time_based_on_complexity]
    )

    github_tool = MCPTool(
        server_label="GitHub",
        server_url="https://api.githubcopilot.com/mcp",
        require_approval="always",
        project_connection_id="GitHub",
    )

    github_agent = AzureAIClient(**settings).as_agent(
        name="GitHubAgent",
        instructions=f"""
            You are a helpful assistant that can create an issue on the user's GitHub repository based on the input provided.
            To create the issue, use the GitHub MCP tool.
            You work on this repository: {os.environ["GITHUB_REPOSITORY"]}
        """,
        tools=[github_tool]
    )

    workflow = GroupChatBuilder(
        participants=[issue_analyzer_agent, github_agent],
        intermediate_outputs=True,
        orchestrator_agent=AzureAIClient(**settings).as_agent(
            name="IssueCreationGroupChatWorkflow",
            instructions="""
                You are a workflow manager that helps create GitHub issues based on user input.
                First, analyze the input using the Issue Analyzer Agent to determine the issue type, likely cause, and complexity.
                If an issue requires additional information from documentation, ask other specialized agents.
                Finally, create a GitHub issue using the GitHub Agent with the analyzed information.
            """,
        ),
    ).build()

    ms_learn_client = AzureAIClient(**settings)
    ms_learn_mcp_tool = ms_learn_client.get_mcp_tool(
        name="Microsoft Learn MCP",
        url="https://learn.microsoft.com/api/mcp",
        description="A Microsoft Learn MCP server for documentation questions",
        approval_mode="never_require",
    ),

    ms_learn_agent = ms_learn_client.as_agent(
        name="DocsAgent",
        instructions="""
            You are a helpful assistant that can help with Microsoft documentation questions.
            Provide accurate and concise information based on the documentation available.
        """,
        tools=ms_learn_mcp_tool,
    )

    group_workflow_agent = workflow.as_agent(
        name="IssueCreationAgentGroup"
    )

    sequential_workflow = SequentialBuilder(
        participants=[ms_learn_agent, group_workflow_agent]).build()

    serve(entities=[issue_analyzer_agent, github_agent, workflow, ms_learn_agent, sequential_workflow],
          port=8090, auto_open=True)


if __name__ == "__main__":
    main()
