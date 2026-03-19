import logging
import os

from agent_framework import ToolMode
from agent_framework.azure import AzureAIClient
from agent_framework_devui import serve
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
        tool_choice=ToolMode(mode="auto"),
        tools=[timePerIssueTools.calculate_time_based_on_complexity]
    )

    github_client = AzureAIClient(**settings)
    github_mcp_tool = github_client.get_mcp_tool(
        name="GitHub MCP",
        url="https://api.githubcopilot.com/mcp",
        approval_mode="never_require",
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_MCP_PAT']}",
        },
    )

    github_agent = github_client.as_agent(
        name="GitHubAgent",
        instructions=f"""
            You are a helpful assistant that can create an issue on the user's GitHub repository based on the input provided.
            To create the issue, use the GitHub MCP tool.
            You work on this repository: {os.environ["GITHUB_REPOSITORY"]}
        """,
        tools=[github_mcp_tool]
    )

    serve(entities=[issue_analyzer_agent, github_agent],
          port=8090, auto_open=True)


if __name__ == "__main__":
    main()
